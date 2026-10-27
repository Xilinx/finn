# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Digest-keyed caches and the fetchers that fill them.

Each cache entry is ``<root>/<name>-<digest16>/`` with a marker file holding the
full digest; an entry without a matching marker is incomplete. A fetch builds
the tree in a temporary directory beside the entry, verifies its digest, writes
the marker and renames it into place, under a file lock, so a failed or
concurrent fetch never publishes a partial or unverified tree.
"""
import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from ._declare import PREFIX, ResourceError

MARKER = ".finn-resource"
DEFAULT_SYSTEM_CACHE = "/opt/finn/resources"


def home(environ=None):
    """FINN's per-user directory: FINN_HOME, by default ``~/.finn``.

    It holds fetched resources (``resources/``), the finn_xsi builds (``xsi/``)
    and, natively, the build directory (``build/``); each has its own override.
    """
    env = os.environ if environ is None else environ
    return Path(env.get("FINN_HOME") or Path.home() / ".finn").expanduser()


def scratch(environ=None):
    """FINN's build directory, where builds and tests allocate their working
    directories: FINN_BUILD_DIR, by default ``$FINN_HOME/build``. Absolute, and
    not created here."""
    env = os.environ if environ is None else environ
    value = env.get("FINN_BUILD_DIR")
    return (Path(value).expanduser() if value else home(env) / "build").resolve()


def tree_digest(path):
    """Content digest of a directory tree: sorted relative paths and file hashes.

    Symbolic links contribute their target, not the content they point to. The
    top-level marker file is excluded.
    """
    root = Path(path)
    if not root.is_dir():
        raise ResourceError("tree-not-directory", f"{root} is not a directory")
    items = []
    for directory, dirs, files in os.walk(root):
        # os.walk lists symbolic links to directories in dirs without entering them.
        for name in files + [d for d in dirs if os.path.islink(os.path.join(directory, d))]:
            item = Path(directory, name)
            relative = item.relative_to(root).as_posix()
            if relative != MARKER:
                items.append((relative.split("/"), relative, item))
    digest = hashlib.sha256()
    for _, relative, item in sorted(items):
        digest.update(relative.encode() + b"\0")
        if item.is_symlink():
            digest.update(hashlib.sha256(b"symlink\0" + os.readlink(item).encode()).digest())
        else:
            digest.update(_file_sha256(item).digest())
    return "sha256:" + digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest


def roots():
    """Cache roots as (path, writable), in search order."""
    own = os.environ.get(PREFIX + "DIR")
    result = [(Path(own).expanduser() if own else home() / "resources", True)]
    system = os.environ.get(PREFIX + "SYSTEM_CACHE") or DEFAULT_SYSTEM_CACHE
    if Path(system).is_dir():
        result.append((Path(system), False))
    return result


def fetch_root():
    """The root new fetches go into: the first writable one."""
    return next(path for path, writable in roots() if writable)


def complete(entry, digest):
    marker = Path(entry) / MARKER
    try:
        return marker.read_text().strip() == digest
    except OSError:
        return False


def lookup(resource):
    """The first complete cached copy of a resource, or None."""
    for root, _ in roots():
        entry = root / resource.entry
        if complete(entry, resource.digest):
            return entry
    return None


@contextlib.contextmanager
def locked(root, entry):
    """Hold an entry's lock in a cache root, creating the root if needed."""
    root = Path(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = open(root / f".{entry}.lock", "w")
    except OSError as error:
        raise ResourceError(
            "cache-not-writable",
            f"Cannot write to the resource cache {root}: {error}",
        ) from None
    with lock:
        # Parallel first uses wait for a single fetch instead of racing.
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield root / entry


def fetch(resource, root):
    """Fetch, verify and publish a resource into a cache root; return the entry."""
    with locked(root, resource.entry) as entry:
        if complete(entry, resource.digest):
            return entry
        with tempfile.TemporaryDirectory(dir=root, prefix=f".{resource.entry}.") as work:
            publish(_verified_tree(resource, Path(work)), entry, resource.digest)
    return entry


def urls(resource):
    """Where a resource's source is fetched from, in order: the declared URL, or
    FINN_RESOURCES_<NAME>_URL in its place, then the declared mirrors."""
    primary = os.environ.get(resource.env + "_URL") or resource.git or resource.url
    return [primary, *resource.mirrors]


def _verified_tree(resource, work):
    """Try each place a resource can come from until one yields its digest."""
    # A copy in another cache (e.g. when filling a directory to carry offline)
    # saves the download; it is verified like a fetched tree.
    copy = lookup(resource)
    failures = []
    offline = os.environ.get(PREFIX + "OFFLINE", "") not in ("", "0")
    if offline and not copy:
        raise ResourceError(
            "resource-offline",
            f"Resource {resource.name} is not cached; offline fetching is disabled",
        )
    for number, where in enumerate(([copy] if copy else []) + ([] if offline else urls(resource))):
        print(f"finn: fetching resource {resource.name} from {where}", file=sys.stderr)
        try:
            if where == copy:
                tree = work / f"copy{number}"
                shutil.copytree(copy, tree, symlinks=True)
            else:
                tree = assemble(resource, work / f"source{number}", where)
        except ResourceError as error:
            failures.append(str(error))
            continue
        actual = tree_digest(tree)
        if actual == resource.digest:
            return tree
        failures.append(f"{where} has digest {actual}")
    message = "\n  ".join(failures)
    raise ResourceError(
        "fetch-failed",
        f"Cannot fetch resource {resource.name} with the digest {resource.origin} declares "
        f"({resource.digest}):\n  {message}\nIf the source changed on purpose, update the "
        "pin with `finn-resources update`.",
    )


def publish(tree, entry, digest):
    """Move a verified tree into place as a complete entry; hold the entry's lock."""
    (tree / MARKER).write_text(digest + "\n")
    if entry.exists():
        # A leftover without a valid marker, e.g. from an interrupted copy.
        shutil.rmtree(entry)
    tree.rename(entry)


def assemble(resource, work, url=None):
    """Fetch a resource's source (by default from its first URL) into work and lay
    out its tree, unverified."""
    url = url or urls(resource)[0]
    if resource.git:
        source = git_source(url, resource.commit, resource.subdir, work / "source")
    else:
        source = archive(url, resource.sha256, work / "source")
    if resource.subdir:
        source = source / resource.subdir
        if not source.is_dir():
            raise ResourceError(
                "subdir-missing",
                f"{resource.subdir} is not a directory in {url}",
            )
    tree = work / "tree"
    shutil.copytree(
        source,
        tree / resource.into if resource.into else tree,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    return tree


def git_source(url, commit, subdir, dest):
    """One commit of a git repository: a sparse checkout, or without git, GitHub's
    archive of the commit."""
    if shutil.which("git"):
        return git_checkout(url, commit, subdir, dest)
    tarball = github_archive(url, commit)
    if tarball is None:
        raise ResourceError("git-unavailable", f"git is required to fetch {url}")
    print(f"finn: git is not installed; using {tarball}", file=sys.stderr)
    # GitHub archives are not byte-stable, so there is no archive checksum; the
    # tree digest verifies the content.
    return archive(tarball, None, dest)


def github_archive(url, commit):
    """The URL of GitHub's tarball of a commit, or None for other hosts."""
    match = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?", url)
    return match and f"https://github.com/{match[1]}/{match[2]}/archive/{commit}.tar.gz"


def git_checkout(url, commit, subdir, dest):
    """Check out one commit, fetching only the blobs under subdir; return the checkout."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(dest), *args], env=env, capture_output=True, text=True
        )
        if result.returncode:
            raise ResourceError(
                "git-failed",
                f"git {args[0]} failed for {url}: {result.stderr.strip()}",
            )

    dest.mkdir(parents=True)
    git("init", "--quiet")
    git("remote", "add", "origin", url)
    if subdir:
        git("sparse-checkout", "set", subdir)
    git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit)
    git("checkout", "--quiet", "FETCH_HEAD")
    return dest


def archive(url, sha256, dest):
    """Download, check (unless sha256 is None) and safely extract an archive;
    return its root directory.

    An archive holding a single top-level directory has that directory as its
    root, as source archives usually do.
    """
    dest.mkdir(parents=True)
    download = dest / "archive"
    try:
        # urllib honours the usual proxy variables.
        with urllib.request.urlopen(url, timeout=60) as response, open(download, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as error:
        raise ResourceError("download-failed", f"Cannot download {url}: {error}") from None
    actual = _file_sha256(download).hexdigest()
    if sha256 is not None and actual != sha256:
        raise ResourceError(
            "archive-sha256-mismatch", f"{url} has sha256 {actual}, expected {sha256}"
        )
    tree = dest / "tree"
    tree.mkdir()
    if tarfile.is_tarfile(download):
        if not hasattr(tarfile, "data_filter"):
            raise ResourceError(
                "archive-extraction-unsupported",
                "Extracting archives safely needs Python 3.11.4 or later",
            )
        with tarfile.open(download) as tar:
            try:
                tar.extractall(tree, filter="data")
            except tarfile.FilterError as error:
                raise ResourceError("archive-unsafe", f"Unsafe archive {url}: {error}") from None
    elif zipfile.is_zipfile(download):
        with zipfile.ZipFile(download) as zip_file:
            for name in zip_file.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ResourceError("archive-unsafe", f"Unsafe archive {url}: member {name}")
            zip_file.extractall(tree)
    else:
        raise ResourceError("archive-format-unknown", f"{url} is not a tar or zip archive")
    download.unlink()
    children = list(tree.iterdir())
    if len(children) == 1 and children[0].is_dir() and not children[0].is_symlink():
        return children[0]
    return tree
