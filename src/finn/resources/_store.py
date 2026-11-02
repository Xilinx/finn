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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ._declare import PREFIX, ResourceError

MARKER = ".finn-resource"
DEFAULT_SYSTEM_CACHE = "/opt/finn/resources"


def home(environ=None):
    """FINN's persistent per-user directory: FINN_HOME, by default ``~/.finn``.

    It holds content-keyed state that outlives builds: fetched resources
    (``resources/``, or FINN_RESOURCES_DIR), finn_xsi builds (``xsi/``, or
    FINN_XSI_PATH), golden references (``golden/``, or FINN_GOLDEN_DIR) and
    container identity files (``identity/``). Deleting it only costs rebuilds.
    """
    env = os.environ if environ is None else environ
    return Path(env.get("FINN_HOME") or Path.home() / ".finn").expanduser()


def scratch(environ=None):
    """FINN's build directory, where builds and tests allocate their working
    directories: FINN_BUILD_DIR, by default ``/tmp/finn_build_<uid>``. Nothing
    there is reused, so it may be wiped at any time. Absolute, and not created
    here."""
    env = os.environ if environ is None else environ
    value = env.get("FINN_BUILD_DIR")
    return Path(value).expanduser().resolve() if value else Path(f"/tmp/finn_build_{os.getuid()}")


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


def url(resource):
    """Where a resource's source is fetched from: FINN_RESOURCES_<NAME>_URL (e.g. a
    site mirror), else the declared repository."""
    return os.environ.get(resource.env + "_URL") or resource.git


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
    for number, where in enumerate(([copy] if copy else []) + ([] if offline else [url(resource)])):
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


def assemble(resource, work, source_url=None):
    """Check out a resource's source (by default from url()) into work and lay out
    its tree, unverified."""
    source_url = source_url or url(resource)
    source = git_checkout(source_url, resource.commit, resource.subdir, work / "source")
    if resource.subdir:
        source = source / resource.subdir
        if not source.is_dir():
            raise ResourceError(
                "subdir-missing",
                f"{resource.subdir} is not a directory in {source_url}",
            )
    tree = work / "tree"
    shutil.copytree(
        source,
        tree / resource.into if resource.into else tree,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git"),
    )
    return tree


def git_checkout(url, commit, subdir, dest):
    """Check out one commit, fetching only the blobs under subdir; return the checkout."""
    if not shutil.which("git"):
        raise ResourceError("git-unavailable", f"git is required to fetch {url}")
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
