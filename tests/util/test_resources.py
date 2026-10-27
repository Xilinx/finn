"""External resources: declarations, caches, fetchers and overrides.

Sources are local git repositories and archives; no network is used.
"""
import pytest

import io
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from finn import resources
from finn.resources import _declare, _store

pytestmark = pytest.mark.util

SRC = str(Path(__file__).resolve().parents[2] / "src")


@contextmanager
def fault(
    code: str, error: type[resources.ResourceError] = resources.ResourceError, message=None
) -> Iterator[pytest.ExceptionInfo[resources.ResourceError]]:
    """Expect ``error`` carrying ``code`` (and its message matching ``message``, if given):
    a resource fault is asserted by its code, each message's text in one test."""
    with pytest.raises(error, match=message) as caught:
        yield caught
    assert caught.value.code == code, f"raised {caught.value.code!r}: {caught.value}"


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository(path, files):
    """A git repository with one commit of files; returns the commit."""
    path.mkdir()
    git(path, "init", "--quiet")
    # Allow fetching an exact commit, as GitHub does.
    git(path, "config", "uploadpack.allowReachableSHA1InWant", "true")
    return commit(path, files)


def commit(path, files):
    write(path, files)
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "fixture")
    return git(path, "rev-parse", "HEAD")


def write(root, files):
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def digest_of(tmp_path, files):
    """The digest of a tree with exactly these files."""
    root = tmp_path / f"expected-{len(list(tmp_path.iterdir()))}"
    root.mkdir()
    return resources.tree_digest(write(root, files))


def declare(project, text):
    (project / "pyproject.toml").write_text(textwrap.dedent(text))
    resources._cache.clear()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty project as the working directory, with private caches."""
    for variable in list(os.environ):
        if variable.startswith("FINN_RESOURCES_"):
            monkeypatch.delenv(variable)
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "finn-home"))
    monkeypatch.setenv("FINN_RESOURCES_SYSTEM_CACHE", str(tmp_path / "system-cache"))
    (tmp_path / "system-cache").mkdir()
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    resources._cache.clear()
    yield root
    resources._cache.clear()


@pytest.fixture
def boards(tmp_path, project):
    """A board repository declared as a sparse resource placed under 'kv260'."""
    repo = tmp_path / "store"
    revision = repository(
        repo,
        {
            "README.md": "root\n",
            "boards/Xilinx/kv260/board.xml": "k\n",
            "boards/Xilinx/other/board.xml": "o\n",
        },
    )
    digest = digest_of(tmp_path, {"kv260/board.xml": "k\n"})
    declare(
        project,
        f"""
        [tool.finn.resources.kv260-boards]
        kind = ["test-boards"]
        git = "{repo.as_uri()}"
        commit = "{revision}"
        subdir = "boards/Xilinx/kv260"
        into = "kv260"
        digest = "{digest}"
        """,
    )
    return repo


def test_finn_declares_hlslib_and_board_files(project):
    declared = resources.declarations()
    assert list(declared)[0] == "hlslib"
    assert declared["hlslib"].kind == ("hls-include",) and declared["hlslib"].redistributable
    boards = [r for r in declared.values() if "vivado-boards" in r.kind]
    assert len(boards) == 5
    assert not any(r.redistributable for r in boards)
    assert all(r.origin == str(_declare.FINN_FILE) for r in declared.values())


def test_sparse_fetch_places_subdir_into_root_and_caches(boards, tmp_path, monkeypatch):
    path = Path(resources.path("kv260-boards"))
    assert path.parent == tmp_path / "finn-home/resources"
    assert path.name.startswith("kv260-boards-")
    assert (path / "kv260/board.xml").read_text() == "k\n"
    assert sorted(p.name for p in path.iterdir()) == [".finn-resource", "kv260"]
    assert resources.status("kv260-boards").state == "cached"

    # Once fetched, lookups never fetch again.
    def refetch(*args):
        raise AssertionError("fetched again")

    monkeypatch.setattr(_store, "assemble", refetch)
    assert resources.path("kv260-boards") == str(path)
    assert resources.paths("test-boards") == [str(path)]


def test_digest_mismatch_publishes_nothing(boards):
    declared = resources.declarations()["kv260-boards"]
    wrong = _declare.Resource(**{**declared.__dict__, "digest": "sha256:" + "0" * 64})
    root = _store.fetch_root()
    with fault("fetch-failed", message="has digest"):
        _store.fetch(wrong, root)
    assert [p.name for p in root.iterdir()] == [f".{wrong.entry}.lock"]


def test_archives_are_checked_and_extracted_safely(tmp_path, project):
    files = {"acme-1.2/rtl/core.v": "module core;\n", "acme-1.2/doc.txt": "doc\n"}
    content = write(tmp_path / "content", files)
    tar_path = tmp_path / "acme.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(content / "acme-1.2", arcname="acme-1.2")
    zip_path = tmp_path / "acme.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for name, text in files.items():
            zip_file.writestr(name, text)
    digest = digest_of(tmp_path, {"core.v": "module core;\n"})
    declare(
        project,
        f"""
        [tool.finn.resources.acme-tar]
        kind = ["rtl"]
        url = "{tar_path.as_uri()}"
        sha256 = "{sha256(tar_path.read_bytes()).hexdigest()}"
        subdir = "rtl"
        digest = "{digest}"

        [tool.finn.resources.acme-zip]
        kind = ["rtl"]
        url = "{zip_path.as_uri()}"
        sha256 = "{sha256(zip_path.read_bytes()).hexdigest()}"
        subdir = "rtl"
        digest = "{digest}"
        """,
    )
    rtl = resources.paths("rtl")
    assert [(Path(p) / "core.v").read_text() for p in rtl] == ["module core;\n"] * 2

    # A changed archive is refused before extraction.
    zip_path.write_bytes(zip_path.read_bytes() + b"\0")
    with fault("archive-sha256-mismatch", message="sha256"):
        _store.archive(zip_path.as_uri(), "0" * 64, tmp_path / "changed")

    evil_tar = tmp_path / "evil.tar"
    with tarfile.open(evil_tar, "w") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    evil_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zip_file:
        zip_file.writestr("../escape", "x")
    for number, evil in enumerate((evil_tar, evil_zip)):
        with fault("archive-unsafe", message="Unsafe"):
            checksum = sha256(evil.read_bytes()).hexdigest()
            _store.archive(evil.as_uri(), checksum, tmp_path / f"evil{number}")
    assert not (tmp_path / "escape").exists()


def test_cache_order_and_read_only_system_cache(boards, tmp_path, monkeypatch):
    resource = resources.declarations()["kv260-boards"]
    system = tmp_path / "system-cache"
    _store.fetch(resource, system)
    system.chmod(0o555)
    try:
        # The system cache serves the resource; nothing is written elsewhere.
        assert Path(resources.path("kv260-boards")).parent == system
        assert not (tmp_path / "finn-home").exists()

        # FINN_RESOURCES_DIR replaces $FINN_HOME/resources; fetches go there.
        site = tmp_path / "site-cache"
        monkeypatch.setenv("FINN_RESOURCES_DIR", str(site))
        assert Path(resources.path("kv260-boards")).parent == system
        _store.fetch(resource, site)
        assert Path(resources.path("kv260-boards")).parent == site
        assert _store.fetch_root() == site
        monkeypatch.delenv("FINN_RESOURCES_DIR")

        # An incomplete entry (no valid marker) is not a copy.
        (site / resource.entry / ".finn-resource").write_text("sha256:bad\n")
        monkeypatch.setenv("FINN_RESOURCES_DIR", str(site))
        assert Path(resources.path("kv260-boards")).parent == system
    finally:
        system.chmod(0o755)


def test_offline_and_no_fetch_report_the_fetch_command(boards, monkeypatch):
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    with fault("resource-not-cached", message="finn-resources fetch kv260-boards") as error:
        resources.path("kv260-boards")
    # The message names settings that exist.
    assert "set FINN_RESOURCES_DIR=DIR" in str(error.value)
    assert "FINN_RESOURCES_SYSTEM_CACHE=DIR" in str(error.value)
    monkeypatch.delenv("FINN_RESOURCES_OFFLINE")
    with fault("resource-not-cached"):
        resources.paths("test-boards", fetch=False)
    assert resources.status("kv260-boards").state == "missing"


def test_overrides(boards, tmp_path, monkeypatch):
    local = tmp_path / "local $ checkout"
    local.mkdir()
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS", str(local))
    assert resources.path("kv260-boards") == str(local)
    assert resources.status("kv260-boards") == resources.Status(
        "kv260-boards", "override", str(local), "FINN_RESOURCES_KV260_BOARDS"
    )
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS", str(tmp_path / "missing"))
    with fault("override-not-directory", message="FINN_RESOURCES_KV260_BOARDS"):
        resources.path("kv260-boards")

    # The retired names select nothing: only the resource's own variable is read.
    monkeypatch.setenv("FINN_HLSLIB_PATH", str(local))
    monkeypatch.setenv("FINN_BOARD_FILES_PATH", str(local))
    resources._cache.clear()
    assert resources.status("hlslib").state != "override"


def _fetch_in_child(queue):
    queue.put(resources.path("kv260-boards"))


def test_parallel_first_use_fetches_once(boards, tmp_path, monkeypatch):
    log = tmp_path / "fetches"
    assemble = _store.assemble

    def counted(resource, *args):
        with open(log, "a") as file:
            file.write(resource.name + "\n")
        return assemble(resource, *args)

    monkeypatch.setattr(_store, "assemble", counted)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [context.Process(target=_fetch_in_child, args=(queue,)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert len({queue.get() for _ in workers}) == 1
    assert log.read_text() == "kv260-boards\n"


def test_project_redefinition_is_logged_and_packages_cannot_redefine(
    boards, tmp_path, caplog, monkeypatch
):
    local = write(tmp_path / "newer-hlslib", {"bnn-library.h": "// newer\n"})
    extra = tmp_path / "extra.toml"
    extra.write_text(
        textwrap.dedent(
            f"""
            [resources.hlslib]
            kind = ["hls-include"]
            url = "{local.as_uri()}"
            sha256 = "{"0" * 64}"
            digest = "{resources.tree_digest(local)}"
            """
        )
    )
    monkeypatch.setenv("FINN_RESOURCES_FILES", str(extra))
    with caplog.at_level(logging.WARNING, logger="finn.resources"):
        declared = resources.declarations()
    assert declared["hlslib"].origin == str(extra)
    # A redefinition keeps FINN's position.
    assert list(declared)[0] == "hlslib"
    assert f"{extra} redefines resource 'hlslib'" in caplog.text

    finn = _declare.load(_declare.FINN_FILE, ("resources",))
    package = _declare.parse({"hlslib": {"package": "acme", "subdir": "hls"}}, "package acme")
    with fault(
        "resource-redeclared", resources.DeclarationError, "already declared by .*resources.toml"
    ):
        _declare.merge(finn, package, [])
    first = _declare.parse({"acme": {"package": "acme", "subdir": "a"}}, "package acme")
    second = _declare.parse({"acme": {"package": "other", "subdir": "a"}}, "package other")
    with fault("resource-redeclared", resources.DeclarationError, "package other.*package acme"):
        _declare.merge([], first + second, [])


@pytest.mark.parametrize(
    "fields, code, message",
    [
        ({"git": "u", "commit": "c" * 40}, "digest-invalid", "digest must be"),
        ({"git": "u"}, "field-required", "commit is required"),
        ({"git": "u", "url": "u"}, "source-not-one", "exactly one source"),
        (
            {"git": "u", "commit": "main", "digest": "sha256:" + "0" * 64},
            "commit-invalid",
            "full 40-character",
        ),
        (
            {"url": "u", "sha256": "0" * 64, "digest": "sha256:" + "0" * 64, "subdir": "../x"},
            "path-not-relative",
            "relative",
        ),
        ({"package": "m", "into": "x"}, "package-fields-invalid", "package source"),
        ({"package": "m", "kind": "rtl"}, "field-type-invalid", "kind must be a list"),
        ({"package": "m", "extra": 1}, "field-unknown", "unknown field"),
        ({"package": "m", "mirrors": ["u"]}, "package-fields-invalid", None),
        (
            {"url": "u", "sha256": "0" * 64, "digest": "sha256:" + "0" * 64, "mirrors": "u"},
            "field-type-invalid",
            "mirrors",
        ),
    ],
)
def test_invalid_declarations_are_rejected(fields, code, message):
    with fault(code, resources.DeclarationError, message):
        _declare.parse({"acme": fields}, "test")


@pytest.mark.parametrize(
    "name, code",
    [
        ("Acme", "name-invalid"),
        ("dir", "name-reserved"),
        ("offline", "name-reserved"),
        ("acme-url", "name-reserved"),
        ("acme_rtl", "name-invalid"),
    ],
)
def test_names_are_restricted(name, code):
    with fault(code, resources.DeclarationError):
        _declare.parse({name: {"package": "m"}}, "test")


def test_package_source_resolves_installed_data(project, tmp_path, monkeypatch):
    write(tmp_path / "site", {"acme_data/__init__.py": "", "acme_data/rtl/core.v": "m\n"})
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    declare(
        project,
        '[tool.finn.resources.acme]\nkind = ["rtl"]\npackage = "acme_data"\nsubdir = "rtl"\n',
    )
    assert resources.path("acme") == str(tmp_path / "site/acme_data/rtl")
    assert resources.status("acme").state == "package"


def test_package_source_names_a_module_not_a_path(tmp_path):
    with fault("package-invalid", resources.DeclarationError):
        _declare.parse({"acme": {"package": "acme_data:rtl"}}, "test")
    with fault("package-fields-invalid", resources.DeclarationError):
        _declare.parse({"acme": {"package": "acme_data", "into": "x"}}, "test")


def test_path_source_is_used_in_place(project, tmp_path):
    write(tmp_path / "libs", {"acme-rtl/core.v": "m\n"})
    declare(project, '[tool.finn.resources.acme]\nkind = ["rtl"]\npath = "../libs/acme-rtl"\n')
    assert resources.path("acme") == str(tmp_path / "libs/acme-rtl")
    assert resources.status("acme").state == "path"
    assert resources.fetch(["acme"]) == []
    # An override still wins, and a missing directory is an error, not a fetch.
    declare(project, '[tool.finn.resources.acme]\npath = "/nonexistent/acme"\n')
    resources._cache.clear()
    with fault("path-not-directory"):
        resources.path("acme")
    with fault("path-fields-invalid", resources.DeclarationError, "a path source is a directory"):
        _declare.parse({"acme": {"path": "x", "digest": "sha256:" + "0" * 64}}, "test")


def test_project_is_the_nearest_pyproject(project, tmp_path):
    declare(project, '[tool.finn.resources.acme]\npackage = "acme"\nsubdir = "rtl"\n')
    nested = project / "a/b"
    nested.mkdir(parents=True)
    assert [f for f, _ in _declare.project_files(nested)] == [project / "pyproject.toml"]
    (project / "a/pyproject.toml").write_text("[project]\nname = 'inner'\n")
    assert [f for f, _ in _declare.project_files(nested)] == [project / "a/pyproject.toml"]


def test_digest_matches_the_previous_board_file_algorithm(tmp_path):
    # Sorted by path components, not strings: "a/x" before "a-b/x".
    tree = write(tmp_path / "t", {"a-b/x": "1", "a/x": "2", ".finn-resource": "ignored"})
    old = __import__("hashlib").sha256()
    for name in ("a/x", "a-b/x"):
        old.update(name.encode() + b"\0" + sha256((tree / name).read_bytes()).digest())
    assert resources.tree_digest(tree) == "sha256:" + old.hexdigest()


def test_resources_import_only_the_standard_library(tmp_path):
    check = textwrap.dedent(
        """
        import sys
        before = set(sys.modules)
        from finn import resources
        resources.declarations()
        resources.status("hlslib")
        loaded = {name.partition(".")[0] for name in set(sys.modules) - before}
        foreign = sorted(loaded - set(sys.stdlib_module_names) - {"finn"})
        assert not foreign, foreign
        assert not any(n.startswith("finn.") and not n.startswith("finn.resources")
                       for n in sys.modules), sorted(sys.modules)
        """
    )
    # -S: no site-packages at all, so any non-stdlib import fails outright.
    subprocess.run(
        [sys.executable, "-S", "-c", check],
        env={"PYTHONPATH": SRC, "HOME": str(tmp_path), "PATH": os.defpath},
        cwd=tmp_path,
        check=True,
    )


def cli(capsys, *args):
    from finn.resources import _cli  # noqa: PLC0415

    code = _cli.main(list(args))
    out, err = capsys.readouterr()
    return code, out, err


def test_cli_list_fetch_verify_and_digest(boards, tmp_path, capsys):
    board = boards / "boards/Xilinx/kv260/board.xml"
    assert cli(capsys, "list", "--kind", "test-boards")[1].splitlines()[1].split()[:3] == [
        "kv260-boards",
        "missing",
        "test-boards",
    ]
    code, out, err = cli(capsys, "fetch", "--kind", "test-boards", "--redistributable-only")
    assert (code, out, err) == (0, "", "")
    dest = tmp_path / "carry"
    code, out, _ = cli(capsys, "fetch", "kv260-boards", "--dest", str(dest))
    assert code == 0 and out.strip() == str(dest / resources.declarations()["kv260-boards"].entry)
    assert cli(capsys, "fetch")[0] == 1
    assert "no resource named 'nope'" in cli(capsys, "fetch", "nope")[2].lower()

    # A board.xml in the fetched tree is listed with --boards.
    entry = Path(out.strip())
    (entry / "kv260/board.xml").write_text(
        '<board vendor="xilinx.com" name="kv260_som" display_name="KV260">'
        "<file_version>1.3</file_version><components>"
        '<component name="som" type="fpga"/></components></board>'
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FINN_RESOURCES_DIR", str(dest))
        out = cli(capsys, "list", "--boards", "--kind", "test-boards")[1]
        assert "xilinx.com:kv260_som:som:1.3  KV260" in out
        # ...which also changed the content: verify notices.
        code, out, _ = cli(capsys, "verify", "kv260-boards")
        assert code == 1 and out.startswith("MODIFIED  kv260-boards")
    assert cli(capsys, "digest", str(entry))[1].strip() == resources.tree_digest(entry)
    assert board.exists()


def test_cli_update_rewrites_only_the_pin(boards, project, capsys, monkeypatch):
    before = (project / "pyproject.toml").read_text()
    (project / "pyproject.toml").write_text(
        "# A project\n[project]\nname = 'acme'\n" + before + "\n[tool.other]\ncommit = 'keep'\n"
    )
    old = resources.declarations()["kv260-boards"]
    revision = commit(boards, {"boards/Xilinx/kv260/board.xml": "k2\n"})
    git(boards, "branch", "release")

    code, out, _ = cli(capsys, "update", "kv260-boards", "--ref", "release")
    assert code == 0 and revision[:12] in out
    text = (project / "pyproject.toml").read_text()
    resources._cache.clear()
    new = resources.declarations()["kv260-boards"]
    assert new.commit == revision and new.digest != old.digest
    assert text.replace(revision, old.commit).replace(new.digest, old.digest) == (
        "# A project\n[project]\nname = 'acme'\n" + before + "\n[tool.other]\ncommit = 'keep'\n"
    )
    # update published the new tree, so no fetch is needed.
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    assert (Path(resources.path("kv260-boards")) / "kv260/board.xml").read_text() == "k2\n"
    assert "already at" in cli(capsys, "update", "kv260-boards", "--ref", revision)[1]

    # Declarations inside an installed package are printed, never edited.
    monkeypatch.setattr("finn.resources._cli._packaged", lambda file: True)
    commit(boards, {"boards/Xilinx/kv260/board.xml": "k3\n"})
    code, out, _ = cli(capsys, "update", "kv260-boards")
    assert code == 0 and "[tool.finn.resources.kv260-boards]" in out
    assert (project / "pyproject.toml").read_text() == text


def test_update_refuses_edits_it_cannot_confine(tmp_path):
    file = tmp_path / "resources.toml"
    inline = '[resources]\nacme = { git = "u", commit = "%s", digest = "d" }\n' % ("a" * 40)
    file.write_text(inline)
    with fault("rewrite-value-missing", resources.DeclarationError, "edit it by hand"):
        _declare.rewrite(file, ("resources",), "acme", {"commit": "b" * 40})
    # A multi-line string that looks like a field would change another value.
    tricky = '[resources.acme]\nnote = """\ncommit = "x"\n"""\ncommit = "%s"\n' % ("a" * 40)
    file.write_text(tricky)
    with fault("rewrite-changes-more", resources.DeclarationError, "change more than"):
        _declare.rewrite(file, ("resources",), "acme", {"commit": "b" * 40})
    assert file.read_text() == tricky


def test_cli_clean_and_check(boards, tmp_path, capsys, monkeypatch):
    kept = Path(resources.path("kv260-boards"))
    stale = kept.parent / "kv260-boards-0123456789abcdef"
    stale.mkdir()
    (kept.parent / "unrelated").mkdir()
    code, out, _ = cli(capsys, "clean", "--unused")
    assert code == 0 and out == f"removed {stale}\n"
    assert kept.is_dir() and (kept.parent / "unrelated").is_dir()
    assert cli(capsys, "clean")[1] == f"removed {kept}\n"

    assert cli(capsys, "check")[0] == 0
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS_URL", "https://mirror.example/kv260.git")
    assert cli(capsys, "check")[0] == 0
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS", str(tmp_path / "missing"))
    monkeypatch.setenv("FINN_RESOURCES_KV206_BOARDS", str(tmp_path))
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    resources._cache.clear()
    code, out, _ = cli(capsys, "check")
    assert code == 1
    assert "FINN_RESOURCES_KV260_BOARDS=" in out and "is not a directory" in out
    assert "FINN_RESOURCES_KV206_BOARDS matches no declared resource" in out
    assert "offline, and not in any cache: hlslib" in out


def test_cli_runs_as_a_module(project):
    result = subprocess.run(
        [sys.executable, "-m", "finn.resources", "list", "--kind", "hls-include"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[1].startswith("hlslib ")


def installed(site, dist, module, declarations):
    """An installed distribution declaring resources through the finn.resources group."""
    info = f"{dist.replace('-', '_')}-1.0.dist-info"
    return write(
        site,
        {
            f"{module}/__init__.py": "",
            f"{module}/resources.toml": textwrap.dedent(declarations),
            f"{module}/rtl/core.v": "module core;\n",
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0\n",
            f"{info}/entry_points.txt": f"[finn.resources]\n{module} = {module}\n",
        },
    )


def test_packages_add_resources_but_only_the_project_redefines(
    project, tmp_path, monkeypatch, caplog
):
    site = installed(
        tmp_path / "site",
        "acme-finn",
        "acme_finn",
        """
        [resources.acme-rtl]
        kind = ["rtl"]
        package = "acme_finn"
        subdir = "rtl"
        """,
    )
    monkeypatch.syspath_prepend(str(site))
    declared = resources.declarations()
    assert list(declared)[-1] == "acme-rtl"
    assert declared["acme-rtl"].origin == str(site / "acme_finn/resources.toml")
    assert resources.paths("rtl") == [str(site / "acme_finn/rtl")]

    # The project may redefine a package's resource.
    declare(project, '[tool.finn.resources.acme-rtl]\nkind = ["rtl"]\npackage = "acme_finn"\n')
    with caplog.at_level(logging.WARNING, logger="finn.resources"):
        assert resources.paths("rtl") == [str(site / "acme_finn")]
    assert "redefines resource 'acme-rtl'" in caplog.text

    # A package may not redefine FINN's resources or another package's.
    installed(site, "zeta", "zeta_finn", '[resources.hlslib]\npackage = "zeta_finn"\n')
    resources._cache.clear()
    with fault("resource-redeclared", resources.DeclarationError, "zeta_finn"):
        resources.declarations()
    installed(site, "zeta", "zeta_finn", '[resources.acme-rtl]\npackage = "zeta_finn"\n')
    resources._cache.clear()
    with fault("resource-redeclared", resources.DeclarationError, "zeta_finn.*acme_finn"):
        resources.declarations()

    # A broken entry point is an error, not silently skipped.
    (site / "zeta_finn/resources.toml").unlink()
    resources._cache.clear()
    with fault(
        "entry-point-no-declarations",
        resources.DeclarationError,
        "zeta_finn contains no resources.toml",
    ):
        resources.declarations()


def test_package_declarations_in_a_virtual_environment(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
    site = next(venv.glob("lib/python*/site-packages"))
    # FINN's resources package only; it needs nothing else.
    (site / "finn-src.pth").write_text(SRC + "\n")
    installed(
        site,
        "acme-finn",
        "acme_finn",
        '[resources.acme-rtl]\nkind = ["rtl"]\npackage = "acme_finn"\nsubdir = "rtl"\n',
    )

    def run(*args):
        return subprocess.run(
            [venv / "bin/python", "-m", "finn.resources", *args],
            cwd=tmp_path,
            env={"PATH": os.defpath, "HOME": str(tmp_path)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert run("list", "--kind", "rtl").splitlines()[1].split()[:3] == [
        "acme-rtl",
        "package",
        "rtl",
    ]
    assert run("path", "acme-rtl").strip() == str(site / "acme_finn/rtl")


def test_fetching_into_another_cache_copies_a_verified_copy(boards, tmp_path):
    fetched = Path(resources.path("kv260-boards"))

    def no_network(*args):
        raise AssertionError("fetched from the source")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_store, "assemble", no_network)
        (carried,) = resources.fetch(["kv260-boards"], tmp_path / "carry")
    assert Path(carried).parent == tmp_path / "carry"
    assert resources.tree_digest(carried) == resources.tree_digest(fetched)

    # A copy that no longer matches its digest is not propagated: the source is used.
    (fetched / "kv260/board.xml").write_text("tampered\n")
    (again,) = resources.fetch(["kv260-boards"], tmp_path / "carry-again")
    assert (Path(again) / "kv260/board.xml").read_text() == "k\n"


def test_mirrors_and_url_override(boards, project, tmp_path, monkeypatch):
    text = (project / "pyproject.toml").read_text()
    mirror = tmp_path / "mirror"
    git(tmp_path, "clone", "--quiet", "--bare", str(boards), str(mirror))
    git(mirror, "config", "uploadpack.allowReachableSHA1InWant", "true")
    missing = (tmp_path / "gone").as_uri()
    declare(
        project,
        text.replace(boards.as_uri(), missing)
        + f'mirrors = ["{(tmp_path / "gone-too").as_uri()}", "{mirror.as_uri()}"]\n',
    )
    path = Path(resources.path("kv260-boards"))
    assert (path / "kv260/board.xml").read_text() == "k\n"

    # Every source failing names each failure.
    for directory in _store.roots():
        shutil.rmtree(directory[0], ignore_errors=True)
    shutil.rmtree(mirror)
    with pytest.raises(resources.ResourceError) as error:
        resources.path("kv260-boards")
    assert str(error.value).count("git fetch failed") == 3

    # FINN_RESOURCES_<NAME>_URL replaces the declared URL, e.g. with a site mirror.
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS_URL", boards.as_uri())
    assert Path(resources.path("kv260-boards")) == path


def test_without_git_github_sources_use_the_commit_archive(boards, project, tmp_path, monkeypatch):
    resource = resources.declarations()["kv260-boards"]
    tarball = tmp_path / "commit.tar.gz"
    git(boards, "archive", f"--prefix=store-{resource.commit}/", "-o", str(tarball), "HEAD")
    requested = []

    def github_archive(url, commit):
        requested.append((url, commit))
        return tarball.as_uri()

    monkeypatch.setattr(_store, "github_archive", github_archive)
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    path = Path(resources.path("kv260-boards"))
    assert (path / "kv260/board.xml").read_text() == "k\n"
    assert requested == [(boards.as_uri(), resource.commit)]


def test_github_archive_urls():
    commit = "8d979e2bdced486dd25d26607d1ff5ae327ed6a8"
    for url in (
        "https://github.com/Xilinx/finn-hlslib.git",
        "https://github.com/Xilinx/finn-hlslib",
        "https://github.com/Xilinx/finn-hlslib/",
    ):
        assert _store.github_archive(url, commit) == (
            f"https://github.com/Xilinx/finn-hlslib/archive/{commit}.tar.gz"
        )
    assert _store.github_archive("https://gitlab.com/a/b.git", commit) is None
    assert _store.github_archive("git@github.com:a/b.git", commit) is None


def test_the_build_directory_is_a_machine_setting_beside_home(tmp_path, monkeypatch):
    """FINN_BUILD_DIR, absolute, else $FINN_HOME/build; named, never created."""
    environ = {"FINN_HOME": str(tmp_path / "home")}
    assert resources.scratch(environ) == tmp_path / "home" / "build"
    environ["FINN_BUILD_DIR"] = str(tmp_path / "scratch")
    assert resources.scratch(environ) == tmp_path / "scratch"
    monkeypatch.chdir(tmp_path)
    assert resources.scratch({"FINN_BUILD_DIR": "relative"}) == tmp_path / "relative"
    monkeypatch.setenv("FINN_BUILD_DIR", str(tmp_path / "from-environment"))
    assert resources.scratch() == tmp_path / "from-environment"
    names = {"build", "scratch", "relative", "from-environment"}
    assert not any(path.name in names for path in tmp_path.rglob("*"))


def test_a_resource_fault_names_its_reason_by_a_code_and_keeps_its_message():
    with pytest.raises(resources.ResourceError, match="No resource named") as error:
        resources.path("no-such-resource")
    assert error.value.code == "resource-undeclared"
    assert str(error.value) == "No resource named 'no-such-resource' is declared"
    declaration = resources.DeclarationError("name-invalid", "bad name")
    assert isinstance(declaration, resources.ResourceError)
    assert (declaration.code, str(declaration)) == ("name-invalid", "bad name")
