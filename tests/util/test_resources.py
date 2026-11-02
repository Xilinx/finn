"""External resources: declarations, caches, fetchers and overrides.

Sources are local git repositories; no network is used.
"""
import pytest

import multiprocessing
import os
import shutil
import subprocess
import sys
import textwrap
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


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Private caches, and a declare(text) that replaces FINN's declarations."""
    for variable in list(os.environ):
        if variable.startswith(("FINN_RESOURCES_", "FINN_BOARD_FILES_PATH")):
            monkeypatch.delenv(variable)
    monkeypatch.setenv("FINN_HOME", str(tmp_path / "finn-home"))
    monkeypatch.setenv("FINN_RESOURCES_SYSTEM_CACHE", str(tmp_path / "system-cache"))
    (tmp_path / "system-cache").mkdir()
    monkeypatch.chdir(tmp_path)
    file = tmp_path / "resources.toml"

    def declare(text):
        file.write_text(textwrap.dedent(text))
        monkeypatch.setattr(_declare, "FINN_FILE", file)
        resources._cache.clear()
        return file

    resources._cache.clear()
    yield declare
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
    project(
        f"""
        # Test resources.
        [resources.kv260-boards]
        kind = ["test-boards"]
        git = "{repo.as_uri()}"
        commit = "{revision}"
        subdir = "boards/Xilinx/kv260"
        into = "kv260"
        digest = "{digest}"
        """
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

        # An incomplete entry (no valid marker) is not a copy.
        (site / resource.entry / ".finn-resource").write_text("sha256:bad\n")
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


def test_retired_names_select_nothing(project, tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_HLSLIB_PATH", str(tmp_path))
    assert resources.status("hlslib").state != "override"


def test_url_override_fetches_the_same_content_elsewhere(boards, tmp_path, monkeypatch):
    """FINN_RESOURCES_<NAME>_URL names a site mirror; the digest still decides."""
    mirror = tmp_path / "mirror"
    git(tmp_path, "clone", "--quiet", "--bare", str(boards), str(mirror))
    git(mirror, "config", "uploadpack.allowReachableSHA1InWant", "true")
    shutil.rmtree(boards)
    with fault("fetch-failed", message="git fetch failed"):
        resources.path("kv260-boards")
    monkeypatch.setenv("FINN_RESOURCES_KV260_BOARDS_URL", mirror.as_uri())
    assert (Path(resources.path("kv260-boards")) / "kv260/board.xml").read_text() == "k\n"


def test_git_is_required_to_fetch(boards, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    with fault("fetch-failed", message="git is required"):
        resources.path("kv260-boards")


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


VALID = {"git": "u", "commit": "c" * 40, "digest": "sha256:" + "0" * 64}


@pytest.mark.parametrize(
    "fields, code, message",
    [
        ({"git": "u", "commit": "c" * 40}, "field-required", "digest required"),
        ({"git": "u"}, "field-required", "commit, digest required"),
        ({**VALID, "digest": "sha256:0"}, "digest-invalid", "digest must be"),
        ({**VALID, "commit": "main"}, "commit-invalid", "full 40-character"),
        ({**VALID, "subdir": "../x"}, "path-not-relative", "relative"),
        ({**VALID, "kind": "rtl"}, "field-type-invalid", "kind must be a list"),
        ({**VALID, "url": "u"}, "field-unknown", "unknown field"),
        ({**VALID, "mirrors": ["u"]}, "field-unknown", "unknown field"),
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
        _declare.parse({name: VALID}, "test")


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
    # A named resource must be present; unfetched ones are fine otherwise.
    assert cli(capsys, "verify", "kv260-boards")[:2] == (1, "missing  kv260-boards\n")
    assert cli(capsys, "verify")[0] == 0
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


def test_cli_update_rewrites_only_the_pin(boards, capsys, monkeypatch):
    file = _declare.FINN_FILE
    before = file.read_text()
    file.write_text(before + '\n[other]\ncommit = "keep"\n')
    old = resources.declarations()["kv260-boards"]
    revision = commit(boards, {"boards/Xilinx/kv260/board.xml": "k2\n"})
    git(boards, "branch", "release")

    code, out, _ = cli(capsys, "update", "kv260-boards", "--ref", "release")
    assert code == 0 and revision[:12] in out
    text = file.read_text()
    resources._cache.clear()
    new = resources.declarations()["kv260-boards"]
    assert new.commit == revision and new.digest != old.digest
    assert text.replace(revision, old.commit).replace(new.digest, old.digest) == (
        before + '\n[other]\ncommit = "keep"\n'
    )
    # update published the new tree, so no fetch is needed.
    monkeypatch.setenv("FINN_RESOURCES_OFFLINE", "1")
    assert (Path(resources.path("kv260-boards")) / "kv260/board.xml").read_text() == "k2\n"
    assert "already at" in cli(capsys, "update", "kv260-boards", "--ref", revision)[1]


def test_update_refuses_edits_it_cannot_confine(tmp_path):
    file = tmp_path / "resources.toml"
    inline = '[resources]\nacme = { git = "u", commit = "%s", digest = "d" }\n' % ("a" * 40)
    file.write_text(inline)
    with fault("rewrite-value-missing", resources.DeclarationError, "edit it by hand"):
        _declare.rewrite(file, "acme", {"commit": "b" * 40})
    # A multi-line string that looks like a field would change another value.
    tricky = '[resources.acme]\nnote = """\ncommit = "x"\n"""\ncommit = "%s"\n' % ("a" * 40)
    file.write_text(tricky)
    with fault("rewrite-changes-more", resources.DeclarationError, "change more than"):
        _declare.rewrite(file, "acme", {"commit": "b" * 40})
    assert file.read_text() == tricky


def test_cli_runs_as_a_module(project):
    result = subprocess.run(
        [sys.executable, "-m", "finn.resources", "list", "--kind", "hls-include"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[1].startswith("hlslib ")


def test_a_resource_fault_names_its_reason_by_a_code_and_keeps_its_message():
    with pytest.raises(resources.ResourceError, match="No resource named") as error:
        resources.path("no-such-resource")
    assert error.value.code == "resource-undeclared"
    assert str(error.value) == "No resource named 'no-such-resource' is declared"
    declaration = resources.DeclarationError("name-invalid", "bad name")
    assert isinstance(declaration, resources.ResourceError)
    assert (declaration.code, str(declaration)) == ("name-invalid", "bad name")
