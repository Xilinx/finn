# FINN environments

FINN uses uv 0.10.0, Python 3.12 and one committed `uv.lock`. Native development,
Docker, documentation and CI use the dependency groups in `pyproject.toml`.
The initial development image includes the full upstream notebook, export,
reporting and benchmark environment. CUDA PyTorch 2.8/cu126 is retained.

Docker builds require Docker Buildx with Bake (`docker buildx bake --help`).
Install `docker-buildx-plugin` alongside Docker Engine if it is missing.
A published-image launch with an explicit verified image needs no Bake.

## Start an environment

```bash
./docker/build                         # prepare the development environment
./docker/run                           # software shell
./docker/run --fpga -- python -m pytest -m util
./docker/run --fpga --runtime xrt -- build_dataflow /path/in/the/checkout
./docker/run --notebook                 # Jupyter token URL, ports 8888 and 8081
```

`docker/run` prepares a missing image; `--no-build` requires an existing one.
Use `--rebuild` to refresh all build layers. `FINN_UV_NO_CACHE=1` skips the
intermediate uv build cache on storage-constrained builders; the final locked
environment and normal Docker layer cache are preserved. Commands and mounts preserve argv,
including spaces. The caller's UID/GID owns outputs. Private NSS files provide its passwd/group identity through `libnss_wrapper`, including for OpenSSH and direct `docker exec`; system account files are unchanged. Site overrides of `LD_PRELOAD` must retain the image's `libnss_wrapper` and licence workaround libraries. Site Docker flags in
`FINN_DOCKER_EXTRA` are shell-tokenized without executing substitutions.
`--env NAME[=VALUE]`, `--volume SOURCE:TARGET[:MODE]`, `--workdir DIR` and
`--name NAME` are repeatable or explicit command options. An external model
folder needs an explicit mount:

```bash
model_dir="/absolute/path/model with spaces"
./docker/run --fpga --volume "$model_dir:$model_dir" --workdir "$model_dir" -- python build.py
```

The checkout and build directory are mirrored at their host paths. The active
`/opt/venv` also applies to `docker exec`. Startup only installs FINN editable,
without dependencies, and checks that its lock matches the image. Setup failure
stops execution. Rebuild after changing dependencies; source edits reuse the
same development environment. Identity uses canonical Git executable bits so differing Jenkins umasks do not change it. The image contains `tini` for child reaping.
Only archives listed by selected `SOURCE=supply` runtime manifests contribute to
identity. A leftover `docker/packages/slashkit.deb` does not change an XRT-only tag.
Attached terminals work for explicit commands such as `docker/run -- ipython`
as well as the default shell; CI invocations remain noninteractive.

For native use, install Ubuntu system
prerequisites (`sudo ./scripts/install-system-deps.sh`), then run:

```bash
./setup-local.sh --ci --skip-xsi
source scripts/activate.sh
./scripts/quicktest-local.sh
```

Setup uses the uv version in `.uv-version`, bootstrapping its standalone binary into `.cache/uv-tool/` if necessary; it does not install compiler packages with pip. Omit `--skip-xsi` to prepare XSI when Vivado is selected. `FINN_VENV` and
`FINN_PYTHON` select the environment and Python 3.12 interpreter. Native uv caches
and default build outputs live under `.cache/`; set `UV_CACHE_DIR` and
`FINN_HOST_BUILD_DIR` for site storage. Activation performs no installation.

## Host tools and licences

`docker/config.py` is a standard-library host resolver; it works before FINN is
installed. It reads `~/.config/finn/xilinx.env`, with environment variables taking
precedence. `FINN_XILINX_ENV` selects another file; empty disables it. Use plain
`NAME=value` lines, without expansion or quotes:

```text
FINN_XILINX_PATH=/opt/Xilinx
FINN_XILINX_VERSION=2024.2
XILINXD_LICENSE_FILE=2100@licence-server
```

Both `$ROOT/Vivado/$VERSION` and `$ROOT/$VERSION/Vivado` layouts are detected.
`--fpga` requests read-only tool and licence-file mounts. FLEXlm file lists and
servers (`PORT@HOST`) are supported; `LM_LICENSE_FILE` remains available.
Alternatively, set `FINN_LICENSE_HOST` and `FINN_LICENSE_PORT` in the machine
file. `FINN_LICENSE_VENDOR_PORT` describes the vendor-daemon port where known;
network requirements reported by inspection are declarations, not firewall rules.

`PLATFORM_REPO_PATHS` accepts colon-separated absolute directories. Explicit
`HLS_PATH`, `VITIS_PATH` and `VIVADO_PATH` overrides are retained.
`FINN_TOOL_DIR_OVERRIDE` retains site interception, including LSF routing. Supply
site-specific mounts/env/hostname through `FINN_DOCKER_EXTRA`. Jenkins keeps its
existing per-agent hostname and farm cleanup. `.Xilinx/` Tcl files in the checkout
are copied into the container's writable home.
Farm interception applies through `resolve_xilinx_tool()`; wrapper directories
are not prepended to PATH for nested/bare commands. With interception enabled,
startup copies hlslib from the image into the writable resource cache without
network access. Keep `FINN_HOST_BUILD_DIR` (and any resource override) on storage
visible at the same absolute path to the farm, since generated Tcl embeds it.
Missing optional platform repositories or tool installations warn and are not
mounted. Native activation prepares board files only when Vivado is selected.

Inspect without running Docker, or render a Compose overlay:

```bash
./docker/config.py inspect --tier build --workspace-policy mirror
FINN_IMAGE=$(./docker/build --print-tag) ./docker/config.py compose --tier build --service build
```

Compose services are optional consumers of the same resolver. `docker/run` uses
Docker directly so site flags such as `--hostname` retain their Docker semantics.

## Dependencies and external resources

Use `uv add`, `uv lock --upgrade-package NAME` and `uv sync --locked` for Python
updates. Git revisions for QONNX and Brevitas remain upstream's selections.
All groups resolve together: `test`, `ci`, `notebooks`, `docs`, and the complete
`dev` group. For native co-development, use uv's local source support
(`uv add --editable /absolute/checkout`), review both manifest and lock changes,
and restore the committed selection before submitting unrelated work. For
containers, mount the local checkout and use a per-command uv overlay instead
of placing a host-only dependency path in an image build's lock:

```bash
./docker/run --volume /absolute/qonnx:/absolute/qonnx -- uv run --locked --with-editable /absolute/qonnx python your_test.py
```

Like the Brevitas candidate overlay, additional dependencies resolve separately
unless constrained with a generated uv export. No dependency fork is selected
by default.

External trees are declared in `src/finn/resources.toml`. `finn-resources` lists,
fetches, verifies and updates them:

```bash
finn-resources list
finn-resources fetch --kind hls-include --kind vivado-boards
finn-resources verify
finn-resources update hlslib --ref FULL_COMMIT_OR_REF
```

Caches are content-addressed and published atomically under file locks. Local
`FINN_RESOURCES_<NAME>` directory overrides are unverified; names use uppercase
and underscores (`FINN_RESOURCES_HLSLIB`). `FINN_RESOURCES_DIR` selects writable
storage; `FINN_RESOURCES_SYSTEM_CACHE` is read-only image storage.
`FINN_RESOURCES_OFFLINE=1` forbids downloads, including explicit fetch commands.
For offline preparation, fetch into a transferable cache with `--dest PATH`,
then mount/select that directory. Create a custom host cache directory before selecting it for a Docker mount. Additional resource declarations can come from
`FINN_RESOURCES_FILES`, project TOML or package entry points.

The five upstream board repositories are assembled into the original flat
layout in the writable resource cache. `FINN_BOARD_FILES_PATH` can select an
already assembled site tree. Images built locally contain the non-redistributable
board sources; the release target omits them. Prepare a local writable board
cache for offline SIF use. Redistributable hlslib and experimental Python sources
are verified at the upstream commits; no FinnLib or QONNX fork is required.

The pinned finn-experimental repository declares `packages=find:` despite a
namespace-only source root, so its wheel contains no Python modules. The small
`packaging/finn-experimental` adapter builds its verified, unchanged sources as a
namespace wheel. This replaces the old editable-clone workaround and can be
removed when upstream fixes that package. Its licence ships with the wheel.

Two migration corrections are explicit: Sphinx 5.0.2 requires docutils below
0.19, so the lock selects 0.18.1; finn-experimental needs the packaging adapter
above. Other direct selections remain upstream's, and formerly floating transitives
are now recorded in the lock. The supported platform is Linux x86-64/Python 3.12.

XSI binaries live under `$FINN_HOME/xsi` and are keyed by source, Python ABI,
compiler identity, effective compile/link settings, pybind11 version and Vivado
installation/header identity. Golden reference output lives under
`$FINN_BUILD_DIR/pynq_golden` (`FINN_GOLDEN_DIR` overrides its parent).
`FINN_VCK190_GOLDEN=1` fetches the pinned PYNQ sources and builds a private copy;
it requires real tools and records the inputs. The pinned PYNQ golden Tcl declares Vivado 2025.2, which needs its own licensed-tool validation. Source resources stay immutable.

## Runtime packages and read-only images

Runtime content is independent of `--fpga`. `--runtime xrt` retains upstream's
XRT 2.18.179 userspace. See `runtimes/*.env` for pinned downloads and supplied
SLASH/slashkit package names. Missing supplied files fail the build. Kernel
modules and host daemons remain host responsibilities.

```bash
./docker/build --release --runtime xrt
./docker/build --export-sif /absolute/finn.sif --runtime xrt
```

Release builds snapshot Git-tracked working-tree files, including tracked edits
and deletions. Ignored downloads and untracked notebook/model output are excluded.
Add intended new source files to Git before building a release.

SIF conversion requires a local Apptainer or Singularity executable. The release
image contains a stable upstream source tree, including HDL, HLS, XSI and shell
assets. No startup package installation is needed. Supplied `.deb`/`.rpm` archives are excluded from that source snapshot; only explicitly selected installed runtime content enters the image. Use `apptainer run --bind`
for tool, licence, resource and scratch mounts, and `--env` for the resolved
host variables. Set writable `FINN_BUILD_DIR`, `FINN_HOME`, `HOME` and
`UV_CACHE_DIR`; retain `/opt/finn/source` as `FINN_ROOT` and `FINN_SYNC=0`.
Run the image's entrypoint (or `apptainer run`) to activate tools and prepare
resources before executing a compiler command. Board caches must be supplied
for offline use. Read-only execution still requires writable external outputs.

## Interface migration

| Previous input | Replacement |
| --- | --- |
| `run-docker.sh` | `docker/run [--fpga] [--runtime NAME] -- COMMAND...` |
| `print-tag` | `docker/build --print-tag` |
| `test`, `quicktest` | Explicit `python -m pytest`, or `bash docker/quicktest.sh` with the same mode |
| `build_custom DIR FLOW` | Explicit `--volume`, `--workdir`, `-- python FLOW.py` |
| `notebook` | `docker/run --notebook`; ports/password remain configurable |
| `scripts/finn-env.sh` | `scripts/activate.sh` |
| `fetch-repos.sh`, Python clones in `deps/` | uv source selections and verified resources |
| `requirements.txt`, `docs/requirements.txt` | Dependency groups and `uv.lock`; generate exports with uv if needed |
| `FINN_DOCKER_TAG` | Input-derived tags; explicit `FINN_IMAGE` must match environment labels |
| `FINN_DOCKER_PREBUILT`, `FINN_DOCKER_SHARED_IMAGE_DIR` | Retained, with required provenance and execution by verified image ID |
| `FINN_DOCKER_CACHE_DIR`, `FINN_DOCKER_RUN_AS_ROOT`, verification and SSH inputs | Retained; SSH mounting is explicit and read-only |
| `FINN_DOCKER_BUILD_EXTRA` | Bake options, e.g. `--set *.args.UBUNTU_TAG=...`; set `UBUNTU_TAG` so the identity matches |
| `FINN_DOCKER_NO_CACHE` | `docker/build --rebuild` |
| `FINN_XRT_PATH` | Retained checksum-verified public download cache; populated and reused for local XRT builds |
| `SLASHKIT_DEB_PACKAGE` | Retained site package path; atomically staged in `docker/packages/` and selects the `slashkit` runtime |
| `FINN_SKIP_XRT_DOWNLOAD`, `XRT_DEB_VERSION` | Explicit runtime selection and pinned manifests; use `SOURCE=supply` for site packages |
| `FINN_SKIP_DEP_REPOS`, `FINN_SKIP_BOARD_FILES` | Prepared resource cache/overrides and `FINN_RESOURCES_OFFLINE=1` |
| `DOCKER_INTERACTIVE` | TTY detection; explicit Docker flags through `FINN_DOCKER_EXTRA` |
| `FINN_SINGULARITY` | `docker/build --export-sif PATH` and standard Apptainer/Singularity launch |

If `SLASHKIT_DEB_PACKAGE` is set, its file must be readable wherever the launcher
runs, including prebuilt-image consumers, to stage and hash identical bytes.

Every in-tree consumer is migrated. External job DSL/site wrappers must update
with this change; there is no second pip/clone-based environment behind a legacy
launcher. Old two-file image publications must be rebuilt before cutover.
See [CI integration](../ci/README.md) for artifact identity and Brevitas candidates.
