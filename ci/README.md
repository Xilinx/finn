# FINN Jenkins CI guide

## How the pipeline works

The [Jenkinsfile](./Jenkinsfile) is a declarative pipeline of four stages, delegating most logic to the `finn_ci` Python package.

1. **Validate**: computes the sharding plan once, prepares a timing snapshot from historical records, checks executor budget, and prunes the shared trees.
2. **Build Docker Image**: builds the FINN image with `run-docker.sh` and publishes it to NFS (if `FINN_CI_NFS_ROOT` is set) so the test shards load it instead of rebuilding.
3. **Run Tests**: fans out one parallel branch per shard. Each branch runs `python -m pytest -m <marker> --num-shards N --shard-id i` inside the container and stashes results/artifacts.
4. **Check Stage Results** unstashes every shard's reports, aggregates one board zip per `(hwTestType, board)`, and refreshes the persistent timing master file.

Terms used throughout:

- **row** is one entry in the `STAGES` list in [finn_ci/config.py](./finn_ci/config.py). Here is where you can configure the sharding/worker policy.
- **stage** means either the four Jenkins pipeline stages above or a row's parallel branch name (e.g. "fpgadataflow (1/2)").
- **shard** is one slice of a row's tests, selected with `--num-shards` and `--shard-id`, running as its own parallel branch on its own agent workspace.
- **stash** is the per-shard report bundle, containing a shard's JUnitXML, HTML, timings, and shard-map sidecar files.
- **group** refers to a `@pytest.mark.xdist_group`. The test-to-shard assignment logic always keeps these groups together.
- **LSF** is IBM's "Load Sharing Facility", the compute-farm infrastructure FINN can use to offload heavy EDA work during CI runs. It is completely optional.

---

## For contributors and test authors

For external contributors who would like to write or edit tests in FINN:

### Run the tests locally

You do not need Jenkins to run the same tests locally. From a checkout:

```bash
./run-docker.sh python -m pytest -m sanity_bnn
```

substituting any marker from the `STAGES` table in [finn_ci/config.py](./finn_ci/config.py). The sharding flags are optional and change nothing when omitted. If running tests in parallel locally with `-n <N>` (i.e. multiple workers), add `--dist loadgroup` too, so the checkpoint-linked tests stay on one worker.

### Add a new test

Decorate it with the existing markers. For example, `@pytest.mark.fpgadataflow`. The next CI run picks it up automatically. If the test reuses a checkpoint another test produces (loaded with `load_test_checkpoint_or_skip`), put both under the same `@pytest.mark.xdist_group(...)` so the sharder keeps them on one worker.

### Clean up scratch in a test

Do not write scratch into the current working directory. Use FINN's `make_build_dir()` and tear it down with the `robust_rmtree()` helper. Successful tests should remove disposable scratch, while failed tests can keep it for diagnosis.

### Add a new BNN parameter value

Edit the `_BNN_WBITS`, `_BNN_ABITS`, and `_BNN_TOPOLOGY` constants in `tests/end2end/test_end2end_bnn_pynq.py`. Nothing else is needed.

---

## For privileged Jenkins users

### Trigger a build

The job DSL targets "dev" by default. Currently, targeting a different branch means editing the DSL and running a seed job.

> See [Known limitations](#known-limitations) for the plan to remove this manual step.

To start a build, click *Build with Parameters* and select the stage you would like to run.


| `STAGES` value     | Rows that run            | Use when                                      | Needs `FINN_CI_NFS_ROOT`?                                      |
| ------------------ | ------------------------ | --------------------------------------------- | -------------------------------------------------------------- |
| `sanity` (default) | Sanity rows only         | Per-PR quick check                            | Recommended (publishes `bnn_build_sanity` zips for HW handoff) |
| `full`             | Every CI row             | Nightly / pre-merge full matrix               | Yes (otherwise no handoff and no timing master update)         |
| `fpgadataflow`     | fpgadataflow row(s) only | Only build-side debug, no HW handoff produced | No                                                             |
| `end2end`          | end2end + BNN rows only  | Debugging just the end2end family             | Recommended (BNN rows publish `bnn_build_full` zips)           |


The above table is unit tested for drift against the actual stage tables. Bitstream artifact handoff is skipped if an NFS root directory is not set. Note that "sanity" is the only stage that will be available directly after running a seed job.

`local_setup` is another stage that can be added for non-Docker tests. Set `FINN_LOCAL_BUILD_LABEL` in the DSL to bind the stage to an agent that has the requisite dependencies set up.

### Debug one stage

Trigger a build with the matching `STAGES` value and use `STAGE_FILTER` in the GUI to match via substring to the shard's display name, for example, `STAGE_FILTER=BNN U250`.

### Pin a test to a specific shard

`@pytest.mark.shard(N)` pins a test and any `xdist_group` siblings to shard N.

### Find which stage and shard runs a given test

For an archived Jenkins build, open `reports/shard_map.txt` and grep for the nodeid or any useful substring. The row format is in the Reference section.

---

## For maintainers

### Infrastructure configuration

`FINN_CI_NFS_ROOT` (shared storage directory) is the only CI-pipeline-specific env var a Jenkins operator sets, and it is optional. Everything else derives from it. Wire it into the job DSL as a global env variable. When unset, the pipeline still runs but with the following features degraded:

- No shared Docker image cache (each agent rebuilds locally)
- No build-to-HW artifact handoff (HW pipeline won't be able to use this build)
- No persistent timing master file (sharding falls back to round-robin)

Optional CI-related overrides are listed below, with sensible defaults:


| Env var                    | Defaults to                            | What it changes                                                                                                             |
| -------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `FINN_LOCAL_BUILD_LABEL`   | `finn-build`                           | Agent label for the optional non-Docker `setup-local.sh` stage, requires a host with dependencies in place.                 |
| `FINN_CI_LOCAL_CACHE_ROOT` | `${WORKSPACE_TMP:-/tmp}/finn-ci-cache` | Pip + XDG cache root for the same non-Docker stage.                                                                         |
| `FINN_CI_MIN_FREE_GB`      | `120`                                  | Minimum free space (GB) on the agent scratch volume below which a shard refuses to start.                                   |
| `FINN_LSF_NFS_STAGING`     | unset                                  | Staging area for LSF jobs, setting this variable enables a range of LSF functionality. Only needed if using an LSF cluster. |

### Test configuration

Every parallel stage is defined by one row of `STAGES` in [finn_ci/config.py](./finn_ci/config.py). The Jenkinsfile loads the entire config bundle during `Validate`, and the pytest plugin responsible for distributing the shards at the executor level ([finn_ci/plugin.py](./finn_ci/plugin.py)) consumes the same configuration.

A `STAGES` row `marker` is restricted to an "a or b or c" pattern because it is interpolated into a shell `-m` argument, so `and`/`not` are rejected. This only constrains `STAGES` rows. Ad-hoc runs such as `pytest -m "fpgadataflow and not slow"` are unaffected and can still be sharded locally.

If a stage is completing slowly, it may be possible to speed it up by increasing the shard or worker count.

### Adding a new CI param

`finn_ci.config.STAGES` rows carry a `param` field that maps onto the `STAGES` Jenkins choice. For instance, "sanity" in Jenkins maps to 'Sanity - Build Hardware', 'Sanity - Unit Tests'. To add a new family, for instance: "quantization":

1. Add `STAGES` rows with `"param": "quantization"`.

   ```
   {
       "param": "quantization",
       "stage": "Quantization - Brevitas",
       "marker": "quant_brevitas",
       "shards": 2,
       "workers": 8,
       "coverage": True,
   }
   ```

2. Run `PYTHONPATH=ci python3 -m finn_ci stage-choices-json` and copy the generated list into the Jenkinsfile's `choice` block. There is a util test that catches drift, should it occur.
3. Add a row to the `STAGES` table in this README (any drift in this table is also tested).

After those three edits, a user picking `STAGES=quantization` in Jenkins gets

```
choice quantization -> rows ['Quantization - Brevitas']
```

### Adding a new BNN board

1. Add the marker `bnn_<board>` to `setup.cfg` under `[tool:pytest]`.
2. In [finn_ci/config.py](./finn_ci/config.py), add a `BOARDS` entry, plus a `STAGES` row that references the board in its `zipArtifacts.boards`. `tests/end2end/test_end2end_bnn_pynq.py` reads `BOARDS[board]["bnnMarker"]`, so the board's scenarios are parametrised automatically.
3. Nothing else is needed. `validate_config()` sanity-checks each `STAGES`/`BOARDS` row.

### Running tools on LSF (optional)

Each shard runs safely as a parallel branch on whatever `finn-build` executor picks it up, so adding capacity is as simple as adding more machines and agents under that label. For this reason, integration with an LSF cluster is not required to run FINN's CI.

However, the intended long-term operational model for FINN CI is a single FINN build machine running several shards at once, delegating any heavy tasks to a compute farm. A tool interception hook has been provided for this reason at `finn.util.basic.resolve_xilinx_tool()`. The agent still drives the FINN flow and pytest, but each `vivado` / `v++` / `vitis_hls` / `xelab` invocation is wrapped with a deployment-specific shim that can delegate heavy subprocesses. The interception hook is generic and can be adapted for a variety of HPC models.

If using IBM's LSF, the pipeline cooperates with such a wrapper through one env var, `FINN_LSF_NFS_STAGING`. When it is set:

- The Jenkinsfile reaps orphaned `bsub` jobs left by an aborted build, both on this build's completion and at the next build's Validate. The site's wrapper must tag every job name `finn_ci_<jobScope>_<TOOL>_<JOB_TAG>` so the reaper can find them and `bkill` only the jobs whose submitting build is no longer running.
- `archive_failure_logs.sh` tails the LSF staging-dir logs into a failed shard's bundle, so a farm-side tool failure is visible in Jenkins without opening the cluster.

When `FINN_LSF_NFS_STAGING` is unset (the default) both behaviours are skipped.

### Sharding and timing state

Shard balancing is managed dynamically and automatically using historical data. A cold start will fall through to round-robin shard assignment. A persistent master timing file is refreshed by any build that ran fully (not aborted or interrupted), including runs with build failures.

The timing master schema is `{"schema_version": 1, "updated_at": ..., "last_update": {...}, "groups": {<name>: {"samples": [last MAX_SAMPLES observations]}}}`. Each qualifying build appends one observation per observed xdist_group/test and trims the window to the five most recent samples. The weight used by the shard assignment logic is the **max** inside the window. For instance, if a group took the following amounts of times in the previous five runs:

```
25 min, 30 min, 18 min, 40 min, 22 min
```

the weight assigned to that test would be *40 minutes*.

This guards conservatively against under-provisioning while bounding how long an outlier will affect timings. A corrupt or unreadable master timing file is logged and treated as empty, so the build degrades to deterministic round-robin sharding and the next build repopulates the master from its own observations. The master is disposable, so a once-off problem self-heals without manual cleanup.

To inspect timing state, open `reports/ci_timings_master.json` from any archived build.

### Build-to-HW zip handoff

The build pipeline stages board deployment directories per shard, then "Check Stage Results" aggregates those staged deployments into one board bitstream zip plus a `.READY` marker in the per-build directory.

```
${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD>/
      zips/<hwTestType>/<board>.zip
      zips/<hwTestType>/<board>.zip.READY
      BUILD_INFO.txt
      deployments/<hwTestType>/<board>/<stash>/<board>/<model>/
```

The `.READY` marker is the build-to-HW handshake. It is touched only after the aggregated zip has been renamed into place. `FINN_CI_NFS_ROOT` is required for any build run that expects bitstream inputs (for example, Jenkinsfile_HW).

> Note: Jenkinsfile_HW hasn't been migrated yet. See [Known limitations](#known-limitations).

A `STAGES` row that produces these zips declares a `zipArtifacts` nested key:

```python
"zipArtifacts": {"hwTestType": "bnn_build_full", "boards": ["U250"]}
```

`hwTestType` (today `bnn_build_sanity` or `bnn_build_full`) selects which HW pipeline category the zip feeds. `boards` lists the board zips the row produces. The nested shape means the pair is either present or absent.

`BUILD_INFO.txt` is simply a human-readable provenance record of the build.

### Storage and retention

Set `FINN_CI_NFS_ROOT` once on the Jenkins controller (in the job DSL) and the build pipeline derives every shared subtree from it. There are no other CI storage env vars to set. Layout under `FINN_CI_NFS_ROOT`:

```
agent_caches/<NODE>/{xrt,finn_cache,vivado_ip_cache}   per-agent caches
docker_images/<jobKey>/<BUILD>/                        shared docker image
artifacts/ci_runs/<jobKey>/<BUILD>/                    build-to-HW handoff + BUILD_INFO
_ci_state/<jobKey>/                                    timing master + snapshots
```

Per-shard scratch lives at `${WORKSPACE_TMP}/finn_ci_runs/<BUILD>/<stash>` (falling back to `${WORKSPACE}/tmp/finn_ci_runs/...` only when `WORKSPACE_TMP` is unset). The workspace itself is per-agent and configured on the DSL side: NFS-mounted via `remote_fs` on the lab build hosts, local SSD elsewhere.

The "Validate" stage rotates the image, artifact, and timing-snapshot trees via the single `rotateBuildTrees()` helper in [Jenkinsfile](./Jenkinsfile). Each rotation keeps the newest N numeric entries and the current build, and deletes older entries whose mtime exceeds M days. All three subcommands skip silently when their parent directory does not exist, and the Python side tolerates concurrent prune races.

---

## Known limitations

- **Targeting a non-default branch needs a DSL edit.** The job DSL targets "dev" by default, so testing a different branch (for example a PR branch) currently means editing the DSL and running a seed job. The intended fix is to target a PR branch without hand-editing the DSL.
- **Jenkinsfile_HW is not migrated yet.** It will continue working with any existing artifacts in the legacy `ARTIFACT_DIR`, but won't work with the new aggregated board zips until it is migrated. The intention is that the HW test runs with the newest valid board zip available, and is marked UNSTABLE if the newest zip wasn't created by the last run (i.e. there was some error in the build stage that is unrelated to HW tests).

---

## Reference

### Artifacts

- `reports/*.xml`, `reports/*.html` from pytest and `pytest_html_merger`.
- `reports/<stash>.timings.json` per shard.
- `reports/<stash>.shardmap.txt` and `reports/<stash>.shardmap.json` per shard.
- `reports/shard_map.txt` and `reports/shard_map.json` merged across all shards.
- `reports/ci_timings_master.json` archived timing preview from this build. Its `last_update` field records the observed group count and whether the shared master was updated.
- `reports/<stash>.empty-shard` per shard that collected zero items. Useful for distinguishing "shard had no work" from "shard crashed".
- `coverage_combined/` one merged HTML report across all rows with `coverage: true`. Per-shard pytest runs write raw `.coverage` data files (one per shard, named via `COVERAGE_FILE=<stash>.coverage`), `aggregateReports` runs `coverage combine` and `coverage html` on the union, and the merged result is archived. Skipped silently when no row opted in.
- `${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD_NUMBER>/zips/<hwTestType>/<board>.zip` per row with a `zipArtifacts` entry. `aggregateReports()` runs `assertZipArtifactsEmitted()` which marks the build UNSTABLE (non-fatal) when an active row declared `zipArtifacts` but no `.READY` was written.
- `${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD_NUMBER>/zips/<hwTestType>/<board>.zip.READY` per-board handshake marker, touched only after the zip is in place. Publishing is idempotent for same-build retries.
- `${FINN_CI_NFS_ROOT}/artifacts/ci_runs/<jobKey>/<BUILD_NUMBER>/BUILD_INFO.txt` for human traceability.

### shard_map.txt row format

Each row of `reports/shard_map.txt` is grep-friendly:

```text
nodeid=<nodeid> stage=<stage> shard=<i>/<n> stash=<stash> group=<group> weight_s=<seconds> source=<known|fallback|pinned|round_robin|single>
```

### DSL environment variables

These are the other env vars a job DSL typically sets for a build-pipeline job, on top of the CI-specific ones in "Infrastructure configuration" (`FINN_CI_NFS_ROOT` and the optional overrides). They are consumed by `run-docker.sh` and the FINN flow rather than by the pipeline itself, so the defaults and meanings match a normal local `run-docker.sh` run.

| Env var               | What it sets                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FINN_XILINX_PATH`    | Path to the Xilinx tools install. `run-docker.sh` warns when unset, and Vivado/Vitis/HLS steps need it.                                                       |
| `FINN_XILINX_VERSION` | Xilinx tool version (for example `2022.2`).                                                                                                                   |
| `PLATFORM_REPO_PATHS` | Vitis platform (DSA) files, required for Vitis-based Alveo cards.                                                                                             |
| `FINN_DOCKER_EXTRA`   | Extra `docker run` arguments (bind mounts, licence, network, and any `-e` vars the tool-dispatch layer needs). The pipeline appends a per-agent `--hostname` and the NFS cache mounts to whatever the DSL sets. |
| `NUM_DEFAULT_WORKERS` | Default xdist worker count for ad-hoc runs. Per-shard worker counts come from `STAGES`, not this.                                                             |

A site that offloads the heavy Xilinx tools to a compute farm (see "Running tools on LSF") needs no pipeline changes. The tool wrapper and its configuration ride into the container through `FINN_DOCKER_EXTRA`, and the only variable FINN itself reads is the shim-directory override below:

| Env var                  | What it sets                                                                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FINN_TOOL_DIR_OVERRIDE` | Shim directory. `finn.util.basic.resolve_xilinx_tool()` resolves `vivado`/`v++`/`vitis_hls`/`vitis-run`/`xelab` to `<dir>/<tool>` when set.        |

The wrapper's own variables are deployment-specific.
