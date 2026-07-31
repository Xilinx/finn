# FINN Test Guidelines

Help keep FINN's testing suite fast, deterministic, and parallelisable when writing or modifying tests.

### 1. Scratch paths

Never write scratch files into the repo root or CWD. Collisions are guaranteed when tests run in parallel. Instead:

- Use `make_build_dir()` to allocate a unique directory under `FINN_BUILD_DIR`.
- Use `robust_rmtree()` to tear down the test.
- If the test's outputs are useful for diagnosis, you may keep the outputs if the test failed.

```python
from finn.util.basic import make_build_dir, robust_rmtree

test_dir = make_build_dir("test_my_feature_")

# later on
if not failed:
    robust_rmtree(test_dir)
```

### 2. Process state & environment

Don't mutate env variables or global process state. Use pytest's `monkeypatch` fixture to override env variables or system attributes.

```python
def test_sim_behaviour(monkeypatch):
    monkeypatch.setenv("VIVADO_PATH", "/custom/path")
    # original VIVADO_PATH restored automatically after the test
```

### 3. Parallel Scheduling (`xdist_group`)

If you have a chain of tests where subsequent stages load the checkpoint from a previous step (using `load_test_checkpoint_or_skip`), they must run on the same worker process.

Group related tests together using the `xdist_group` marker:

```python
@pytest.mark.xdist_group(name="my_feature_chain")
def test_step_1(): ...

@pytest.mark.xdist_group(name="my_feature_chain")
def test_step_2(): ...
```

Run tests with `--dist loadgroup` if running with multiple workers (i.e. `-n <N>`) so that checkpoint chains stay on the same worker.


### 4. Markers

Decorate tests with the existing markers. For example, `@pytest.mark.fpgadataflow`.

*For more detailed marker, pipeline, sharding, and Jenkins configurations, see [ci/README.md](../ci/README.md).*
