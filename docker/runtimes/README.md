# Runtime targets

A runtime target is a set of `.deb` files installed into the image on top of
the base. It is how FINN supports a specific accelerator stack: XRT for
Vitis/Alveo, SLASH for the V80.

Select them at build time. The names go into the tag, sorted:

```bash
FINN_RUNTIMES=xrt docker buildx bake -f docker-bake.hcl finn-xrt
#   -> xilinx/finn:<git>.xrt
```

## Add a runtime target

Write one file, `docker/runtimes/<name>.env`. The parameterized
`finn-runtime`/`finn-sbx-runtime` Bake targets and launchers can use it without a
new named target. Add an explicit named target only when the combination should
join the supported CI matrix.

| key | required | meaning |
|---|---|---|
| `SOURCE` | yes | `fetch` or `supply` |
| `DEB` | `fetch` | package base name, without `.deb` |
| `URL` | `fetch` | where to get it |
| `SHA256` | no | checked when set; a warning prints the real sum when not |
| `FILES` | `supply` | one or more file names in `docker/packages/` |
| `EXPECT_OS` | no | `VERSION_ID` the package targets, e.g. `24.04` |

`SOURCE=fetch` downloads the package into the build. Use it only for a stable,
published URL.

`SOURCE=supply` requires the user to put the file in `docker/packages/`. A
missing file stops the build and the error names the path. It is never
silently absent.

`EXPECT_OS` is compared with `/etc/os-release` in the image. A mismatch stops
the build. Install a package that targets a different release and apt will
usually resolve it rather than refuse, which gives a subtly wrong runtime
instead of a clean failure.

## What belongs here

Add a runtime target only if code that runs INSIDE the image links or execs
against it.

- **XRT** — `v++` links against it and `make_driver.py` calls `xclbinutil`.
- **SLASH** — `MakeCPPDriver` builds `finn-vrt-driver` against VRT.
- **slashkit** — the Slash Alveo link flow executes this linker in the image.
- **PyNQ** — does NOT belong. `from pynq import ...` occurs only in
  `src/finn/deploy/data/pynq_driver/driver_base.py`, which is a template
  copied to the board. Nothing in the image imports it.

## What does not belong here

**Kernel modules and daemons.** Every runtime target has a host half: `xocl`
and `xclmgmt` for XRT, the SLASH kernel module and `vrtd` for SLASH. A
container cannot own them. Install them on the host.

**Source builds.** FINN fetches published artifacts and accepts supplied
packages. It does not build third-party runtime stacks. A user with a V80 has
already built and installed SLASH on the host, because the kernel module has to
be there, so the SLASH and slashkit `.deb` files are by-products they already
hold. FINN has no V80 to test a build against.

**The Xilinx toolchain.** Vivado and Vitis are host facts. `docker/config.py`
resolves them at launch and the lane mounts them read-only.
