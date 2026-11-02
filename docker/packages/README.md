# Supplied runtime packages

Put `.deb` files for `SOURCE=supply` runtime targets here. See
`docker/runtimes/README.md`.

The files are NOT committed. `.gitignore` excludes `*.deb` in this directory,
because these are vendor packages that FINN neither owns nor redistributes.

```bash
cp slash.deb docker/packages/
cp slashkit.deb docker/packages/
FINN_RUNTIMES=xrt,slash,slashkit docker buildx bake -f docker-bake.hcl finn-runtime
```

A runtime target that needs a file you have not put here stops the build and
names the path it wanted.
