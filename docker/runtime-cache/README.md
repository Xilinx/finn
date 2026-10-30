This build-context directory stages only the checksum-verified public XRT
package from `FINN_XRT_PATH`. Its archive is ignored by Git, omitted from
release sources, and does not change image identity: the runtime manifest
already pins its exact checksum. Supplied private runtimes belong in
`docker/packages/` instead.
