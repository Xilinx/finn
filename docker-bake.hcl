# Environment inputs determine development tags; FINN source is mounted at run time.
variable "FINN_IMAGE_REVISION" { default = "unresolved" }
variable "FINN_SOURCE_REVISION" { default = "unknown" }
variable "FINN_SOURCE_DIRTY" { default = "unknown" }
variable "REGISTRY" { default = "xilinx/finn" }
variable "FINN_RUNTIMES" { default = "" }
variable "FINN_UV_NO_CACHE" { default = "0" }
variable "UBUNTU_TAG" { default = "noble-20240605" }
variable "FINN_RELEASE_SOURCE_DIR" { default = ".cache/release-source" }
function "runtime_set" {
  params = [runtimes]
  result = join(",", distinct(sort(compact(split(",", runtimes)))))
}
function "runtime_suffix" {
  params = [runtimes]
  result = runtime_set(runtimes) == "" ? "" : ".${join(".", split(",", runtime_set(runtimes)))}"
}
target "_common" {
  dockerfile = "docker/Dockerfile.finn"
  context    = "."
  # XRT and Vivado are x86-64 only; so is the LSB loader alias in the image.
  platforms = ["linux/amd64"]
  args = {
    FINN_UV_NO_CACHE     = FINN_UV_NO_CACHE
    UBUNTU_TAG           = UBUNTU_TAG
    FINN_SOURCE_REVISION = FINN_SOURCE_REVISION
    FINN_SOURCE_DIRTY    = FINN_SOURCE_DIRTY
  }
}

function "labels" {
  params = [runtimes]
  result = {
    "org.opencontainers.image.title" = "FINN"
    "org.opencontainers.image.source" = "https://github.com/Xilinx/finn"
    "dev.finn.image-revision" = FINN_IMAGE_REVISION
    "dev.finn.runtimes" = runtime_set(runtimes)
    "dev.finn.source-mode" = "checkout"
  }
}

# Build the large locked environment once, independent of runtime labels/args.
# Named target contexts make this sharing explicit across the whole Bake graph.
target "_python" {
  inherits = ["_common"]
  target = "python"
  output = ["type=cacheonly"]
}

target "_prepared" {
  inherits = ["_common"]
  contexts = { finn-python = "target:_python" }
}

target "finn" {
  inherits = ["_prepared"]
  target = "dev"
  args = { FINN_RUNTIMES = "" }
  labels = labels("")
  tags = ["${REGISTRY}:${FINN_IMAGE_REVISION}${runtime_suffix("")}"]
}

target "finn-xrt" {
  inherits = ["_prepared"]
  target = "dev"
  args = { FINN_RUNTIMES = "xrt" }
  labels = labels("xrt")
  tags = ["${REGISTRY}:${FINN_IMAGE_REVISION}${runtime_suffix("xrt")}"]
}

target "finn-runtime" {
  inherits = ["_prepared"]
  target = "dev"
  args = { FINN_RUNTIMES = runtime_set(FINN_RUNTIMES) }
  labels = labels(runtime_set(FINN_RUNTIMES))
  tags = ["${REGISTRY}:${FINN_IMAGE_REVISION}${runtime_suffix(runtime_set(FINN_RUNTIMES))}"]
}

target "finn-release" {
  inherits = ["_prepared"]
  contexts = { finn-source = FINN_RELEASE_SOURCE_DIR }
  target = "release"
  args = { FINN_RUNTIMES = runtime_set(FINN_RUNTIMES) }
  labels = merge(labels(FINN_RUNTIMES), {
    "dev.finn.source-mode" = "embedded"
    "org.opencontainers.image.revision" = FINN_SOURCE_REVISION
    "dev.finn.source-dirty" = FINN_SOURCE_DIRTY
  })
  tags = ["${REGISTRY}:release-${FINN_IMAGE_REVISION}-${substr(FINN_SOURCE_REVISION, 0, 12)}${FINN_SOURCE_DIRTY == "1" ? "-dirty" : ""}${runtime_suffix(FINN_RUNTIMES)}"]
}

group "default" { targets = ["finn"] }
group "supported" { targets = ["finn", "finn-xrt"] }
