// Shared helpers loaded by the build pipeline Jenkinsfile.

boolean paramBool(String name) {
  def v = params.get(name)
  if (v == null) { return false }
  if (v instanceof Boolean) { return v }
  return v.toString().toBoolean()
}

String paramString(String name) {
  def v = params.get(name)
  return v == null ? '' : v.toString()
}

// Sole shell-quoting primitive. Wraps the argument in single quotes and
// escapes embedded single quotes.
String shellQuote(String s) {
  return "'" + (s ?: '').replace("'", "'\"'\"'") + "'"
}

// Sets FINN_DOCKER_PREBUILT=1 when a shared image is configured so non-builder
// agents load the image from NFS instead of rebuilding.
void runDockerCommand(String command) {
  if (env.FINN_DOCKER_SHARED_IMAGE_DIR) {
    withEnv(['FINN_DOCKER_PREBUILT=1']) {
      sh command
    }
  } else {
    sh command
  }
}

void unstashIfPresent(String stashName) {
  try {
    unstash stashName
  } catch (Exception ignored) {
    echo "No stash '${stashName}' (stage skipped or failed before publishing)"
  }
}

// Single stash-with-catchError primitive. requireFile, if given, gates the
// stash on that file existing. allowEmpty controls the stash step.
void _stashReport(String stashName, String includes, boolean allowEmpty, String requireFile) {
  catchError(buildResult: null, stageResult: null,
             message: "safeStashReport(${stashName}) failed, aggregation may be partial") {
    if (requireFile && !fileExists(requireFile)) { return }
    stash name: stashName, includes: includes, allowEmpty: allowEmpty
  }
}

// Build pipeline stashes the full per-shard report sidecar set. Some are
// missing when a shard fails early, so allowEmpty is true. The .coverage
// entry only exists on rows that opted into coverage in STAGES.
void safeStashShardReport(String stashName) {
  _stashReport(
    stashName,
    "${stashName}.xml,${stashName}.html,${stashName}.timings.json," +
    "${stashName}.shardmap.txt,${stashName}.shardmap.json,${stashName}.stagemap," +
    "${stashName}.empty-shard,${stashName}.coverage",
    true,
    null,
  )
}

// Hard-fail on root-owned residue. Factored out so the build forms below
// cannot diverge on the error message or detection logic.
void _assertNoResidue(String caller, String q) {
  sh """
    if [ -d ${q} ]; then
      echo "${caller}: ${q} still exists after rm. Likely root-owned residue. Ask an admin to 'sudo rm -rf' the directory on this agent."
      ls -la ${q} | head -40
      exit 1
    fi
  """
}

// Tolerant rm, hard-fail on root-owned residue, then pre-create as the
// unprivileged user so docker -v does not bind the mount as root.
void cleanPreviousBuildFiles(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String q = shellQuote(buildDir)
  sh "rm -rf ${q} 2>/dev/null || true"
  _assertNoResidue('cleanPreviousBuildFiles', q)
  sh "mkdir -p ${q}"
}

// All shared NFS subtrees derive from FINN_CI_NFS_ROOT. Returning '' from any
// resolver means "no NFS available". Callers must handle that as a fallback.
String finnCiNfsRoot() { return (env.FINN_CI_NFS_ROOT ?: '').trim() }

String finnSubdir(String... segments) {
  String r = finnCiNfsRoot()
  if (!r) { return '' }
  for (int i = 0; i < segments.length; i++) {
    if (!segments[i]) { return '' }
  }
  return ([r] + (segments as List)).join('/')
}

String finnAgentCachesDir(String node)    { return finnSubdir('agent_caches', node) }
String finnDockerImagesRoot()             { return finnSubdir('docker_images') }
String finnDockerImagesDir(String jobKey) { return finnSubdir('docker_images', jobKey) }
String finnArtifactsRoot()                { return finnSubdir('artifacts') }
String finnCiStateRoot()                  { return finnSubdir('_ci_state') }
String finnCiStateDir(String jobKey)      { return finnSubdir('_ci_state', jobKey) }

// Append `value` to the list at `map[key]`, creating the list lazily.
// Replaces the inline `Map.computeIfAbsent` idiom because CPS does not
// reliably transform SAM closures to java.util.function.Function. Returns
// the (possibly newly created) list so callers can chain.
List mapAppend(Map map, Object key, Object value) {
  def existing = map.get(key)
  if (existing == null) {
    existing = []
    map.put(key, existing)
  }
  existing << value
  return existing
}

return this
