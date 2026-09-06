// Shared helpers loaded by both Jenkinsfile (build pipeline) and Jenkinsfile_HW.
// Helpers that diverge between the two expose distinct entry points.

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

// Sole shell-quoting primitive for both pipelines. Wraps the argument in single
// quotes and escapes embedded ones via the canonical '"'"' dance.
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

// Single stash-with-catchError primitive for both pipelines. requireFile, if
// given, gates the stash on that file existing. allowEmpty controls the stash step.
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

// HW reports are named ${testType}_hw_${board} but stashed as ${testType}_${board},
// so fileBase is passed explicitly.
void safeStashHwReport(String stashName, String fileBase) {
  _stashReport(stashName, "${fileBase}.xml,${fileBase}.html", false, "${fileBase}.xml")
}

// Counts the agents carrying `labelName` and how many are online. @NonCPS and
// plain ints out, so no non-serialisable LabelAtom or Node is live when CPS
// persists the program. @NonCPS does not exempt this from Script Security, so
// every call must be an approved signature and no pipeline step belongs here.
@NonCPS
private List<Integer> _countLabelAgents(String labelName) {
  // matched by object identity, because Label.getName() is not approved
  def label = Jenkins.instance.getLabel(labelName)
  int total = 0
  int online = 0
  for (node in Jenkins.instance.getNodes()) {
    if (!node.getAssignedLabels().contains(label)) { continue }
    total++
    def computer = node.toComputer()
    if (computer != null && computer.isOnline()) { online++ }
  }
  return [total, online]
}

// Is any agent carrying `labelName` online? A Script Security rejection reports
// offline with an approval hint rather than crashing the stage.
boolean isNodeOnline(String labelName) {
  try {
    List<Integer> counts = _countLabelAgents(labelName)
    int total = counts[0]
    int online = counts[1]
    if (total == 0) {
      echo "Node with label ${labelName} not found"
      return false
    }
    if (online == 0) {
      // counted rather than named, because Node.getDisplayName() is not approved
      echo "All ${total} agent${total == 1 ? '' : 's'} with label ${labelName} offline"
      return false
    }
    return true
  } catch (org.jenkinsci.plugins.scriptsecurity.sandbox.RejectedAccessException e) {
    echo "isNodeOnline(${labelName}): Jenkins API rejected by Script Security (${e.message}). " +
         "Treating as offline; approve Jenkins.instance.getLabel, Jenkins.instance.getNodes, " +
         "Node.getAssignedLabels, Node.toComputer and Computer.isOnline in Manage Jenkins to " +
         "restore the check."
    return false
  } catch (Exception e) {
    echo "isNodeOnline(${labelName}): query failed (${e.class.name}: ${e.message}), treating as offline"
    return false
  }
}

// Does this interruption carry the timeout step's cause? @NonCPS and a plain
// boolean out, so no CauseOfInterruption is live when CPS persists the program.
// @NonCPS does not exempt this from Script Security either.
@NonCPS
private boolean _hasTimeoutCause(Throwable e) {
  for (cause in ((org.jenkinsci.plugins.workflow.steps.FlowInterruptedException) e).getCauses()) {
    if (cause instanceof org.jenkinsci.plugins.workflow.steps.TimeoutStepExecution.ExceededTimeout) {
      return true
    }
  }
  return false
}

// Did a timeout step raise this, rather than an operator's stop, a removed agent
// or a cancelled queue item? Anything unrecognised answers false, so a caller
// acting on a timeout cannot act on somebody else's interruption.
boolean isTimeoutInterruption(Throwable e) {
  try {
    return _hasTimeoutCause(e)
  } catch (org.jenkinsci.plugins.scriptsecurity.sandbox.RejectedAccessException e2) {
    echo "isTimeoutInterruption: Jenkins API rejected by Script Security (${e2.message}). " +
         "Treating as a real interruption. Approve " +
         "org.jenkinsci.plugins.workflow.steps.FlowInterruptedException getCauses in Manage " +
         "Jenkins to tell a bound firing from an abort."
    return false
  } catch (Exception e2) {
    echo "isTimeoutInterruption: cause query failed (${e2.class.name}: ${e2.message}), treating " +
         "as a real interruption"
    return false
  }
}

// Hard-fail on root-owned residue. Factored out so the build and HW forms
// below cannot diverge on the error message or detection logic.
void _assertNoResidue(String caller, String q) {
  sh """
    if [ -d ${q} ]; then
      echo "${caller}: ${q} still exists after rm. Likely root-owned residue. Ask an admin to 'sudo rm -rf' the directory on this agent."
      ls -la ${q} | head -40
      exit 1
    fi
  """
}

// Build pipeline form: tolerant rm, hard-fail on root-owned residue, then
// pre-create as the unprivileged user so docker -v does not bind the mount as root.
void cleanPreviousBuildFiles(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String q = shellQuote(buildDir)
  sh "rm -rf ${q} 2>/dev/null || true"
  _assertNoResidue('cleanPreviousBuildFiles', q)
  sh "mkdir -p ${q}"
}

// HW per-board workdir form: rm the build dir and its sibling .zip, with sudo when
// HW credentials are bound, since board agents can leave root-owned residue.
void cleanBoardWorkdirHw(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String prefix = env.USER_CREDENTIALS ? 'echo "$USER_CREDENTIALS_PSW" | sudo -S ' : ''
  String q = shellQuote(buildDir)
  String qZip = shellQuote(buildDir + '.zip')
  sh "${prefix}rm -rf ${q} ${qZip}"
  _assertNoResidue('cleanBoardWorkdirHw', q)
}

// HW reports-dir form: rm the dir only. No sudo, since this runs on the aggregator
// rather than on a board.
void cleanReportsDirHw(String dir) {
  if (!dir || dir.empty) { return }
  String q = shellQuote(dir)
  sh "rm -rf ${q}"
  _assertNoResidue('cleanReportsDirHw', q)
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
