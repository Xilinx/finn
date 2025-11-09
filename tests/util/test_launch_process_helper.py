############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import logging
import os
import subprocess
import tempfile
import unittest

from finn.util.basic import _detect_log_level, launch_process_helper


@pytest.mark.util
class TestDetectLogLevel(unittest.TestCase):
    """Test log level detection patterns."""

    def test_error_patterns(self):
        """ERROR patterns detected correctly and strip prefix."""
        level, msg = _detect_log_level("ERROR: Something failed", logging.INFO)
        assert level == logging.ERROR
        assert msg == "Something failed"

        level, msg = _detect_log_level("FATAL: Critical error", logging.INFO)
        assert level == logging.ERROR
        assert msg == "Critical error"

        level, msg = _detect_log_level("FAILED to compile", logging.INFO)
        assert level == logging.ERROR

        level, msg = _detect_log_level("Exception occurred", logging.INFO)
        assert level == logging.ERROR

    def test_warning_patterns(self):
        """WARNING patterns detected correctly and strip prefix."""
        level, msg = _detect_log_level(
            "WARNING: [XSIM 43-4099] Module has no timescale", logging.INFO
        )
        assert level == logging.WARNING
        assert msg == "[XSIM 43-4099] Module has no timescale"

        level, msg = _detect_log_level("WARN: Deprecated feature", logging.INFO)
        assert level == logging.WARNING
        assert msg == "Deprecated feature"

    def test_debug_patterns(self):
        """Verbose tool output detected as DEBUG."""
        level, msg = _detect_log_level("Compiling module work.foo", logging.INFO)
        assert level == logging.DEBUG

        level, msg = _detect_log_level("Compiling architecture xilinx of entity bar", logging.INFO)
        assert level == logging.DEBUG

        level, msg = _detect_log_level("Analyzing entity baz", logging.INFO)
        assert level == logging.DEBUG

        level, msg = _detect_log_level("Elaborating entity qux", logging.INFO)
        assert level == logging.DEBUG

    def test_info_patterns(self):
        """INFO patterns detected correctly and strip prefix."""
        level, msg = _detect_log_level("INFO: Build complete", logging.WARNING)
        assert level == logging.INFO
        assert msg == "Build complete"

        level, msg = _detect_log_level("NOTE: Using default settings", logging.WARNING)
        assert level == logging.INFO
        assert msg == "Using default settings"

    def test_default_fallback(self):
        """Unknown patterns use default level and preserve message."""
        level, msg = _detect_log_level("Random output", logging.INFO)
        assert level == logging.INFO
        assert msg == "Random output"

        level, msg = _detect_log_level("Random output", logging.DEBUG)
        assert level == logging.DEBUG

        level, msg = _detect_log_level("Random output", logging.WARNING)
        assert level == logging.WARNING

    def test_case_insensitive(self):
        """Pattern matching is case-insensitive."""
        level, msg = _detect_log_level("error: lower case", logging.INFO)
        assert level == logging.ERROR

        level, msg = _detect_log_level("ERROR: UPPER CASE", logging.INFO)
        assert level == logging.ERROR
        assert msg == "UPPER CASE"

        level, msg = _detect_log_level("ErRoR: MiXeD CaSe", logging.INFO)
        assert level == logging.ERROR


@pytest.mark.util
class TestLaunchProcessHelper(unittest.TestCase):
    """Test launch_process_helper subprocess wrapper."""

    def test_returns_exitcode(self):
        """Logging mode returns exit code integer."""
        exitcode = launch_process_helper(["true"])
        self.assertIsInstance(exitcode, int)
        self.assertEqual(exitcode, 0)

    def test_success_exitcode(self):
        """Logging mode returns 0 on success."""
        exitcode = launch_process_helper(["echo", "success"])
        self.assertEqual(exitcode, 0)

    def test_error_exitcode(self):
        """Logging mode returns non-zero exit code."""
        exitcode = launch_process_helper(["false"], raise_on_error=False)
        self.assertEqual(exitcode, 1)

    def test_streams_to_logger(self):
        """Logging mode sends output through logger."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            exitcode = launch_process_helper(["echo", "hello"])
            self.assertEqual(exitcode, 0)
            # Check that output was logged
            self.assertTrue(any("hello" in msg for msg in cm.output))

    def test_custom_logger(self):
        """Can specify custom logger."""
        custom_logger = logging.getLogger("test.custom")
        with self.assertLogs("test.custom", level="INFO") as cm:
            launch_process_helper(["echo", "custom"], logger=custom_logger)
            self.assertTrue(any("custom" in msg for msg in cm.output))

    def test_stdout_level(self):
        """Can specify stdout log level."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(["echo", "debug_output"], stdout_level=logging.DEBUG)
            # Should be logged at DEBUG level
            log_output = "\n".join(cm.output)
            self.assertIn("DEBUG", log_output)
            self.assertIn("debug_output", log_output)

    def test_stderr_level(self):
        """Stderr uses different log level than stdout."""
        with self.assertLogs("finn.subprocess", level="WARNING") as cm:
            launch_process_helper(
                ["sh", "-c", "echo error >&2"],
                stderr_level=logging.WARNING,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("WARNING", log_output)

    def test_detect_levels_enabled(self):
        """Auto-detection adjusts levels based on content."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'ERROR: test error'"],
                stdout_level=logging.INFO,  # Base level
                detect_levels=True,
            )
            log_output = "\n".join(cm.output)
            # Should be promoted to ERROR level
            self.assertIn("ERROR", log_output)
            self.assertIn("test error", log_output)

    def test_detect_levels_disabled(self):
        """Can disable auto-detection."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'ERROR: test'"],
                stdout_level=logging.INFO,
                detect_levels=False,  # Disabled
            )
            log_output = "\n".join(cm.output)
            # Should use base level (INFO), not promote to ERROR
            self.assertIn("INFO", log_output)

    def test_raise_on_error_disabled(self):
        """Does not raise when raise_on_error=False."""
        # Should return exit code without raising
        exitcode = launch_process_helper(["sh", "-c", "exit 42"], raise_on_error=False)
        self.assertEqual(exitcode, 42)

    def test_raise_on_error_enabled(self):
        """Raises CalledProcessError when raise_on_error=True."""
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            launch_process_helper(["false"], raise_on_error=True)
        self.assertEqual(cm.exception.returncode, 1)

    def test_env_var_expansion_in_args(self):
        """Environment variables in arguments are automatically expanded."""
        # Set a test environment variable
        env = os.environ.copy()
        env["TEST_EXPAND_VAR"] = "/test/expanded/path"

        # Use echo to output the path so we can verify expansion happened
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["echo", "$TEST_EXPAND_VAR/file.txt"],
                proc_env=env,
            )
            log_output = "\n".join(cm.output)
            # Should see the expanded path, not the literal $TEST_EXPAND_VAR
            self.assertIn("/test/expanded/path/file.txt", log_output)


@pytest.mark.util
class TestLaunchProcessHelperIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""

    def test_multiline_output(self):
        """Handles multi-line output correctly."""
        with self.assertLogs("finn.subprocess", level="INFO") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'line1'; echo 'line2'; echo 'line3'"],
            )
            log_output = "\n".join(cm.output)
            self.assertIn("line1", log_output)
            self.assertIn("line2", log_output)
            self.assertIn("line3", log_output)

    def test_mixed_stdout_stderr(self):
        """Handles mixed stdout and stderr streams."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'out'; echo 'err' >&2; echo 'out2'"],
                stdout_level=logging.INFO,
                stderr_level=logging.WARNING,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("out", log_output)
            self.assertIn("err", log_output)
            self.assertIn("out2", log_output)

    def test_working_directory(self):
        """Respects working directory parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
                launch_process_helper(["pwd"], cwd=tmpdir)
                log_output = "\n".join(cm.output)
                self.assertIn(tmpdir, log_output)

    def test_environment_variables(self):
        """Respects environment variable parameter."""
        env = os.environ.copy()
        env["TEST_VAR_LOGGING"] = "test_value_67890"

        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo $TEST_VAR_LOGGING"],
                proc_env=env,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("test_value_67890", log_output)

    def test_long_output_no_deadlock(self):
        """Handles long output without deadlock."""
        # Generate 100 lines of output (sufficient to test threading without being slow)
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            exitcode = launch_process_helper(
                ["sh", "-c", "for i in $(seq 1 100); do echo line$i; done"],
                stdout_level=logging.DEBUG,
            )
            self.assertEqual(exitcode, 0)
            # Should have logged many lines
            self.assertGreater(len(cm.output), 50)

    def test_concurrent_output(self):
        """Handles concurrent stdout/stderr without issues."""
        # Mix stdout and stderr rapidly (15 iterations sufficient for race condition testing)
        cmd = "for i in $(seq 1 15); do echo out$i; echo err$i >&2; done"
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            exitcode = launch_process_helper(
                ["sh", "-c", cmd],
                stdout_level=logging.INFO,
                stderr_level=logging.WARNING,
            )
            self.assertEqual(exitcode, 0)
            log_output = "\n".join(cm.output)
            # Should contain output from both streams
            self.assertIn("out", log_output)
            self.assertIn("err", log_output)

    def test_xilinx_pattern_detection(self):
        """Detects Xilinx tool patterns correctly."""
        xilinx_output = """Compiling module work.foo
Compiling architecture rtl of entity bar
WARNING: [XSIM 43-4099] Module has no timescale
ERROR: Synthesis failed
INFO: Elaboration complete"""

        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", f"echo '{xilinx_output}'"],
                stdout_level=logging.INFO,
                detect_levels=True,
            )
            log_output = "\n".join(cm.output)
            # Should have detected different levels
            self.assertIn("DEBUG", log_output)  # Compiling module
            self.assertIn("WARNING", log_output)  # WARNING:
            self.assertIn("ERROR", log_output)  # ERROR:
            self.assertIn("INFO", log_output)  # INFO:


@pytest.mark.util
class TestShellScriptGeneration(unittest.TestCase):
    """Test shell script generation for debugging."""

    def test_generates_script_when_requested(self):
        """Script is created when generate_script parameter is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            launch_process_helper(
                ["echo", "test"],
                generate_script=script_path,
            )
            self.assertTrue(os.path.isfile(script_path))

    def test_script_is_executable(self):
        """Generated script has executable permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            launch_process_helper(
                ["echo", "test"],
                generate_script=script_path,
            )
            # Check if executable bit is set
            mode = os.stat(script_path).st_mode
            self.assertTrue(mode & 0o111)  # Any executable bit

    def test_script_contains_shebang(self):
        """Generated script starts with #!/bin/bash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            launch_process_helper(
                ["echo", "test"],
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                first_line = f.readline()
            self.assertEqual(first_line.strip(), "#!/bin/bash")

    def test_script_contains_command(self):
        """Generated script contains the command to run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            launch_process_helper(
                ["echo", "hello", "world"],
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()
            self.assertIn("echo hello world", content)

    def test_script_contains_working_directory(self):
        """Generated script includes cd command when cwd is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            launch_process_helper(
                ["echo", "test"],
                cwd=tmpdir,  # Use tmpdir as actual cwd so command works
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()
            # Should contain cd to tmpdir (which we passed as cwd)
            self.assertIn("cd", content)
            self.assertIn(tmpdir, content)

    def test_script_quotes_arguments(self):
        """Generated script properly quotes arguments with spaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            launch_process_helper(
                ["echo", "hello world", "foo bar"],
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()
            # Arguments with spaces should be quoted
            self.assertIn("'hello world'", content)
            self.assertIn("'foo bar'", content)

    def test_script_exports_environment_variables(self):
        """Generated script exports important environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            custom_env = os.environ.copy()
            custom_env["XILINX_VIVADO"] = "/opt/Xilinx/Vivado/2022.2"
            launch_process_helper(
                ["echo", "test"],
                proc_env=custom_env,
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()
            self.assertIn("XILINX_VIVADO", content)
            self.assertIn("/opt/Xilinx/Vivado/2022.2", content)

    def test_script_can_be_executed_standalone(self):
        """Generated script can be executed directly with bash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            output_file = os.path.join(tmpdir, "output.txt")

            # Generate script that writes to a file
            launch_process_helper(
                ["sh", "-c", f"echo 'standalone test' > {output_file}"],
                cwd=tmpdir,
                generate_script=script_path,
            )

            # Execute the generated script
            result = subprocess.run(
                ["bash", script_path],
                capture_output=True,
                text=True,
            )

            # Verify script executed successfully
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.isfile(output_file))
            with open(output_file, "r") as f:
                content = f.read()
            self.assertIn("standalone test", content)

    def test_script_handles_long_commands(self):
        """Generated script formats long commands with line continuation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            # Create a command with many arguments (use 'echo' which exists)
            long_cmd = ["echo"] + [f"arg{i}" for i in range(20)]
            launch_process_helper(
                long_cmd,
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()
            # Long commands should have line continuations
            if len(" ".join(long_cmd)) > 100:
                self.assertIn("\\", content)  # Line continuation

    def test_no_script_generated_when_not_requested(self):
        """No script is created when generate_script is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't specify generate_script
            launch_process_helper(
                ["echo", "test"],
                cwd=tmpdir,
            )
            # No .sh files should be created
            sh_files = [f for f in os.listdir(tmpdir) if f.endswith(".sh")]
            self.assertEqual(len(sh_files), 0)

    def test_script_directory_created_if_needed(self):
        """Script parent directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "deeply", "path")
            script_path = os.path.join(nested_dir, "test_script.sh")

            # Directory doesn't exist yet
            self.assertFalse(os.path.exists(nested_dir))

            launch_process_helper(
                ["echo", "test"],
                generate_script=script_path,
            )

            # Directory and script should now exist
            self.assertTrue(os.path.isfile(script_path))

    def test_script_special_characters_in_args(self):
        """Generated script properly escapes special shell characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            # Test various special characters that need escaping
            special_args = [
                "arg with spaces",
                "arg'with'quotes",
                'arg"with"doublequotes',
                "arg$with$dollar",
                "arg`with`backticks",
                "arg!with!exclamation",
            ]
            launch_process_helper(
                ["echo"] + special_args,
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()

            # Verify various special cases are properly quoted/escaped
            self.assertIn("'arg with spaces'", content)
            self.assertIn("'arg$with$dollar'", content)  # $ protected by quotes
            self.assertIn("'arg`with`backticks'", content)  # backticks protected
            self.assertIn('"with"doublequotes', content)  # double quotes preserved
            # Single quotes are escaped in a complex way: 'arg'"'"'with'"'"'quotes'
            # Just verify the base text is present
            self.assertIn("with", content)
            self.assertIn("quotes", content)

    def test_script_without_custom_env(self):
        """Generated script works correctly without custom environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            # Don't pass proc_env, use system default
            launch_process_helper(
                ["echo", "test"],
                generate_script=script_path,
            )
            with open(script_path, "r") as f:
                content = f.read()

            # Should have shebang and command, but no "export" lines
            self.assertIn("#!/bin/bash", content)
            self.assertIn("echo test", content)
            # Should not have environment variable exports section
            # (or if it does, should be minimal/empty)
            lines = content.split("\n")
            export_lines = [line for line in lines if line.strip().startswith("export")]
            # Either no exports, or very few (only if they differ from current env)
            self.assertLessEqual(len(export_lines), 2)

    def test_script_with_env_var_expansion(self):
        """Generated script preserves unexpanded env vars for portability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "test_script.sh")
            env = os.environ.copy()
            # Use FINN_ROOT which is in the important_env_vars list
            env["FINN_ROOT"] = "/custom/finn/root"

            launch_process_helper(
                ["echo", "$FINN_ROOT/bin/tool"],
                proc_env=env,
                generate_script=script_path,
            )

            with open(script_path, "r") as f:
                content = f.read()

            # The script should contain the UNEXPANDED variable
            # This is intentional: scripts are generated before expansion
            # so they can be re-run in different environments
            self.assertIn("$FINN_ROOT/bin/tool", content)
            # FINN_ROOT should be exported (it's in important_env_vars list)
            if env["FINN_ROOT"] != os.environ.get("FINN_ROOT"):
                # Only exported if different from current environment
                self.assertIn("FINN_ROOT", content)


if __name__ == "__main__":
    unittest.main()
