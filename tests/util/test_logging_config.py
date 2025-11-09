############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import json
import logging
import shutil
import sys
import unittest

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import make_build_dir


@pytest.mark.util
class TestLoggingConfig(unittest.TestCase):
    """Test build_dataflow logging configuration behavior."""

    def setUp(self):
        """Clear all logger handlers before each test."""
        # Clear handlers from all FINN loggers
        for logger_name in [
            "finn",
            "finn.builder",
            "finn.hls",
            "finn.vivado",
            "finn.xsim",
            "finn.gcc",
        ]:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True

        # Clear root logger handlers
        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)

        root.setLevel(logging.WARNING)

    def tearDown(self):
        """Clean up after each test."""
        self.setUp()  # Reuse cleanup logic

    def _configure_loggers(self, cfg, stdout_override=None):
        """Configure loggers as build_dataflow.py does.

        Args:
            cfg: DataflowBuildConfig
            stdout_override: Optional stream to use instead of sys.stdout (for testing)
        """
        stdout = stdout_override if stdout_override else sys.stdout

        # Set up root logger with file handler
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
            filename=cfg.output_dir + "/build_dataflow.log",
            filemode="a",
        )

        # Configure finn.builder logger (progress messages)
        builder_logger = logging.getLogger("finn.builder")
        builder_logger.setLevel(logging.INFO)
        if cfg.show_progress:
            builder_console = logging.StreamHandler(stdout)
            builder_console.setFormatter(logging.Formatter("%(message)s"))
            builder_logger.addHandler(builder_console)
        builder_file = logging.FileHandler(cfg.output_dir + "/build_dataflow.log", mode="a")
        builder_file.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s")
        )
        builder_logger.addHandler(builder_file)
        builder_logger.propagate = False

        # Configure finn tool loggers (subprocess output)
        finn_logger = logging.getLogger("finn")
        finn_logger.setLevel(logging.DEBUG)

        console_formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")

        if cfg.verbose:
            finn_console_handler = logging.StreamHandler(stdout)
            finn_console_handler.setFormatter(console_formatter)
            finn_console_handler.setLevel(logging.ERROR)
            finn_logger.addHandler(finn_console_handler)

        finn_logger.propagate = True

        # Apply subprocess log level overrides
        all_categories = set()
        if cfg.subprocess_console_levels:
            all_categories.update(cfg.subprocess_console_levels.keys())
        if cfg.subprocess_log_levels:
            all_categories.update(cfg.subprocess_log_levels.keys())

        configured_logger_names = []
        for category in all_categories:
            logger_name = f"finn.{category}"
            configured_logger_names.append(logger_name)
            subprocess_logger = logging.getLogger(logger_name)

            # Convert string levels to int if needed
            console_level = (cfg.subprocess_console_levels or {}).get(category, logging.ERROR)
            if isinstance(console_level, str):
                console_level = getattr(logging, console_level)

            file_level = (cfg.subprocess_log_levels or {}).get(category, logging.DEBUG)
            if isinstance(file_level, str):
                file_level = getattr(logging, file_level)

            # Set logger level to minimum needed by active destinations
            # When verbose=False, console_level is irrelevant (no console handler exists)
            if cfg.verbose:
                subprocess_logger.setLevel(min(console_level, file_level))
            else:
                subprocess_logger.setLevel(file_level)

            if cfg.verbose:
                child_console_handler = logging.StreamHandler(stdout)
                child_console_handler.setFormatter(console_formatter)
                child_console_handler.setLevel(console_level)
                subprocess_logger.addHandler(child_console_handler)

            subprocess_logger.propagate = True

        if cfg.verbose and configured_logger_names:

            class ExcludeConfiguredLoggersFilter(logging.Filter):
                def filter(self, record):
                    return not any(record.name.startswith(name) for name in configured_logger_names)

            finn_console_handler.addFilter(ExcludeConfiguredLoggersFilter())

    def test_show_progress_true_shows_console(self):
        """show_progress=True displays progress messages on console."""
        output_dir = make_build_dir("test_show_progress_true_")

        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=True,
            verbose=False,
        )

        # Capture stdout
        from io import StringIO

        captured = StringIO()

        self._configure_loggers(cfg, stdout_override=captured)

        try:
            builder_logger = logging.getLogger("finn.builder")
            builder_logger.info("Running step 1/19")

            output = captured.getvalue()
            self.assertIn("Running step 1/19", output)
        finally:
            shutil.rmtree(output_dir)

    def test_show_progress_false_silent_console(self):
        """show_progress=False produces no progress output on console."""
        output_dir = make_build_dir("test_show_progress_false_")

        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=False,
            verbose=False,
        )

        self._configure_loggers(cfg)

        # Capture stdout
        from io import StringIO

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            builder_logger = logging.getLogger("finn.builder")
            builder_logger.info("Running step 1/19")

            output = captured.getvalue()
            self.assertEqual("", output)  # Should be completely silent
        finally:
            sys.stdout = old_stdout
            shutil.rmtree(output_dir)

    def test_verbose_true_shows_subprocess_console(self):
        """verbose=True displays subprocess output on console (ERROR level by default)."""
        output_dir = make_build_dir("test_verbose_true_")

        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=False,
            verbose=True,
        )

        # Capture stdout
        from io import StringIO

        captured = StringIO()

        self._configure_loggers(cfg, stdout_override=captured)

        try:
            hls_logger = logging.getLogger("finn.hls")
            # Default console level is ERROR, so WARNING won't show
            hls_logger.warning("This warning should not appear")
            hls_logger.error("HLS synthesis error")

            output = captured.getvalue()
            self.assertNotIn("This warning should not appear", output)
            self.assertIn("finn.hls", output)
            self.assertIn("ERROR", output)
        finally:
            shutil.rmtree(output_dir)

    def test_verbose_false_hides_subprocess_console(self):
        """verbose=False hides subprocess output from console."""
        output_dir = make_build_dir("test_verbose_false_")

        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=False,
            verbose=False,
        )

        self._configure_loggers(cfg)

        # Capture stdout
        from io import StringIO

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            hls_logger = logging.getLogger("finn.hls")
            hls_logger.warning("HLS synthesis warning")

            output = captured.getvalue()
            self.assertEqual("", output)  # Should be silent
        finally:
            sys.stdout = old_stdout
            shutil.rmtree(output_dir)

    def test_subprocess_log_levels_filters_file(self):
        """subprocess_log_levels controls file logging per tool."""
        output_dir = make_build_dir("test_subprocess_log_levels_")

        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=False,
            verbose=False,
            subprocess_log_levels={"hls": "ERROR"},  # Only errors
        )

        self._configure_loggers(cfg)

        try:
            hls_logger = logging.getLogger("finn.hls")
            hls_logger.info("This INFO should be filtered")
            hls_logger.warning("This WARNING should be filtered")
            hls_logger.error("This ERROR should appear")

            # Check log file
            log_file = cfg.output_dir + "/build_dataflow.log"
            with open(log_file, "r") as f:
                content = f.read()
                self.assertNotIn("This INFO should be filtered", content)
                self.assertNotIn("This WARNING should be filtered", content)
                self.assertIn("This ERROR should appear", content)
        finally:
            shutil.rmtree(output_dir)

    def test_subprocess_console_levels_filters_console(self):
        """subprocess_console_levels controls console output per tool when verbose=True."""
        output_dir = make_build_dir("test_subprocess_console_levels_")

        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=False,
            verbose=True,
            subprocess_console_levels={"vivado": "ERROR"},  # Only errors on console
            subprocess_log_levels={"vivado": "DEBUG"},  # Everything in file
        )

        # Capture stdout
        from io import StringIO

        captured = StringIO()

        self._configure_loggers(cfg, stdout_override=captured)

        try:
            vivado_logger = logging.getLogger("finn.vivado")
            vivado_logger.info("This INFO should not appear on console")
            vivado_logger.warning("This WARNING should not appear on console")
            vivado_logger.error("This ERROR should appear on console")

            console_output = captured.getvalue()
            self.assertNotIn("This INFO should not appear", console_output)
            self.assertNotIn("This WARNING should not appear", console_output)
            self.assertIn("This ERROR should appear", console_output)

            # But file should have everything
            log_file = cfg.output_dir + "/build_dataflow.log"
            with open(log_file, "r") as f:
                file_content = f.read()
                self.assertIn("This INFO should not appear on console", file_content)
                self.assertIn("This WARNING should not appear on console", file_content)
                self.assertIn("This ERROR should appear on console", file_content)
        finally:
            shutil.rmtree(output_dir)

    def test_json_serialization_roundtrip(self):
        """New logging config fields serialize/deserialize correctly."""
        cfg = DataflowBuildConfig(
            output_dir="/tmp/test",
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            show_progress=False,
            subprocess_log_levels={"vivado": "ERROR", "hls": "WARNING"},
            subprocess_console_levels={"vivado": "WARNING"},
        )

        # Serialize to JSON
        json_str = cfg.to_json()
        json_dict = json.loads(json_str)

        # Verify fields present in JSON
        self.assertEqual(json_dict["show_progress"], False)
        self.assertEqual(json_dict["subprocess_log_levels"], {"vivado": "ERROR", "hls": "WARNING"})
        self.assertEqual(json_dict["subprocess_console_levels"], {"vivado": "WARNING"})

        # Deserialize back
        cfg2 = DataflowBuildConfig.from_json(json_str)

        self.assertEqual(cfg2.show_progress, False)
        self.assertEqual(cfg2.subprocess_log_levels, {"vivado": "ERROR", "hls": "WARNING"})
        self.assertEqual(cfg2.subprocess_console_levels, {"vivado": "WARNING"})

    def test_backwards_compatible_verbose_only(self):
        """Old code using only verbose flag still works (new fields default correctly)."""
        output_dir = make_build_dir("test_backwards_compat_")

        # Old-style config: only verbose, no new fields
        cfg = DataflowBuildConfig(
            output_dir=output_dir,
            synth_clk_period_ns=5.0,
            generate_outputs=[],
            verbose=True,
        )

        # Verify defaults
        self.assertEqual(cfg.show_progress, True)  # Default should be True
        self.assertIsNone(cfg.subprocess_log_levels)
        self.assertIsNone(cfg.subprocess_console_levels)

        # Capture stdout
        from io import StringIO

        captured = StringIO()

        # Should configure without errors
        self._configure_loggers(cfg, stdout_override=captured)

        try:
            # Progress should show (default)
            builder_logger = logging.getLogger("finn.builder")
            builder_logger.info("Progress message")

            # Subprocess should show errors (verbose=True, default console level is ERROR)
            hls_logger = logging.getLogger("finn.hls")
            hls_logger.warning("HLS warning - should not appear")
            hls_logger.error("HLS error - should appear")

            output = captured.getvalue()
            self.assertIn("Progress message", output)
            self.assertNotIn("HLS warning", output)
            self.assertIn("HLS error", output)
        finally:
            shutil.rmtree(output_dir)
