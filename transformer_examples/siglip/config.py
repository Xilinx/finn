# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration loading for the SigLIP transformer example."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = EXAMPLE_DIR / "configs" / "siglip2_base_patch16_224_w6a7_qv_lsq.json"


@dataclass(frozen=True)
class SiglipProfile:
    """Validated build profile loaded from JSON."""

    path: Path
    name: str
    model: dict[str, Any]
    quantization: dict[str, Any]
    build: dict[str, Any]
    reference_metrics: dict[str, Any]

    def resolve_file(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.path.parent / path


def _require_int(mapping: dict[str, Any], key: str, minimum: int = 1) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def load_profile(path: str | Path = DEFAULT_PROFILE) -> SiglipProfile:
    """Load and validate a SigLIP build profile."""

    profile_path = Path(path).resolve()
    with profile_path.open(encoding="utf-8") as profile_file:
        raw = json.load(profile_file)

    for key in ("name", "model", "quantization", "build", "reference_metrics"):
        if key not in raw:
            raise ValueError(f"Profile is missing required field: {key}")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("name must be a non-empty string")
    for key in ("model", "quantization", "build", "reference_metrics"):
        if not isinstance(raw[key], dict):
            raise ValueError(f"{key} must be a JSON object")

    model = raw["model"]
    quantization = raw["quantization"]
    build = raw["build"]
    _require_int(model, "vision_depth")
    _require_int(model, "image_size")
    _require_int(model, "patch_size")
    if not isinstance(model.get("output_name"), str) or not model["output_name"]:
        raise ValueError("output_name must be a non-empty string")
    _require_int(quantization, "weight_bits")
    _require_int(quantization, "activation_bits")
    _require_int(quantization, "edge_bits")
    if build.get("board") != "VCK190":
        raise ValueError("The initial SigLIP profile currently supports board=VCK190")
    if not isinstance(build.get("clock_ns"), (int, float)) or build["clock_ns"] <= 0:
        raise ValueError("clock_ns must be positive")
    if not isinstance(build.get("target_fps"), (int, float)) or build["target_fps"] <= 0:
        raise ValueError("target_fps must be positive")
    _require_int(build, "fifo_depth_cap", minimum=2)

    return SiglipProfile(
        path=profile_path,
        name=raw["name"],
        model=model,
        quantization=quantization,
        build=build,
        reference_metrics=raw["reference_metrics"],
    )
