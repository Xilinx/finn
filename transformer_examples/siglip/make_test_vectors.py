# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Create deterministic reference I/O for a static SigLIP QONNX graph."""

from __future__ import annotations

import argparse
import json
import numpy as np
import warnings
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

DEFAULT_OUTPUT_NAME = "image_embeds"


def create_test_vectors(
    model_path: Path,
    output_dir: Path,
    seed: int,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict:
    model = ModelWrapper(str(model_path))
    if len(model.graph.input) != 1 or len(model.graph.output) < 1:
        raise ValueError("Reference-vector generation requires one input and at least one output")
    input_info = model.graph.input[0]
    input_shape = model.get_tensor_shape(input_info.name)
    if input_shape is None or any(
        not isinstance(value, int) or value <= 0 for value in input_shape
    ):
        raise ValueError(f"Input must have a fully static shape, found {input_shape}")

    rng = np.random.default_rng(seed)
    input_tensor = rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)
    available_outputs = [value_info.name for value_info in model.graph.output]
    if output_name not in available_outputs:
        raise ValueError(f"Unknown output {output_name!r}; choose one of {available_outputs}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Output shapes disagree after node.*")
        output_tensor = execute_onnx(model, {input_info.name: input_tensor})[output_name]

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input.npy"
    expected_path = output_dir / "expected_output.npy"
    np.save(input_path, input_tensor)
    np.save(expected_path, output_tensor)
    metadata = {
        "model": str(model_path.resolve()),
        "seed": seed,
        "input_name": input_info.name,
        "input_shape": list(input_tensor.shape),
        "output_name": output_name,
        "output_shape": list(output_tensor.shape),
        "input_npy": str(input_path),
        "expected_output_npy": str(expected_path),
    }
    with (output_dir / "test_vectors.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Model output to verify (default: {DEFAULT_OUTPUT_NAME})",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            create_test_vectors(args.model, args.output_dir, args.seed, args.output_name), indent=2
        )
    )


if __name__ == "__main__":
    main()
