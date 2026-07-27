#!/usr/bin/env python3
"""Run software verification for prepared TinyDeiT FINN models."""

from __future__ import annotations

import argparse
import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from transformer_examples.tinydeit.common import DEFAULT_BUILD_DIR, repo_path


def prepare_cppsim(model: ModelWrapper, num_workers: int | None) -> ModelWrapper:
    model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
    model = model.transform(PrepareCppSim(num_workers), apply_to_subgraphs=True)
    model = model.transform(CompileCppSim(num_workers), apply_to_subgraphs=True)
    model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str((DEFAULT_BUILD_DIR / "flow" / "tinydeit_mlo.onnx").relative_to(repo_path("."))),
    )
    parser.add_argument("--reference", default=None)
    parser.add_argument("--input-npy", default=None)
    parser.add_argument(
        "--output-dir", default=str((DEFAULT_BUILD_DIR / "verify").relative_to(repo_path(".")))
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-cppsim-prepare", dest="cppsim_prepare", action="store_false")
    parser.add_argument("--cppsim-workers", type=int, default=None)
    parser.add_argument("--reference-cppsim-prepare", action="store_true")
    parser.add_argument("--reference-cppsim-workers", type=int, default=None)
    parser.set_defaults(cppsim_prepare=True)
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ModelWrapper(str(repo_path(args.model)))
    if args.cppsim_prepare:
        model = prepare_cppsim(model, args.cppsim_workers)
    input_name = model.get_first_global_in()
    output_name = model.get_first_global_out()
    if args.input_npy:
        input_tensor = np.load(repo_path(args.input_npy))
    else:
        np.random.seed(args.seed)
        input_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], model.get_tensor_shape(input_name))
    np.save(output_dir / "input.npy", input_tensor)

    produced_ctx = execute_onnx(model, {input_name: input_tensor}, return_full_exec_context=True)
    produced = produced_ctx[output_name]
    np.save(output_dir / "produced_output.npy", produced)
    np.savez(output_dir / "produced_context.npz", **produced_ctx)

    if args.reference:
        ref_model = ModelWrapper(str(repo_path(args.reference)))
        if args.reference_cppsim_prepare:
            ref_model = prepare_cppsim(ref_model, args.reference_cppsim_workers)
        ref_ctx = execute_onnx(
            ref_model,
            {ref_model.get_first_global_in(): input_tensor},
            return_full_exec_context=True,
        )
        expected = ref_ctx[ref_model.get_first_global_out()]
        np.save(output_dir / "expected_output.npy", expected)
        np.savez(output_dir / "expected_context.npz", **ref_ctx)
        max_abs = float(np.max(np.abs(produced - expected)))
        close = bool(np.allclose(produced, expected, atol=args.atol))
        print(f"max_abs_diff={max_abs}")
        print(f"allclose_atol_{args.atol}={close}")
        if not close:
            raise SystemExit(1)
    else:
        print(f"Executed {args.model}; output shape={produced.shape}")


if __name__ == "__main__":
    main()
