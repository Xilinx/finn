#!/usr/bin/env python3
"""Inspect the TinyDeiT ONNX checkpoint and repeated block boundaries."""

from __future__ import annotations

import argparse
import onnx
from pathlib import Path

from transformer_examples.tinydeit.common import (
    add_common_args,
    resolve_common_paths,
    summarize_model,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--summary",
        default="summary.json",
        help="Summary JSON filename inside --output-dir.",
    )
    args = parser.parse_args()
    input_path, output_dir = resolve_common_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=False)
    summary = summarize_model(model)
    write_json(output_dir / args.summary, summary)

    print(f"Model: {input_path}")
    print(f"Nodes: {summary['nodes']}")
    print(f"Initializers: {summary['initializers']}")
    print(f"Inputs: {summary['inputs']}")
    print(f"Outputs: {summary['outputs']}")
    print("Top ops:")
    for op_type, count in list(summary["op_counts"].items())[:20]:
        print(f"  {op_type}: {count}")
    print(f"Detected transformer blocks: {len(summary['blocks'])}")
    for block in summary["blocks"]:
        print(
            "  block {block}: {start_node} ({start_index}) -> "
            "{end_node} ({end_index}), nodes={node_count}, softmax={softmax_nodes}".format(**block)
        )
    print(f"Wrote {Path(output_dir / args.summary)}")


if __name__ == "__main__":
    main()
