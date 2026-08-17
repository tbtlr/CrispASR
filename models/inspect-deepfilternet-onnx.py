#!/usr/bin/env python3
"""Dump the DeepFilterNet3 ONNX graphs in node-execution order.

The output is the reference used when writing the GGML compute graph
in `src/dfn.cpp::predict_dfn3`. For every node we print:

  * op type and short name
  * input tensor names (with `[init]` for graph initializers)
  * output tensor names
  * attribute summary for ops whose behaviour depends on it
    (Conv pad/stride/kernel, GRU hidden_size, Transpose perm, ...)

Run after `convert-deepfilternet-to-gguf.py` has confirmed the export
directory layout. With `-v` the per-node initializer shapes/dtypes are
shown inline — useful for matching against `m->weights.tensors` lookup
names.

Usage:

    python models/inspect-deepfilternet-onnx.py \\
        --input /path/to/DeepFilterNet3 [-v]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    sys.exit("pip install onnx")


def attr_summary(node) -> str:
    """One-line summary of the attributes that matter for layer
    interpretation (kernel/stride/pad for Conv, hidden_size for GRU,
    perm for Transpose, axes for Slice/Unsqueeze)."""
    keep = []
    for a in node.attribute:
        name = a.name
        if name in ("kernel_shape", "strides", "pads", "dilations", "group",
                    "hidden_size", "perm", "axes", "axis", "epsilon",
                    "direction"):
            if a.type == onnx.AttributeProto.INTS:
                v = list(a.ints)
            elif a.type == onnx.AttributeProto.INT:
                v = a.i
            elif a.type == onnx.AttributeProto.FLOAT:
                v = a.f
            elif a.type == onnx.AttributeProto.STRING:
                v = a.s.decode() if isinstance(a.s, bytes) else a.s
            else:
                v = "?"
            keep.append(f"{name}={v}")
    return " ".join(keep)


def dump_graph(path: Path, verbose: int) -> None:
    model = onnx.load(str(path))
    g = model.graph

    inits = {i.name: i for i in g.initializer}

    print(f"\n=== {path.name} ===")
    print(f"inputs:  {[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input]}")
    print(f"outputs: {[(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.output]}")
    print(f"nodes:   {len(g.node)}\n")

    for idx, n in enumerate(g.node):
        # Mark inputs that are initializers (trained weights) vs runtime tensors.
        ins = []
        for inp in n.input:
            if inp in inits:
                init = inits[inp]
                shape = tuple(init.dims)
                if verbose:
                    ins.append(f"{inp}[init {shape}]")
                else:
                    ins.append(f"{inp}[init]")
            else:
                ins.append(inp)
        outs = list(n.output)
        attrs = attr_summary(n)
        # `Constant` ops dominate the node list and aren't interesting
        # for the GGML translation — they're absorbed into op attrs.
        if n.op_type == "Constant" and not verbose:
            continue
        print(f"  #{idx:3d} {n.op_type:14s} {n.name or '<anon>':22s}"
              f" in=[{', '.join(ins)}]"
              f" out=[{', '.join(outs)}]"
              + (f"  {attrs}" if attrs else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="DeepFilterNet3 directory (with `export/*.onnx`) or a folder of .onnx files.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Also print initializer shapes inline.")
    args = p.parse_args()

    export_dir = args.input / "export" if (args.input / "export").is_dir() else args.input
    for name in ("enc", "erb_dec", "df_dec"):
        path = export_dir / f"{name}.onnx"
        if path.exists():
            dump_graph(path, args.verbose)


if __name__ == "__main__":
    main()
