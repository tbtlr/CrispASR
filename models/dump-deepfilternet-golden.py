#!/usr/bin/env python3
"""Generate golden intermediate-tensor values for DeepFilterNet3.

Runs the official ONNX exports with a deterministic synthetic input
and captures every named output (encoder feature pyramid e0..e3,
combined-features c0, embedding emb, ERB-decoder mask m, DF-decoder
coefs + α), then writes them to a GGUF.

The C++ port loads the same GGUF via `core_gguf::load_weights` (the
exact path used for the model weights themselves) so layer-by-layer
asserts in `predict_dfn3` are a tensor lookup + a `max(|a-b|)` away.

Usage:

    python models/dump-deepfilternet-golden.py \\
        --onnx-dir /path/to/DeepFilterNet3 \\
        --output  models/deepfilternet3-golden.gguf \\
        --frames  8 --seed 42

`--frames` controls the sequence length T baked into the golden
tensors. Keep it small (≤ 16) — every layer's full tensor lands in
the GGUF, so the file grows with T.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnx onnxruntime")
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# Tensors we ask onnxruntime to surface from each graph. These names
# come straight from the ONNX outputs (see `inspect-deepfilternet-onnx.py`).
ENC_OUTPUTS     = ["e0", "e1", "e2", "e3", "emb", "c0"]            # lsnr skipped — auxiliary head
ERB_DEC_OUTPUTS = ["m"]
DF_DEC_OUTPUTS  = ["coefs", "235"]                                  # 235 == α (sigmoid output)


def make_inputs(rng: np.random.Generator, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic encoder inputs.

    `feat_erb`  : (1, 1, T, 32) — standardised log-mag features per ERB band.
                  We sample from N(0, 1) since that's what the running-mean/
                  variance normaliser in src/dfn.cpp produces at steady state.
    `feat_spec` : (1, 2, T, 96) — real/imag parts of the unit-norm spectrum
                  on the first df_bins=96 STFT bins. Magnitudes should be
                  bounded by 1; we sample from N(0, 0.3) which gives a
                  realistic envelope without clipping the encoder's input
                  range.
    """
    feat_erb  = rng.standard_normal((1, 1, T, 32)).astype(np.float32)
    feat_spec = (rng.standard_normal((1, 2, T, 96)) * 0.3).astype(np.float32)
    return feat_erb, feat_spec


def run_session(path: Path, inputs: dict, want_outputs: list[str],
                extra_outputs: dict[str, str] | None = None) -> dict[str, np.ndarray]:
    """Run an ONNX model and return the requested outputs.

    `extra_outputs` maps short logical names to the internal tensor
    name they should be surfaced as. The model is patched on the fly:
    each requested intermediate gets added to graph.output so
    onnxruntime can return it. This is how we expose pre-GRU
    activations for layer-by-layer GGML validation."""
    model = onnx.load(str(path))
    extras = list((extra_outputs or {}).items())
    if extras:
        # Build a fresh model whose outputs include the intermediates.
        existing = {o.name for o in model.graph.output}
        for short_name, internal in extras:
            if internal in existing:
                continue
            vi = onnx.helper.make_tensor_value_info(internal, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(vi)
        # Serialise to bytes for InferenceSession.
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    else:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    out_names_from_session = [o.name for o in sess.get_outputs()]
    out_names = [n for n in out_names_from_session if n in want_outputs or n in (e[1] for e in extras)]
    missing = set(want_outputs) - set(out_names)
    if missing:
        print(f"  warning: {path.name} missing outputs {missing}", file=sys.stderr)
    arrays = sess.run(out_names, inputs)
    res = dict(zip(out_names, arrays))
    # Rename intermediate keys to their short logical names so the
    # caller (and the GGUF) sees stable, human-readable names instead
    # of `/df_fc_emb/0/Einsum_output_0` etc.
    for short_name, internal in extras:
        if internal in res:
            res[short_name] = res.pop(internal)
    return res


def dump(onnx_dir: Path, output: Path, frames: int, seed: int, verbose: int) -> None:
    export_dir = onnx_dir / "export" if (onnx_dir / "export").is_dir() else onnx_dir
    rng = np.random.default_rng(seed)

    feat_erb, feat_spec = make_inputs(rng, frames)

    # Encoder. Surface a handful of intermediates so the GGML port can
    # be validated layer-by-layer between e3/c0 and the GRU's output.
    enc_extras = {
        # Output of the Add node at #90 — e3 reshape + df_fc_emb-Relu →
        # the 512-dim sequence fed into emb_gru.
        "combined":    "/combine/Add_output_0",
        # Output of linear_in's trailing Relu (#111) — direct input to
        # the ONNX GRU step.
        "pre_gru":     "/emb_gru/linear_in/1/Relu_output_0",
        # df_fc_emb's Relu output (#80) — the path before combine.
        "df_fc_emb":   "/df_fc_emb/1/Relu_output_0",
    }
    enc_out = run_session(export_dir / "enc.onnx",
                          {"feat_erb": feat_erb, "feat_spec": feat_spec},
                          ENC_OUTPUTS,
                          extra_outputs=enc_extras)

    # ERB decoder consumes emb + e0..e3.
    erb_in = {
        "emb": enc_out["emb"],
        "e3":  enc_out["e3"],
        "e2":  enc_out["e2"],
        "e1":  enc_out["e1"],
        "e0":  enc_out["e0"],
    }
    # Same trick as on enc.onnx — expose intermediates so the GGML
    # port of the decoder backbone can be validated layer-by-layer.
    erb_extras = {
        "emb_gru_out": "/emb_gru/linear_out/Relu_output_0",   # post SqueezedGRU
        "convt3_out":  "/convt3/Relu_output_0",                # after first dw+pw stack
        "convt2_out":  "/convt2/3/Relu_output_0",              # after ConvTranspose#1 (F: 8 → 16)
        "convt1_out":  "/convt1/Relu_output_0",                # after ConvTranspose#2 (F: 16 → 32)
    }
    erb_out = run_session(export_dir / "erb_dec.onnx", erb_in, ERB_DEC_OUTPUTS, extra_outputs=erb_extras)

    # DF decoder consumes emb + c0. Also surface intermediates so the
    # GGML port can be validated layer-by-layer alongside the final
    # coefs / α outputs.
    df_in = {"emb": enc_out["emb"], "c0": enc_out["c0"]}
    df_extras = {
        # Post GRU + df_skip Add — input to both the α head and df_out.
        "skip_add":   "/Add_output_0",
        # Post df_out Tanh — the part of `coefs` that comes from emb.
        # (Adds together with df_convp transposed for the final coefs.)
        "df_out_tanh": "/df_out/df_out.1/Tanh_output_0",
        # Post df_convp Relu (after the 5×1 conv stack), before the
        # final Transpose into (1, T, 96, 10).
        "df_convp_out": "/df_convp/df_convp.4/Relu_output_0",
    }
    df_out = run_session(export_dir / "df_dec.onnx", df_in, DF_DEC_OUTPUTS, extra_outputs=df_extras)

    # Write everything to a single GGUF tagged `deepfilternet3-golden`.
    # Tensor names mirror the graph output names so the C++ side reads
    # `dfn.golden.<name>` and compares per layer.
    output.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(output), arch="deepfilternet3-golden")
    w.add_uint32("dfn.golden.frames", frames)
    w.add_uint32("dfn.golden.seed", seed)

    def add(prefix: str, name: str, arr: np.ndarray) -> None:
        gguf_name = f"dfn.golden.{prefix}.{name}"
        # GGUF tensors are float32 here — these are reference values; we
        # want every available bit when comparing GGML float32 output.
        a = np.ascontiguousarray(arr.astype(np.float32))
        w.add_tensor(gguf_name, a)
        if verbose:
            print(f"  {gguf_name:50s} {str(a.shape):20s} "
                  f"min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f}")

    add("input", "feat_erb",  feat_erb)
    add("input", "feat_spec", feat_spec)
    for k, v in enc_out.items():
        # Output `235` is alpha — rename to a meaningful key.
        add("enc", k, v)
    for k, v in erb_out.items():
        add("erb_dec", k, v)
    for k, v in df_out.items():
        gguf_key = "alpha" if k == "235" else k
        add("df_dec", gguf_key, v)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Done: {output} ({output.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx-dir", required=True, type=Path)
    p.add_argument("--output",   required=True, type=Path)
    p.add_argument("--frames",   type=int, default=8)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    dump(args.onnx_dir, args.output, args.frames, args.seed, args.verbose)


if __name__ == "__main__":
    main()
