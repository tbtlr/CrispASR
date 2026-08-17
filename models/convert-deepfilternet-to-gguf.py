#!/usr/bin/env python3
"""Convert DeepFilterNet3 ONNX exports to a single GGUF.

Source: https://github.com/Rikorose/DeepFilterNet

DFN3 ships three separate ONNX graphs — one per network module — plus
a `config.ini` with the architecture hyperparameters. After `pip
install deepfilternet` the cached layout looks like:

    DeepFilterNet3/
      config.ini
      checkpoints/
        model_120.ckpt.best        # PyTorch (not needed here)
      export/
        enc.onnx                   # shared encoder
        erb_dec.onnx               # Stage-1 ERB-gain decoder
        df_dec.onnx                # Stage-2 DF complex-coef decoder

The same layout is what `df.io.load_model(..., epoch="best")` writes
when you run it once on any host that can reach the GitHub release.

This script walks every initializer in each ONNX graph, namespaces it
under `dfn.<module>.` and writes one combined GGUF that
`src/dfn.cpp` can mmap-load via `core_gguf::load_weights`. The
hyperparameters from `config.ini` get baked into GGUF metadata keys so
the C runtime needs no separate config file.

Usage:

    python models/convert-deepfilternet-to-gguf.py \\
        --input  /path/to/DeepFilterNet3 \\
        --output models/deepfilternet3.gguf

Pass `-v` to print every tensor name + shape — useful for verifying
the C-side probe names in `gguf_has_dfn3_weights` against the actual
export.
"""

from __future__ import annotations

import argparse
import configparser
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("pip install onnx")
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


# ── DFN3 defaults (used when config.ini is missing a key) ────────────
DEFAULTS = dict(
    sample_rate=48_000,
    n_fft=960,
    hop=480,
    n_erb=32,
    df_bins=96,
    df_order=5,
    lookahead_frames=2,
)

# The three ONNX modules + the GGUF namespace each gets. Anything that
# isn't a `.onnx` file in `export/` is ignored.
MODULES = [
    ("enc",     "enc.onnx"),
    ("erb_dec", "erb_dec.onnx"),
    ("df_dec",  "df_dec.onnx"),
]


def vorbis_window(n: int) -> np.ndarray:
    """Vorbis window — matches libdf's DFState::new and src/dfn.cpp's
    `build_default_window`. w[i] = sin(π/2 · sin²(π/2 · (i + 0.5) / (N/2))).
    Earlier versions of this converter wrote a sqrt-Hann window here
    which silently caused the C runtime to feed the model a different
    spectral envelope from what DFN3 was trained on."""
    half = n // 2
    s = np.sin(0.5 * math.pi * (np.arange(n) + 0.5) / half)
    return np.sin(0.5 * math.pi * s * s).astype(np.float32)


def erb_filterbank(sample_rate: int, n_fft: int, n_bands: int, min_nb_freqs: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """ERB filterbank — exact port of `libdf::erb_fb` (Rust).

    Critical for matching the model's training-time band layout: the
    naive linspace-and-clamp version (what this used to be) gave
    widths [1,1,1,1,1,1,2,2,2,3,3,4,...] for the first 12 bands at 48 kHz,
    but libdf yields [2,2,2,2,2,2,2,2,2,2,2,2,...] because it enforces
    `min_nb_erb_freqs = 2` per band with a `freq_over` "borrow" tracker
    that accounts for bins lent forward when a band would have come up
    short. Without this, `feat_erb` for the lowest 6 bands sees
    completely different bin groupings → model outputs garbage gains.

    Returns (widths, indices) as int32. `widths` sums to n_fft/2 + 1.
    """
    def freq2erb(f: float) -> float:
        return 9.265 * math.log1p(f / (24.7 * 9.265))

    def erb2freq(e: float) -> float:
        return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)

    nyq        = sample_rate // 2
    freq_width = sample_rate / n_fft
    erb_low    = freq2erb(0.0)
    erb_high   = freq2erb(float(nyq))
    step       = (erb_high - erb_low) / n_bands

    widths    = [0] * n_bands
    prev_freq = 0
    freq_over = 0
    for i in range(1, n_bands + 1):
        f         = erb2freq(erb_low + i * step)
        fb        = int(round(f / freq_width))
        nb_freqs  = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs  = min_nb_freqs
        else:
            freq_over = 0
        widths[i - 1] = nb_freqs
        prev_freq     = fb
    widths[-1] += 1  # account for the WINDOW_SIZE/2+1 bin count
    too_large = sum(widths) - (n_fft // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    assert sum(widths) == n_fft // 2 + 1, f"widths sum {sum(widths)} != {n_fft // 2 + 1}"

    widths_arr  = np.array(widths, dtype=np.int32)
    indices_arr = np.concatenate(([0], np.cumsum(widths_arr[:-1]))).astype(np.int32)
    return widths_arr, indices_arr


def read_config(cfg_path: Path) -> dict:
    """Read `config.ini` if present; merge over DEFAULTS.

    DFN's ini file is sectioned (`[df]`, `[deepfilternet]`, ...); we
    only need a handful of integer keys. Anything missing falls back
    to the defaults baked into src/dfn.cpp so a partial config still
    produces a usable GGUF.
    """
    cfg = dict(DEFAULTS)
    if not cfg_path.exists():
        return cfg
    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    # Walk every section, pull any key that matches a known name. The
    # actual section name varies between DFN versions; matching by key
    # name is robust.
    aliases = {
        "sr": "sample_rate",
        "samplerate": "sample_rate",
        "fft_size": "n_fft",
        "hop_size": "hop",
        "nb_erb": "n_erb",
        "nb_df": "df_bins",
        "df_lookahead": "lookahead_frames",
    }
    for section in parser.sections():
        for key, val in parser.items(section):
            k = aliases.get(key, key)
            if k in cfg:
                try:
                    cfg[k] = int(val)
                except ValueError:
                    pass
    return cfg


def sanitise_name(name: str) -> str:
    """ONNX initializer names sometimes carry `/` separators (from
    PyTorch's exported scopes) and `:0` output suffixes. GGUF tensor
    names are conventionally dot-delimited and have no suffixes."""
    n = name
    if n.endswith(":0"):
        n = n[:-2]
    n = n.replace("/", ".")
    # Collapse consecutive dots and strip leading/trailing dots.
    n = re.sub(r"\.{2,}", ".", n).strip(".")
    return n


def walk_onnx(path: Path, namespace: str, writer: gguf.GGUFWriter, verbose: int) -> int:
    """Add every initializer in `path` to `writer`, prefixed with
    `dfn.<namespace>.`. Returns the count written.

    GRU operators are special-cased: in enc.onnx the W / R / B weights
    arrive as `Constant` op outputs, not as graph initializers (the
    other DFN ONNX files store them as initializers). We promote those
    constants into the GGUF with stable derived names like
    `dfn.enc.emb_gru.gru.{W,R,B}` so the C runtime sees one consistent
    layout regardless of the source export style."""
    if not path.exists():
        print(f"  warning: {path} not found — skipping", file=sys.stderr)
        return 0
    model = onnx.load(str(path))

    # Map: Constant op output_name → numpy array. Used to follow GRU
    # inputs through the Constant indirection.
    const_outs: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for a in node.attribute:
            if a.name in ("value", "value_float", "value_floats") and a.t.dims:
                const_outs[node.output[0]] = numpy_helper.to_array(a.t)

    # Pre-pass: collect the set of initializer names that are referenced
    # by a GRU as W/R/B. These get clean prefix-based aliases below and
    # we skip the original (ugly) name to keep the GGUF compact.
    gru_init_inputs: dict[str, tuple[str, str]] = {}
    for node in model.graph.node:
        if node.op_type != "GRU":
            continue
        mod = node.name.lstrip("/").replace("/", ".").lower() or "gru"
        for idx, key in zip((1, 2, 3), ("W", "R", "B")):
            if idx >= len(node.input):
                continue
            inp = node.input[idx]
            if not inp:
                continue
            gru_init_inputs[inp] = (mod, key)

    n = 0
    init_names = {i.name for i in model.graph.initializer}

    # 1) Standard initializers — skip those that will be re-emitted
    #    under a clean GRU-module name below.
    for init in model.graph.initializer:
        if init.name in gru_init_inputs:
            continue
        arr = numpy_helper.to_array(init)
        out = arr.astype(np.int32, copy=False) if arr.dtype.kind in ("i", "u") else arr.astype(np.float16, copy=False)
        gguf_name = f"dfn.{namespace}.{sanitise_name(init.name)}"
        writer.add_tensor(gguf_name, out)
        n += 1
        if verbose:
            print(f"  {gguf_name:65s} {str(out.shape):20s} {out.dtype}")

    # 2) GRU W/R/B — promoted to stable per-module names regardless of
    #    whether they were Constant outputs (enc.onnx) or initializers
    #    (df_dec.onnx).
    for node in model.graph.node:
        if node.op_type != "GRU":
            continue
        mod = node.name.lstrip("/").replace("/", ".").lower() or "gru"
        for idx, key in zip((1, 2, 3), ("W", "R", "B")):
            if idx >= len(node.input):
                continue
            inp = node.input[idx]
            if not inp:
                continue
            if inp in const_outs:
                arr = const_outs[inp]
            elif inp in init_names:
                arr = numpy_helper.to_array(next(i for i in model.graph.initializer if i.name == inp))
            else:
                continue
            out = arr.astype(np.float16, copy=False)
            gguf_name = f"dfn.{namespace}.{mod}.{key}"
            writer.add_tensor(gguf_name, out)
            n += 1
            if verbose:
                print(f"  {gguf_name:65s} {str(out.shape):20s} {out.dtype}")
    return n


def convert(input_dir: Path, output: Path, verbose: int) -> None:
    cfg = read_config(input_dir / "config.ini")
    output.parent.mkdir(parents=True, exist_ok=True)

    w = gguf.GGUFWriter(str(output), arch="deepfilternet3")

    # Architecture metadata (consumed by `src/dfn.cpp::dfn_model_load`).
    for key in ("sample_rate", "n_fft", "hop", "n_erb", "df_bins", "df_order", "lookahead_frames"):
        w.add_uint32(f"dfn.{key}", int(cfg[key]))

    # Precomputed analysis tables. Baked into the GGUF so the C side
    # has zero Python-side constants to mirror.
    widths, indices = erb_filterbank(cfg["sample_rate"], cfg["n_fft"], cfg["n_erb"])
    w.add_tensor("dfn.erb_widths", widths)
    w.add_tensor("dfn.erb_indices", indices)
    w.add_tensor("dfn.window", vorbis_window(cfg["n_fft"]))

    # Trained weights — one ONNX per module.
    export_dir = input_dir / "export" if (input_dir / "export").is_dir() else input_dir
    total = 0
    for ns, fname in MODULES:
        path = export_dir / fname
        n = walk_onnx(path, ns, w, verbose)
        total += n
        print(f"  {ns:8s}: {n} tensors from {path.name}")

    print(f"\n  total trained tensors: {total}")
    if total == 0:
        print("  warning: no trained tensors found — the resulting GGUF will load via",
              "          the identity fallback in src/dfn.cpp (passthrough).", file=sys.stderr)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Done: {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="DeepFilterNet3 directory (contains config.ini + export/*.onnx). "
             "If you only have a plain folder of .onnx files, point at it directly.",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Print every emitted tensor name and shape.")
    args = p.parse_args()
    convert(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
