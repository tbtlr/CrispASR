// dfn.cpp — DeepFilterNet3 streaming speech enhancement (GGML port).
//
// Stage 4 (this file, current): GGML weight loader is wired in. The
// model owns a `ggml_backend_t` and a `core_gguf::WeightLoad` (same
// pattern every other model in src/ uses). On load we detect whether
// the GGUF carries trained neural weights (via probe tensor names);
// if not, the predict step transparently falls back to the identity
// pass-through, so a dev / metadata-only GGUF still loads and runs.
//
// Stage 3 (still in place): the spectrum-processing layer between
// STFT and iSTFT — all weight-free, deterministic. Specifically:
//
//   * ERB filterbank projection (481 STFT bins → 32 ERB bands).
//   * Log-magnitude features with per-band exponential moving-average
//     mean/std normalisation (the input the DFN encoder consumes).
//   * Application of a per-band gain back to the full spectrum
//     ("Stage-1" of DFN3's masking).
//   * Deep-filter complex linear-filter operation over a sliding
//     history of recent spectra ("Stage-2"), with the canonical
//     2-frame lookahead.
//
// All of these are deterministic transforms — no learned weights —
// so they can be exercised today against synthetic inputs. The
// remaining piece is the neural network itself (encoder convs, GRUs,
// ERB-gain decoder, DF-coefficient decoder): given a per-frame
// magnitude/log-magnitude/complex feature stack, it would produce
// the (n_erb)-dim real gains and the (df_bins × df_order)-shape
// complex DF coefficients that drive these helpers.
//
// Until that neural net lands, the per-frame "predict" step returns
// identity values (gain=1, df coefs = δ at the centre tap), so the
// round-trip is still numerically the input — useful as a
// pass-through regression while the model converter and GGML graph
// are wired up incrementally.
//
// The GGUF loader reads architecture metadata + ERB tables + window
// directly from the file (with built-in fallbacks that match the
// converter's formulas, so a partial / dev GGUF still loads).

#include "dfn.h"
#include "dfn_fft.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include "core/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Native DFN3 parameters — defaults used when the GGUF doesn't carry
// the matching metadata key. The converter writes all of these.
static constexpr int kDfnSampleRate    = 48000;
static constexpr int kDfnFrameSize     = 480;  // 10 ms hop
static constexpr int kDfnFftSize       = 960;  // 50% overlap, sqrt-Hann
static constexpr int kDfnLookaheadFrms = 2;    // ~20 ms
static constexpr int kDfnNErb          = 32;
static constexpr int kDfnDfBins        = 96;   // ~5 kHz
static constexpr int kDfnDfOrder       = 5;

// Normalisation hyperparameters — verified against libdf
// (deep_filter/src/lib.rs band_mean_norm_erb / band_unit_norm).
//   α     = exp(-hop/sr / τ) with hop=480, sr=48000, τ=norm_tau=1 s
//   bands : ERB log-power mean state seeded from linspace(-60, -90) dB
//   bins  : per-freq EMA of |spec|, seeded from linspace(0.001, 0.0001)
//   feat_erb = (10·log10(power+eps) - state) / 40
//   feat_spec = spec / sqrt(state)
static constexpr float kDfnEmaAlpha     = 0.99005f;  // exp(-0.01 / 1) ≈ 0.99005, matches get_norm_alpha
static constexpr float kDfnLogEpsilon   = 1e-10f;
static constexpr float kDfnErbNormDiv   = 40.0f;     // dB-range normaliser
static constexpr float kDfnMeanNormHi   = -60.0f;    // MEAN_NORM_INIT[0]
static constexpr float kDfnMeanNormLo   = -90.0f;    // MEAN_NORM_INIT[1]
static constexpr float kDfnUnitNormHi   = 0.001f;    // UNIT_NORM_INIT[0]
static constexpr float kDfnUnitNormLo   = 0.0001f;   // UNIT_NORM_INIT[1]

struct dfn_model {
    dfn_params params{};
    int sample_rate    = kDfnSampleRate;
    int frame_size     = kDfnFrameSize;
    int fft_size       = kDfnFftSize;
    int lookahead_frms = kDfnLookaheadFrms;
    int n_erb          = kDfnNErb;
    int df_bins        = kDfnDfBins;
    int df_order       = kDfnDfOrder;

    std::vector<float> window;       // length fft_size — sqrt-Hann
    std::vector<int>   erb_widths;   // length n_erb — # of STFT bins per ERB band
    std::vector<int>   erb_indices;  // length n_erb — first STFT bin of each ERB band

    // FFT plan reused across streams (state-free; the per-call scratch
    // lives on the plan but never mutates from a calling-thread view).
    dfn_fft_plan* fft = nullptr;

    // GGML weight storage. `weights.tensors` is the canonical
    // `name → ggml_tensor*` map (built by `core_gguf::load_weights`);
    // `backend` owns the buffer those tensors live in. When the GGUF
    // doesn't carry trained neural-net weights yet (e.g. a converter
    // that only baked the ERB tables), `tensors` is still allocated
    // but empty — `predict_dfn3()` then falls back to identity.
    ggml_backend_t        backend = nullptr;
    ggml_backend_sched_t  sched   = nullptr; // re-used across debug + per-frame graphs
    core_gguf::WeightLoad weights{};
    bool                  has_neural_weights = false;
};

struct dfn_stream {
    const dfn_model* model = nullptr;

    std::vector<float> in_buf;     // pending input samples
    std::vector<float> frame;      // windowed analysis / iSTFT result (length fft_size)
    std::vector<float> spec_re;    // current frame complex spectrum (length n_bins)
    std::vector<float> spec_im;
    std::vector<float> ola;        // overlap-add (length fft_size)
    std::vector<float> out_queue;  // emitted samples waiting for the lookahead-aware drain

    // Per-band log-power EMA state. `libdf::band_mean_norm_erb` only
    // tracks the running mean (no variance); we initialise on first
    // frame from linspace(-60, -90) dB to match the upstream init.
    std::vector<float> log_mag_mean;  // length n_erb
    bool               norm_seeded = false;

    // Rolling complex-spectrum history for the DF complex filter. Holds
    // the last (df_order + lookahead_frms) frames as flat (re, im)
    // pairs. The newest frame is at the end; `df_apply` indexes
    // backwards into it.
    //
    // Layout: stride = n_bins; oldest frame at offset 0, newest at
    // offset (history_frames - 1) * n_bins.
    std::vector<float> df_hist_re;
    std::vector<float> df_hist_im;
    int                df_hist_frames = 0; // df_order + lookahead_frms

    // ─── Streaming GRU hidden states ────────────────────────────────
    // Five 256-dim buffers, one per GRU instance in the network. Zero
    // at stream start; updated every frame the GGML graph runs.
    //   [0] encoder emb_gru
    //   [1] erb_dec emb_gru #1
    //   [2] erb_dec emb_gru #2
    //   [3] df_dec df_gru #1
    //   [4] df_dec df_gru #2
    static constexpr int kNumGruStates = 5;
    static constexpr int kGruHidden    = 256;
    std::vector<float> gru_hidden;     // size = kNumGruStates * kGruHidden

    // Per-frequency-bin EMA of |spec| (libdf `band_unit_norm`). The
    // model's `feat_spec` input is the complex spectrum divided by
    // sqrt(this state) at each bin. Length df_bins=96; seeded from
    // linspace(0.001, 0.0001) on the first frame.
    std::vector<float> unit_norm_state;   // length df_bins
    bool               unit_norm_seeded = false;

    // ─── Lookahead alignment counter ────────────────────────────────
    // Upstream's `pad_feat = ConstantPad2d((0, 0, -2, 2))` shifts the
    // encoder's feature stream forward by 2 frames at training time —
    // so mask[t] is computed from feat[t..t+2] (current + 2 future).
    // In streaming we buffer the first 2 input frames without running
    // the model so that, from call 2 onwards, encoder input
    // [feat[t-2], feat[t-1], feat[t]] feeds the model at the same
    // GRU-step count upstream batch would have at output time t-2.
    // The model output is then the mask/coefs for output time t-2,
    // applied to spec[t-2] via the existing df_history time-alignment.
    // Net result: output is delayed 2 frames (20 ms) relative to input
    // but bit-equivalent to upstream batch.
    int frame_counter = 0;
    // One frame of warmup, not two — our STFT uses left-aligned framing
    // (frame k = audio[k·hop, k·hop + n_fft)) while libdf's `df.analysis`
    // uses center=true with reflection padding (frame k centered at
    // sample k·hop). This makes stream frame k ≈ libdf frame k+1, so the
    // 2-frame pad_feat discard upstream applies to libdf frames 0..1
    // collapses to discarding our stream frame 0 only (libdf frame 0 is
    // not in our stream at all). Empirically verified via
    // `/tmp/dfn_batch_ab.py`: stream feat[t] ≈ libdf feat[t+1] within
    // ~2·10⁻³ MAE, so feeding [0, 0, stream feat[1]] = [0, 0, libdf
    // feat[2]] to the encoder at the first model-active call reproduces
    // batch's `mask[t'=0]` input exactly.
    static constexpr int kModelWarmupFrames = 1;

    // ─── Streaming conv-history buffers ─────────────────────────────
    // The encoder's first conv stack has a 3-tap time kernel with
    // 2-frame causal left-pad. With pure zero-padding every frame,
    // the encoder always sees only 1 real input + 2 zeros, effectively
    // running as a 1-tap conv. To recover the 3-tap context across
    // streaming calls we keep the previous two feat_erb / feat_spec
    // frames and prepend them to the input each call.
    //
    // Layout matches the encoder input tensors:
    //   feat_erb_hist  : 2 frames × 32 ERB bands           (channel-major then time)
    //   feat_spec_hist : 2 frames × 2 channels × 96 bins
    std::vector<float> feat_erb_hist;
    std::vector<float> feat_spec_hist;

    // df_dec's `df_convp` has a 5-tap *time* conv with 4-frame causal
    // left-pad. We maintain the last 4 frames of the encoder's `c0`
    // output here so df_convp sees real context instead of zeros.
    //
    // Layout: 4 frames × 64 channels × 96 bins, with the
    // GGML-natural (freq, time, channel) ordering: address(c, t, f)
    // = (c * 4 + t) * 96 + f. Total 24576 floats per stream.
    std::vector<float> c0_hist;

    // ─── Persistent T=1 inference graph ─────────────────────────────
    // The full DFN3 graph (encoder + erb_dec + df_dec) is identical
    // across streaming frames at T=1, so we build it once and reuse
    // the topology + tensor allocations across `predict_dfn3` calls.
    // Per call we just sched_reset + sched_alloc + stage inputs +
    // compute + extract outputs — saving the ~2-10× cost of graph
    // construction every 10 ms frame.
    ggml_context* infer_ctx = nullptr;
    ggml_cgraph*  infer_gf  = nullptr;
    // Input tensor handles (allocated inside infer_ctx).
    ggml_tensor*  in_feat_erb  = nullptr;
    ggml_tensor*  in_feat_spec = nullptr;
    ggml_tensor*  in_h_enc     = nullptr;
    ggml_tensor*  in_h_erb1    = nullptr;
    ggml_tensor*  in_h_erb2    = nullptr;
    ggml_tensor*  in_h_df1     = nullptr;
    ggml_tensor*  in_h_df2     = nullptr;
    ggml_tensor*  in_c0_hist   = nullptr; // (96, 4, 64, 1) — last 4 c0 frames
    // Output tensor handles.
    ggml_tensor*  out_mask      = nullptr;
    ggml_tensor*  out_coefs     = nullptr;
    ggml_tensor*  out_alpha     = nullptr;
    ggml_tensor*  out_h_enc     = nullptr;
    ggml_tensor*  out_h_erb1    = nullptr;
    ggml_tensor*  out_h_erb2    = nullptr;
    ggml_tensor*  out_h_df1     = nullptr;
    ggml_tensor*  out_h_df2     = nullptr;
    ggml_tensor*  out_c0_current = nullptr; // current frame's c0 — pushed into c0_hist after compute
};

namespace {

void build_default_window(std::vector<float>& w, int n) {
    // Vorbis window: w[i] = sin(π/2 · sin²(π/2 · (i + 0.5) / (N/2))).
    // This is what libdf uses in `DFState::new`
    // (deep_filter/src/lib.rs:121). It satisfies w²[i] + w²[i+N/2] = 1
    // (so 50 % overlap-add still reconstructs perfectly), but has
    // different spectral characteristics than sqrt-Hann — softer
    // high-frequency content, which is what the model was trained on.
    // Using sqrt-Hann here produced a subtly "broken" output because
    // the encoder saw a different spectral envelope from training.
    w.resize(n);
    const double pi    = 3.14159265358979323846264338327950288;
    const double half  = (double)(n / 2);
    for (int i = 0; i < n; ++i) {
        double s = std::sin(0.5 * pi * ((double)i + 0.5) / half);
        w[i]     = (float)std::sin(0.5 * pi * s * s);
    }
}

// Exact port of `libdf::erb_fb` (deep_filter/src/lib.rs:62). The
// previous version was a linspace-and-clamp implementation that
// produced widths starting at 1 for the lowest bands, whereas libdf
// uses min_nb_erb_freqs=2 with a "borrow forward" tracker so the low
// bands all get exactly 2 bins. The model was trained against libdf's
// layout — any other layout makes feat_erb for the lowest 6 bands see
// completely different bin groupings and the model output is garbage.
void build_default_erb_table(std::vector<int>& widths, std::vector<int>& indices, int sample_rate, int n_fft,
                             int n_bands) {
    constexpr int kMinNbFreqs = 2;
    auto freq2erb = [](double f) { return 9.265 * std::log1p(f / (24.7 * 9.265)); };
    auto erb2freq = [](double e) { return 24.7 * 9.265 * (std::exp(e / 9.265) - 1.0); };

    const double nyq        = sample_rate / 2.0;
    const double freq_width = (double)sample_rate / (double)n_fft;
    const double erb_low    = freq2erb(0.0);
    const double erb_high   = freq2erb(nyq);
    const double step       = (erb_high - erb_low) / (double)n_bands;

    widths.assign(n_bands, 0);
    int prev_freq = 0;
    int freq_over = 0;
    for (int i = 1; i <= n_bands; ++i) {
        double f   = erb2freq(erb_low + (double)i * step);
        int    fb  = (int)std::round(f / freq_width);
        int    nb  = fb - prev_freq - freq_over;
        if (nb < kMinNbFreqs) {
            freq_over = kMinNbFreqs - nb;
            nb        = kMinNbFreqs;
        } else {
            freq_over = 0;
        }
        widths[i - 1] = nb;
        prev_freq     = fb;
    }
    widths[n_bands - 1] += 1; // account for the WINDOW_SIZE/2+1 bin count
    int sum_w = 0;
    for (int w : widths) sum_w += w;
    int too_large = sum_w - (n_fft / 2 + 1);
    if (too_large > 0) widths[n_bands - 1] -= too_large;

    indices.assign(n_bands, 0);
    int acc = 0;
    for (int i = 0; i < n_bands; ++i) {
        indices[i] = acc;
        acc += widths[i];
    }
}

// Mean band POWER (|x|² per bin, averaged over the bins in each ERB
// band). Matches libdf's `compute_band_corr(out, x, x, erb_fb)` with
// k = 1/band_size — i.e. arithmetic mean of squared magnitudes.
void compute_erb_power(const float* re, const float* im, int n_bins, const std::vector<int>& widths,
                        const std::vector<int>& indices, std::vector<float>& out_erb) {
    const int n_erb = (int)widths.size();
    out_erb.assign(n_erb, 0.0f);
    for (int b = 0; b < n_erb; ++b) {
        int start = indices[b];
        int end   = std::min(n_bins, start + widths[b]);
        float acc = 0.0f;
        for (int k = start; k < end; ++k) {
            acc += re[k] * re[k] + im[k] * im[k];
        }
        out_erb[b] = acc / std::max(1, end - start);
    }
}

// Upstream `band_mean_norm_erb` in libdf (Rust). Operates on dB-domain
// band power:
//   dB[b]    = 10 * log10(power[b] + 1e-10)
//   state[b] = (1-α)·dB[b] + α·state[b]
//   out[b]   = (dB[b] - state[b]) / 40
//
// `state` is seeded externally with linspace(-60, -90) on first frame.
void log_normalise_erb(const std::vector<float>& erb_power, std::vector<float>& state, bool& seeded,
                       std::vector<float>& out) {
    const int   n = (int)erb_power.size();
    const float a = kDfnEmaAlpha;
    out.resize(n);
    if (!seeded) {
        for (int b = 0; b < n; ++b) {
            float t   = n <= 1 ? 0.0f : (float)b / (float)(n - 1);
            state[b]  = kDfnMeanNormHi + t * (kDfnMeanNormLo - kDfnMeanNormHi); // linspace(-60, -90)
        }
        seeded = true;
    }
    for (int b = 0; b < n; ++b) {
        float dB  = 10.0f * std::log10(erb_power[b] + kDfnLogEpsilon);
        state[b]  = (1.0f - a) * dB + a * state[b];
        out[b]    = (dB - state[b]) / kDfnErbNormDiv;
    }
}

// Apply a per-ERB-band real gain to the full spectrum: every STFT bin
// is scaled by the gain of whatever ERB band it falls into. This is the
// canonical "Stage-1" mask in DFN.
void apply_erb_gains(float* re, float* im, int n_bins, const std::vector<int>& widths,
                     const std::vector<int>& indices, const std::vector<float>& gains) {
    const int n_erb = (int)widths.size();
    for (int b = 0; b < n_erb; ++b) {
        int   start = indices[b];
        int   end   = std::min(n_bins, start + widths[b]);
        float g     = gains[b];
        for (int k = start; k < end; ++k) {
            re[k] *= g;
            im[k] *= g;
        }
    }
}

// Slide the DF complex-spectrum history one frame: drop the oldest,
// append the current spectrum to the newest slot.
void df_history_push(dfn_stream* st, const float* re, const float* im, int n_bins) {
    const int n_frames = st->df_hist_frames;
    if (n_frames <= 0) return;
    const int stride = n_bins;
    // Shift older frames left by one slot.
    std::memmove(st->df_hist_re.data(), st->df_hist_re.data() + stride,
                 (size_t)stride * (n_frames - 1) * sizeof(float));
    std::memmove(st->df_hist_im.data(), st->df_hist_im.data() + stride,
                 (size_t)stride * (n_frames - 1) * sizeof(float));
    float* dst_re = st->df_hist_re.data() + (size_t)stride * (n_frames - 1);
    float* dst_im = st->df_hist_im.data() + (size_t)stride * (n_frames - 1);
    std::memcpy(dst_re, re, (size_t)n_bins * sizeof(float));
    std::memcpy(dst_im, im, (size_t)n_bins * sizeof(float));
}

// Apply DFN3's "deep filter": a per-frequency, per-tap complex linear
// filter over the recent spectrum history.
//
//   Y[t, f] = Σ_{d=0..order-1} C[t, f, d] · X[t + lookahead - d, f]
//
// for f in [0, df_bins). Bins above df_bins pass through unchanged.
// `coef_re`, `coef_im` are layout (df_order × df_bins), contiguous.
// `target` is the spectrum slot the result lands in (typically `st->spec_*`).
void apply_df_complex_filter(dfn_stream* st, const float* coef_re, const float* coef_im, float* target_re,
                             float* target_im, int n_bins, float /*alpha*/) {
    // DFN3 does not use the α gate — its `df_op` is called without
    // alpha and the filter output replaces the low bins outright
    // (see deepfilternet3.py:442 + modules.py::assign_df). The α
    // parameter is kept on the signature for future DFN2 support and
    // for callers that already pass it; we ignore it here.
    const dfn_model* m         = st->model;
    const int        df_bins   = std::min(m->df_bins, n_bins);
    const int        order     = m->df_order;
    const int        lookahead = m->lookahead_frms;
    const int        n_frames  = st->df_hist_frames;
    if (n_frames < order + lookahead) return;

    // Upstream `MF.DF.forward` (multiframe.py) unfolds the spec with
    // ConstantPad2d(pre = order-1-lookahead, post = lookahead) then
    // applies an einsum `"...tfn,...ntf->...tf"`. For the n-th tap
    // (n=0..order-1) and an output time t, this evaluates
    //   spec[t + n - (order-1-lookahead)]   =   spec[t + n - 2]   (DFN3)
    // i.e. coef[n=0] multiplies the OLDEST relevant frame (spec[t-2]),
    // coef[n=order-1] multiplies the NEWEST (spec[t+lookahead]).
    //
    // In our history ring (oldest at 0, newest at n_frames-1), with
    // current = newest - lookahead, the n-th tap reads from
    //   src = current + n - (order-1-lookahead)
    //       = current + n - order + 1 + lookahead
    // For order=5, lookahead=2: src = current + n - 2  →  past..future.
    // `df_hist` is the ORIGINAL (un-masked) spectrum — `df_history_push`
    // runs before `apply_erb_gains` in the caller.
    const int current_idx = n_frames - 1 - lookahead;        // network's "current" denoise target
    const int past_off    = order - 1 - lookahead;           // number of past frames in the window
    for (int f = 0; f < df_bins; ++f) {
        float y_re = 0.0f;
        float y_im = 0.0f;
        for (int d = 0; d < order; ++d) {
            int src_idx = current_idx + d - past_off;
            if (src_idx < 0 || src_idx >= n_frames) continue;
            float x_re = st->df_hist_re[(size_t)src_idx * n_bins + f];
            float x_im = st->df_hist_im[(size_t)src_idx * n_bins + f];
            float c_re = coef_re[(size_t)d * m->df_bins + f];
            float c_im = coef_im[(size_t)d * m->df_bins + f];
            y_re += c_re * x_re - c_im * x_im;
            y_im += c_re * x_im + c_im * x_re;
        }
        // Overwrite low bins with the filter result. Stage-1 ERB gain
        // applies only above df_bins (which already passed through
        // `apply_erb_gains` before us; we simply replace the low
        // portion now).
        target_re[f] = y_re;
        target_im[f] = y_im;
    }
}

// Identity fallback: gains = 1, DF coefs = δ at the centre tap. With
// these the spectrum passes through unchanged and the streaming
// round-trip is sample-accurate.
void predict_identity(const dfn_model* m, std::vector<float>& gains, std::vector<float>& coef_re,
                      std::vector<float>& coef_im) {
    gains.assign(m->n_erb, 1.0f);
    coef_re.assign((size_t)m->df_order * m->df_bins, 0.0f);
    coef_im.assign((size_t)m->df_order * m->df_bins, 0.0f);
    int delta = m->lookahead_frms; // tap index that corresponds to the centre frame
    if (delta < 0 || delta >= m->df_order)
        delta = 0;
    for (int f = 0; f < m->df_bins; ++f)
        coef_re[(size_t)delta * m->df_bins + f] = 1.0f;
}

// Forward declaration — actual GGML implementation lives further down,
// after all the encoder/decoder graph builders are defined. File-scope
// `static` linkage because the builders live in a different anonymous
// namespace lower in the file; two anonymous namespaces are distinct
// scopes.
} // namespace (close so the forward decl lives at file scope)
static bool predict_dfn3(const dfn_model* m, const std::vector<float>& log_erb_feat,
                          const std::vector<float>& erb_power, const float* spec_re, const float* spec_im, int n_bins,
                          dfn_stream* stream, std::vector<float>& gains, std::vector<float>& coef_re,
                          std::vector<float>& coef_im, float* alpha_out);
namespace { // reopen the original anonymous namespace

// One analysis → predict → mask → synthesis → OLA frame.
void process_one_frame(dfn_stream* st, const float* in_n_fft) {
    const dfn_model* m   = st->model;
    const int n_fft      = m->fft_size;
    const int hop        = m->frame_size;
    const int n_bins     = n_fft / 2 + 1;

    if ((int)st->spec_re.size() < n_bins) {
        st->spec_re.assign(n_bins, 0.0f);
        st->spec_im.assign(n_bins, 0.0f);
    }
    if ((int)st->frame.size() < n_fft)
        st->frame.assign(n_fft, 0.0f);
    for (int i = 0; i < n_fft; ++i)
        st->frame[i] = in_n_fft[i] * m->window[i];
    dfn_rfft(m->fft, st->frame.data(), st->spec_re.data(), st->spec_im.data());

    // Push the analysis spectrum into the DF history before applying
    // any masking — the deep filter wants the un-masked context.
    df_history_push(st, st->spec_re.data(), st->spec_im.data(), n_bins);

    // ERB-band features → 10·log10(power) → EMA-normalise.
    //
    // libdf's `frame_analysis` multiplies the post-FFT spec by
    // `wnorm = 2·hop/N²` before doing anything else. We don't apply
    // it to the spec used for masking/iSTFT (`coefs` are
    // scale-equivariant so the output magnitude follows the input);
    // but we DO need to apply it to the values fed into the feature
    // EMAs, since `feat_spec = spec / sqrt(state)` is √-sensitive to
    // the spec magnitude (so even in steady state, native-scale spec
    // gives `feat_spec` that's √N ≈ 31× off from training-time).
    const float wnorm    = 2.0f * (float)m->frame_size / ((float)m->fft_size * (float)m->fft_size);
    const float wnorm_sq = wnorm * wnorm;
    std::vector<float> erb_power, log_feat;
    compute_erb_power(st->spec_re.data(), st->spec_im.data(), n_bins, m->erb_widths, m->erb_indices, erb_power);
    for (auto& v : erb_power) v *= wnorm_sq;
    log_normalise_erb(erb_power, st->log_mag_mean, st->norm_seeded, log_feat);

    // ─── Lookahead alignment warmup ──────────────────────────────────
    // Upstream's `pad_feat = ConstantPad2d((0, 0, -2, 2))` shifts the
    // encoder's feature stream forward by 2 frames before the encoder
    // sees them — so mask[t'] is computed from feat[t'..t'+2] (current
    // + 2 future). Our STFT uses left-aligned framing while libdf's
    // `df.analysis` uses center=true with reflection padding, which
    // gives stream-frame[k] ≈ libdf-frame[k+1] — a free 1-frame shift.
    // The remaining work is to discard one more input frame on our
    // side so encoder input at the first model-active call is
    // [0, 0, feat[2]] (= libdf-pad_feat[0..2]). We do this with a
    // 1-frame warmup: keep updating the EMAs (libdf updates them on
    // every frame before the shift trims) but skip pushing feat into
    // the encoder history. Empirically (`/tmp/dfn_batch_ab.py`) the
    // streaming mask at output time 0 then matches batch within
    // 4·10⁻³ max-abs and overall correlation > 0.99 vs upstream ONNX.
    constexpr int kEncFInFeat = 32;
    constexpr int kEncFInSpec = 96;
    if (st->frame_counter < dfn_stream::kModelWarmupFrames) {
        // Update the unit_norm EMA on the current spec (upstream's
        // `unit_norm` runs across ALL frames before the pad_feat shift
        // trims them, so the EMA at output time t' has been updated
        // through every spec[0..t'+2]) — but DO NOT push the resulting
        // feat_spec / log_feat into the encoder history buffers. The
        // `pad_feat = ConstantPad2d((0,0,-2,2))` shift discards feat[0]
        // and feat[1] outright; for output time 0 the encoder sees
        // [0, 0, feat[2]], not [feat[0], feat[1], feat[2]]. Keeping
        // feat_*_hist at zero through the warmup yields the same
        // encoder input upstream batch sees at t'=0 and t'=1.
        const float wnorm_unit = 2.0f * (float)m->frame_size / ((float)m->fft_size * (float)m->fft_size);
        if (!st->unit_norm_seeded) {
            for (int f = 0; f < kEncFInSpec; ++f) {
                float t = kEncFInSpec <= 1 ? 0.0f : (float)f / (float)(kEncFInSpec - 1);
                st->unit_norm_state[f] = kDfnUnitNormHi + t * (kDfnUnitNormLo - kDfnUnitNormHi);
            }
            st->unit_norm_seeded = true;
        }
        for (int f = 0; f < kEncFInSpec && f < n_bins; ++f) {
            float re_wn = st->spec_re[f] * wnorm_unit;
            float im_wn = st->spec_im[f] * wnorm_unit;
            float mag   = std::sqrt(re_wn * re_wn + im_wn * im_wn);
            st->unit_norm_state[f] = (1.0f - kDfnEmaAlpha) * mag + kDfnEmaAlpha * st->unit_norm_state[f];
        }
        ++st->frame_counter;
        return; // no output emitted during warmup
    }
    ++st->frame_counter;

    // ─── Neural-net hook ─────────────────────────────────────────────
    // Real DFN3 graph runs when the GGUF carries trained weights;
    // otherwise identity fallback keeps the spectrum sample-accurate.
    std::vector<float> gains, coef_re, coef_im;
    float alpha = 1.0f; // pure Stage-1 (identity DF) under the fallback
    bool  ok    = false;
    if (m->has_neural_weights) {
        ok = predict_dfn3(m, log_feat, erb_power, st->spec_re.data(), st->spec_im.data(), n_bins, st, gains, coef_re,
                          coef_im, &alpha);
    }
    if (!ok)
        predict_identity(m, gains, coef_re, coef_im);
    // ─────────────────────────────────────────────────────────────────

    // ─── Streaming time-alignment fix ───────────────────────────────
    // Replace the just-FFT'd spec[N] in `spec_re/spec_im` with the
    // OLDER frame from df_history corresponding to the network's
    // "current denoise target" — spec[N - lookahead]. Stage-1 mask
    // and Stage-2 DF filter results are both *for* that older time;
    // iSTFT'ing a Frankenstein mix of spec[N] (high bins) and
    // DF-filtered spec[N-2] (low bins) was producing exactly the
    // 2-frame time-misalignment artefact heard at output start.
    //
    // The mask is still `mask[N]` (latest model output) — we don't
    // buffer masks across frames, matching upstream's
    // `forward_real_hidden_state_loop` streaming behaviour
    // (modules.py:446).
    {
        const int t_idx = st->df_hist_frames - 1 - m->lookahead_frms;
        const float* hist_re = st->df_hist_re.data() + (size_t)t_idx * n_bins;
        const float* hist_im = st->df_hist_im.data() + (size_t)t_idx * n_bins;
        std::memcpy(st->spec_re.data(), hist_re, (size_t)n_bins * sizeof(float));
        std::memcpy(st->spec_im.data(), hist_im, (size_t)n_bins * sizeof(float));
    }

    // DEBUG: skip mask/DF filter to verify pure STFT/iSTFT round-trip.
    static const bool dfn_bypass = std::getenv("DFN_BYPASS") != nullptr;
    if (!dfn_bypass) {
        // Stage 1: per-band gain applied to spec[N - lookahead].
        apply_erb_gains(st->spec_re.data(), st->spec_im.data(), n_bins, m->erb_widths, m->erb_indices, gains);

        // Stage 2: deep filter writes the low bins (output for time
        // N - lookahead, using coefs[N] applied to df_history's
        // spec[N - lookahead ± 2] window).
        apply_df_complex_filter(st, coef_re.data(), coef_im.data(), st->spec_re.data(), st->spec_im.data(), n_bins,
                                alpha);
    }

    // iSTFT + synthesis window + overlap-add.
    dfn_irfft(m->fft, st->spec_re.data(), st->spec_im.data(), st->frame.data());
    for (int i = 0; i < n_fft; ++i)
        st->frame[i] *= m->window[i];

    if ((int)st->ola.size() < n_fft)
        st->ola.assign(n_fft, 0.0f);
    for (int i = 0; i < n_fft; ++i)
        st->ola[i] += st->frame[i];

    st->out_queue.insert(st->out_queue.end(), st->ola.begin(), st->ola.begin() + hop);
    std::memmove(st->ola.data(), st->ola.data() + hop, (size_t)(n_fft - hop) * sizeof(float));
    std::memset(st->ola.data() + (n_fft - hop), 0, (size_t)hop * sizeof(float));
}

} // namespace

extern "C" {

struct dfn_params dfn_default_params(void) {
    dfn_params p{};
    p.n_threads    = 2;
    p.use_gpu      = false;
    p.verbosity    = 0;
    p.atten_lim_db = 100.0f;
    return p;
}

// True iff the loaded WeightLoad carries trained DFN3 weights. Probe
// names are taken verbatim from the ONNX export
// (`models/convert-deepfilternet-to-gguf.py`):
//
//   enc.onnx     : dfn.enc.{erb_conv1,emb_gru.linear_in}.0.weight, df_conv0.1.weight, ...
//   erb_dec.onnx : dfn.erb_dec.{emb_gru.linear_in,convt1}.0.weight, ...
//   df_dec.onnx  : dfn.df_dec.{df_convp.1,df_gru.linear_in.0,df_out.0}.weight, ...
//
// A metadata-only GGUF (ERB tables + window but no trained weights)
// flunks every probe and the predict step falls back to identity.
static bool gguf_has_dfn3_weights(const std::map<std::string, ggml_tensor*>& tensors) {
    static const char* kProbes[] = {
        "dfn.enc.erb_conv1.0.weight",
        "dfn.enc.emb_gru.linear_in.0.weight",
        "dfn.erb_dec.emb_gru.linear_in.0.weight",
        "dfn.df_dec.df_gru.linear_in.0.weight",
        "dfn.df_dec.df_convp.1.weight",
    };
    for (const char* probe : kProbes) {
        if (tensors.count(probe)) return true;
    }
    return false;
}

struct dfn_model* dfn_model_load(const char* gguf_path, struct dfn_params params) {
    if (!gguf_path)
        return nullptr;

    // Pass 1 — metadata. We need this before we can allocate the
    // backend buffer (no shape decisions to make from the keys — but
    // we want to validate the architecture tag and read the sizes
    // before mmap'ing weights we may not want).
    gguf_context* meta = core_gguf::open_metadata(gguf_path);
    if (!meta) {
        if (params.verbosity > 0)
            fprintf(stderr, "dfn: failed to open GGUF '%s'\n", gguf_path);
        return nullptr;
    }
    std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    if (!arch.empty() && arch != "deepfilternet3") {
        if (params.verbosity > 0)
            fprintf(stderr, "dfn: GGUF arch '%s' != 'deepfilternet3'\n", arch.c_str());
        core_gguf::free_metadata(meta);
        return nullptr;
    }

    auto* m   = new dfn_model();
    m->params = params;

    m->sample_rate    = (int)core_gguf::kv_u32(meta, "dfn.sample_rate", kDfnSampleRate);
    m->frame_size     = (int)core_gguf::kv_u32(meta, "dfn.hop", kDfnFrameSize);
    m->fft_size       = (int)core_gguf::kv_u32(meta, "dfn.n_fft", kDfnFftSize);
    m->lookahead_frms = (int)core_gguf::kv_u32(meta, "dfn.lookahead_frames", kDfnLookaheadFrms);
    m->n_erb          = (int)core_gguf::kv_u32(meta, "dfn.n_erb", kDfnNErb);
    m->df_bins        = (int)core_gguf::kv_u32(meta, "dfn.df_bins", kDfnDfBins);
    m->df_order       = (int)core_gguf::kv_u32(meta, "dfn.df_order", kDfnDfOrder);

    core_gguf::free_metadata(meta);

    // Pass 2 — backend + weight buffer. CPU only for now: the DFN3
    // network is tiny (~3 MB f16) so the GPU launch overhead would
    // dwarf the per-frame inference cost. `use_gpu` is reserved for
    // when we re-evaluate after the graph lands.
    m->backend = ggml_backend_cpu_init();
    if (!m->backend) {
        delete m;
        return nullptr;
    }
    if (!core_gguf::load_weights(gguf_path, m->backend, "dfn", m->weights)) {
        if (params.verbosity > 0)
            fprintf(stderr, "dfn: load_weights failed for '%s'\n", gguf_path);
        ggml_backend_free(m->backend);
        delete m;
        return nullptr;
    }

    // sqrt-Hann window: GGUF if present, otherwise rebuild.
    ggml_tensor* win_t = core_gguf::try_get(m->weights.tensors, "dfn.window");
    if (win_t && win_t->type == GGML_TYPE_F32 && ggml_nelements(win_t) == m->fft_size) {
        m->window.assign((float*)win_t->data, (float*)win_t->data + m->fft_size);
    } else {
        build_default_window(m->window, m->fft_size);
    }

    // ERB filterbank — GGUF int32 tensors, fallback to recomputed.
    auto load_i32 = [](ggml_tensor* t, std::vector<int>& out) {
        if (!t || t->type != GGML_TYPE_I32) return false;
        int64_t n = ggml_nelements(t);
        out.assign((int*)t->data, (int*)t->data + n);
        return true;
    };
    bool ok_w = load_i32(core_gguf::try_get(m->weights.tensors, "dfn.erb_widths"), m->erb_widths);
    bool ok_i = load_i32(core_gguf::try_get(m->weights.tensors, "dfn.erb_indices"), m->erb_indices);
    if (!ok_w || !ok_i)
        build_default_erb_table(m->erb_widths, m->erb_indices, m->sample_rate, m->fft_size, m->n_erb);

    m->has_neural_weights = gguf_has_dfn3_weights(m->weights.tensors);

    m->fft = dfn_fft_plan_create(m->fft_size);
    if (!m->fft) {
        core_gguf::free_weights(m->weights);
        ggml_backend_free(m->backend);
        delete m;
        return nullptr;
    }

    // Backend scheduler. CPU-only for now; the last backend in the
    // list must be the CPU one (sched assertion — see GH issue #68).
    if (m->has_neural_weights) {
        ggml_backend_t backends[1] = {m->backend};
        // graph_size sets the sched hash-set capacity — must exceed
        // the largest graph we'll run. Streaming (T=1) needs ~1k
        // nodes; batch mode (`dfn_debug_full_predict`, T up to ~200)
        // needs ~30k. 64k covers both with headroom.
        m->sched = ggml_backend_sched_new(backends, nullptr, 1, /*graph_size=*/65536,
                                          /*parallel=*/false, /*op_offload=*/false);
        if (!m->sched) {
            // Non-fatal: predict_dfn3 will return false and the
            // identity fallback continues to run.
            if (params.verbosity > 0)
                fprintf(stderr, "dfn: sched alloc failed; running with identity fallback\n");
        }
    }
    if (params.verbosity > 0) {
        fprintf(stderr,
                "dfn: model loaded — sr=%d n_fft=%d hop=%d n_erb=%d df_bins=%d df_order=%d lookahead=%d "
                "neural_weights=%s tensors=%zu\n",
                m->sample_rate, m->fft_size, m->frame_size, m->n_erb, m->df_bins, m->df_order, m->lookahead_frms,
                m->has_neural_weights ? "yes" : "no (identity fallback)", m->weights.tensors.size());
    }
    return m;
}

void dfn_model_free(struct dfn_model* m) {
    if (!m) return;
    if (m->fft) dfn_fft_plan_free(m->fft);
    if (m->sched) ggml_backend_sched_free(m->sched);
    core_gguf::free_weights(m->weights);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

int dfn_model_sample_rate(const struct dfn_model* m) { return m ? m->sample_rate : kDfnSampleRate; }
int dfn_model_frame_size(const struct dfn_model* m) { return m ? m->frame_size : kDfnFrameSize; }
int dfn_model_lookahead_frames(const struct dfn_model* m) { return m ? m->lookahead_frms : kDfnLookaheadFrms; }

struct dfn_stream* dfn_stream_create(const struct dfn_model* m) {
    if (!m) return nullptr;
    auto* st = new dfn_stream();
    st->model = m;
    const int n_fft  = m->fft_size;
    const int n_bins = n_fft / 2 + 1;

    st->in_buf.reserve((size_t)n_fft * 4);
    st->frame.assign(n_fft, 0.0f);
    st->ola.assign(n_fft, 0.0f);
    st->spec_re.assign(n_bins, 0.0f);
    st->spec_im.assign(n_bins, 0.0f);

    st->log_mag_mean.assign(m->n_erb, 0.0f);
    st->norm_seeded = false;
    st->unit_norm_state.assign(m->df_bins, 0.0f);
    st->unit_norm_seeded = false;

    st->df_hist_frames = m->df_order + m->lookahead_frms;
    st->df_hist_re.assign((size_t)st->df_hist_frames * n_bins, 0.0f);
    st->df_hist_im.assign((size_t)st->df_hist_frames * n_bins, 0.0f);

    st->gru_hidden.assign((size_t)dfn_stream::kNumGruStates * dfn_stream::kGruHidden, 0.0f);
    st->feat_erb_hist.assign((size_t)2 * 32, 0.0f);
    st->feat_spec_hist.assign((size_t)2 * 2 * 96, 0.0f);
    st->c0_hist.assign((size_t)4 * 64 * 96, 0.0f);
    return st;
}

void dfn_stream_reset(struct dfn_stream* st) {
    if (!st) return;
    st->in_buf.clear();
    std::fill(st->ola.begin(), st->ola.end(), 0.0f);
    st->out_queue.clear();
    std::fill(st->log_mag_mean.begin(), st->log_mag_mean.end(), 0.0f);
    st->norm_seeded = false;
    std::fill(st->unit_norm_state.begin(), st->unit_norm_state.end(), 0.0f);
    st->unit_norm_seeded = false;
    std::fill(st->df_hist_re.begin(), st->df_hist_re.end(), 0.0f);
    std::fill(st->df_hist_im.begin(), st->df_hist_im.end(), 0.0f);
    std::fill(st->gru_hidden.begin(), st->gru_hidden.end(), 0.0f);
    std::fill(st->feat_erb_hist.begin(), st->feat_erb_hist.end(), 0.0f);
    std::fill(st->feat_spec_hist.begin(), st->feat_spec_hist.end(), 0.0f);
    std::fill(st->c0_hist.begin(), st->c0_hist.end(), 0.0f);
    st->frame_counter = 0;
}

void dfn_stream_free(struct dfn_stream* st) {
    if (!st) return;
    if (st->infer_ctx) ggml_free(st->infer_ctx);
    delete st;
}

int dfn_stream_warmup(struct dfn_stream* st, int n_frames) {
    if (!st || n_frames <= 0) return 0;
    const int hop          = st->model->frame_size;
    // Push n_frames worth of silence through the full pipeline,
    // throwing away whatever comes out the other side. The lookahead
    // buffer holds the resulting samples internally and would emit
    // them on subsequent calls — we drain the entire output queue at
    // the end so the next real `dfn_stream_process` call doesn't see
    // any leftover warm-up audio.
    std::vector<float> silence((size_t)hop * (size_t)n_frames, 0.0f);
    std::vector<float> sink(silence.size() + 4096, 0.0f);
    int                produced = 0;
    int rc = dfn_stream_process(st, silence.data(), (int)silence.size(), sink.data(), (int)sink.size(), &produced,
                                /*final_chunk=*/false);
    if (rc != 0) return rc;
    // Drain anything still buffered behind the lookahead delay so the
    // warm-up audio doesn't leak into the first real output. We do
    // this without flagging end-of-stream because the caller will
    // continue feeding real audio.
    while ((int)st->out_queue.size() > st->model->lookahead_frms * hop) {
        st->out_queue.erase(st->out_queue.begin(),
                             st->out_queue.begin() + ((int)st->out_queue.size() - st->model->lookahead_frms * hop));
    }
    return 0;
}

int dfn_stream_process(struct dfn_stream* st, const float* in, int n_in, float* out, int out_cap, int* produced,
                       bool final_chunk) {
    if (!st || !out || !produced) return -1;
    if (n_in < 0 || out_cap < 0) return -1;
    *produced = 0;

    const int n_fft = st->model->fft_size;
    const int hop   = st->model->frame_size;

    if (in && n_in > 0)
        st->in_buf.insert(st->in_buf.end(), in, in + n_in);

    while ((int)st->in_buf.size() >= n_fft) {
        process_one_frame(st, st->in_buf.data());
        st->in_buf.erase(st->in_buf.begin(), st->in_buf.begin() + hop);
    }

    if (final_chunk) {
        if (!st->in_buf.empty()) {
            std::vector<float> padded(n_fft, 0.0f);
            std::memcpy(padded.data(), st->in_buf.data(), st->in_buf.size() * sizeof(float));
            process_one_frame(st, padded.data());
            st->in_buf.clear();
        }
        st->out_queue.insert(st->out_queue.end(), st->ola.begin(), st->ola.begin() + hop);
        std::fill(st->ola.begin(), st->ola.end(), 0.0f);
    }

    const int delay = st->model->lookahead_frms * hop;
    int       avail = (int)st->out_queue.size() - (final_chunk ? 0 : delay);
    if (avail <= 0) return 0;
    int emit = std::min(avail, out_cap);
    if (emit <= 0) return 0;
    std::memcpy(out, st->out_queue.data(), (size_t)emit * sizeof(float));
    st->out_queue.erase(st->out_queue.begin(), st->out_queue.begin() + emit);
    *produced = emit;
    return 0;
}

} // extern "C"  — namespaces below have C++ linkage; the test-only
//                  entries reopen `extern "C"` further down.

// ─── Debug / golden-value entry points ───────────────────────────────
//
// Test-only surface that drives one sub-graph of the encoder at a
// time so the implementation can be landed layer-by-layer with
// numerical pass/fail. Not part of the public `dfn.h` ABI — these are
// declared in `dfn_debug.h` and used by `tests/test-dfn-graph.cpp`.

namespace {

// Inputs to the ERB-stream branch helper.
struct enc_erb_io {
    ggml_tensor* e0 = nullptr; // (32, T, 64, 1)
    ggml_tensor* e1 = nullptr; // (16, T, 64, 1)
    ggml_tensor* e2 = nullptr; // ( 8, T, 64, 1)
    ggml_tensor* e3 = nullptr; // ( 8, T, 64, 1)
};

struct enc_df_io {
    ggml_tensor* c0      = nullptr; // (96, T, 64, 1) — feat after df_conv0 + Relu
    ggml_tensor* df_feat = nullptr; // (48, T, 64, 1) — feat after df_conv1 + Relu (for df_fc_emb)
};

struct enc_combine_io {
    ggml_tensor* df_emb   = nullptr; // (512, T, 1, 1) — post df_fc_emb GroupedLinear + Relu
    ggml_tensor* combined = nullptr; // (512, T, 1, 1) — Add(e3_flat, df_emb), input to emb_gru
};

struct enc_gru_io {
    ggml_tensor* pre_gru = nullptr; // (256, T, 1, 1) — linear_in + Relu, input to GRU
    ggml_tensor* emb     = nullptr; // (512, T, 1, 1) — linear_out + Relu, final encoder output
    ggml_tensor* h_seq   = nullptr; // (256, T, 1, 1) — GRU outputs (= final hidden state for T=1)
};

// Build the ERB-feature branch of the encoder: feat_erb → e0 → e1 → e2 → e3.
// `feat_erb_padded` already has PAD_T=2 zero frames on its time axis (left)
// so the 3×3 conv0 runs with time-pad=0 inside ggml.
enc_erb_io build_enc_erb_branch(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* feat_erb_padded) {
    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };
    auto B = [&](const char* n) {
        return ggml_reshape_4d(ctx0, ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32), 1, 1, 64, 1);
    };

    // Conv0: 3×3 g=1 — see node #17. Bias broadcast over W/H/N.
    ggml_tensor* x = ggml_conv_2d(ctx0, W("dfn.enc.onnx::Conv_282"), feat_erb_padded,
                                  /*sW=*/1, /*sH=*/1, /*pW=*/1, /*pH=*/0, /*dW=*/1, /*dH=*/1);
    x = ggml_add(ctx0, x, B("dfn.enc.onnx::Conv_283"));
    x = ggml_relu(ctx0, x);
    ggml_tensor* e0 = x;
    ggml_set_name(e0, "e0");

    // erb_conv1: depthwise 1×3 group=64 stride 1×2 + pointwise 1×1.
    x = ggml_conv_2d_dw(ctx0, W("dfn.enc.erb_conv1.0.weight"), e0,
                        /*sW=*/2, /*sH=*/1, /*pW=*/1, /*pH=*/0, /*dW=*/1, /*dH=*/1);
    x = ggml_conv_2d(ctx0, W("dfn.enc.onnx::Conv_285"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.enc.onnx::Conv_286"));
    x = ggml_relu(ctx0, x);
    ggml_tensor* e1 = x;
    ggml_set_name(e1, "e1");

    // erb_conv2: depthwise 1×3 g=64 stride 1×2 + pointwise 1×1.
    x = ggml_conv_2d_dw(ctx0, W("dfn.enc.erb_conv2.0.weight"), e1, 2, 1, 1, 0, 1, 1);
    x = ggml_conv_2d(ctx0, W("dfn.enc.onnx::Conv_288"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.enc.onnx::Conv_289"));
    x = ggml_relu(ctx0, x);
    ggml_tensor* e2 = x;
    ggml_set_name(e2, "e2");

    // erb_conv3: depthwise 1×3 g=64 stride 1×1 + pointwise 1×1.
    x = ggml_conv_2d_dw(ctx0, W("dfn.enc.erb_conv3.0.weight"), e2, 1, 1, 1, 0, 1, 1);
    x = ggml_conv_2d(ctx0, W("dfn.enc.onnx::Conv_291"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.enc.onnx::Conv_292"));
    x = ggml_relu(ctx0, x);
    ggml_tensor* e3 = x;
    ggml_set_name(e3, "e3");

    return {e0, e1, e2, e3};
}

// Build the DF-feature branch of the encoder: feat_spec → c0 → df_feat.
// `feat_spec_padded` already has 2 zero time-frames prepended (causal pad
// for df_conv0's 3×3 kernel).
//
// `df_conv0` is a grouped conv (group=2, IC/group=1, OC=64): each input
// channel (re, im) feeds 32 disjoint output channels. GGML has no native
// groups parameter for conv_2d, so we split manually into two 1-channel
// sub-convs and concat the results along the channel axis. PyTorch /
// onnxruntime lower it the same way internally.
enc_df_io build_enc_df_branch(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* feat_spec_padded) {
    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };
    auto B = [&](const char* n) {
        return ggml_reshape_4d(ctx0, ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32), 1, 1, 64, 1);
    };

    // Weight `df_conv0.1.weight` ne=[3, 3, 1, 64]. Split along OC into
    // two ne=[3, 3, 1, 32] sub-kernels.
    ggml_tensor* w_full = W("dfn.enc.df_conv0.1.weight");
    ggml_tensor* w0 = ggml_view_4d(ctx0, w_full, 3, 3, 1, 32, w_full->nb[1], w_full->nb[2], w_full->nb[3], 0);
    ggml_tensor* w1 = ggml_view_4d(ctx0, w_full, 3, 3, 1, 32, w_full->nb[1], w_full->nb[2], w_full->nb[3],
                                   32 * w_full->nb[3]);
    // Conv ops need contiguous kernels (the im2col path indexes them
    // assuming standard strides).
    w0 = ggml_cont(ctx0, w0);
    w1 = ggml_cont(ctx0, w1);

    // Input `feat_spec_padded` ne=[96, T+2, 2, 1]. Split along channel
    // into two ne=[96, T+2, 1, 1] sub-inputs.
    ggml_tensor* in_full = feat_spec_padded;
    ggml_tensor* in0 = ggml_view_4d(ctx0, in_full, in_full->ne[0], in_full->ne[1], 1, 1, in_full->nb[1], in_full->nb[2],
                                    in_full->nb[3], 0);
    ggml_tensor* in1 = ggml_view_4d(ctx0, in_full, in_full->ne[0], in_full->ne[1], 1, 1, in_full->nb[1], in_full->nb[2],
                                    in_full->nb[3], in_full->nb[2]);
    in0 = ggml_cont(ctx0, in0);
    in1 = ggml_cont(ctx0, in1);

    // Each sub-conv: KW=3 KH=3 IC=1 OC=32, stride 1×1, pad freq=1 time=0.
    ggml_tensor* out0 = ggml_conv_2d(ctx0, w0, in0, 1, 1, 1, 0, 1, 1); // (96, T, 32, 1)
    ggml_tensor* out1 = ggml_conv_2d(ctx0, w1, in1, 1, 1, 1, 0, 1, 1);
    // Concat along the channel axis (ne[2]) → (96, T, 64, 1).
    ggml_tensor* x = ggml_concat(ctx0, out0, out1, /*dim=*/2);

    // Pointwise 1×1 64→64 + bias broadcast + Relu → c0.
    x = ggml_conv_2d(ctx0, W("dfn.enc.onnx::Conv_294"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.enc.onnx::Conv_295"));
    x = ggml_relu(ctx0, x);
    ggml_tensor* c0 = x;
    ggml_set_name(c0, "c0");

    // df_conv1: depthwise 1×3 g=64 stride 1×2 + pointwise 1×1 + Relu →
    // df_feat (96 → 48 freq bins). Used downstream by df_fc_emb; not
    // currently exposed as a golden target but the cost is trivial.
    x = ggml_conv_2d_dw(ctx0, W("dfn.enc.df_conv1.0.weight"), c0, 2, 1, 1, 0, 1, 1);
    x = ggml_conv_2d(ctx0, W("dfn.enc.onnx::Conv_297"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.enc.onnx::Conv_298"));
    x = ggml_relu(ctx0, x);
    ggml_tensor* df_feat = x;
    ggml_set_name(df_feat, "df_feat");

    return {c0, df_feat};
}

// Build df_fc_emb (GroupedLinear via batched mul_mat) + the e3 reshape
// + combine_add. Produces the 512-dim sequence fed into emb_gru.
//
// PyTorch / ONNX flow:
//   df_feat (1, 64, T, 48) → transpose(0,2,3,1) → (1, T, 48, 64)
//                          → reshape (1, T, 32, 96)
//                          → einsum "btgi,gio->btgo" with weight (32, 96, 16)
//                          → (1, T, 32, 16) → reshape (1, T, 512) → Relu → df_emb
//   e3      (1, 64, T,  8) → transpose(0,2,3,1) → (1, T, 8, 64)
//                          → reshape (1, T, 512) → e3_flat
//   combined = e3_flat + df_emb
enc_combine_io build_enc_combine(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* df_feat,
                                  ggml_tensor* e3) {
    const int T = (int)df_feat->ne[1]; // (W=48, H=T, C=64, N=1)

    // ── df_fc_emb branch ───────────────────────────────────────────
    // df_feat ne=[48, T, 64, 1] → permute to ne=[64, 48, T, 1]
    // (PyTorch transpose perm=[0,2,3,1] semantically (1, T, 48, 64))
    //
    // ggml_permute(a, axis0, axis1, axis2, axis3) sends old axis N to
    // new axis axisN (i.e. new ne[axisN] = old ne[N]). To produce
    // new ne=[64, 48, T, 1] from old ne=[48, T, 64, 1] we want:
    //   old 0 (=48) → new 1   ⇒ axis0=1
    //   old 1 (=T)  → new 2   ⇒ axis1=2
    //   old 2 (=64) → new 0   ⇒ axis2=0
    //   old 3 (=1)  → new 3   ⇒ axis3=3
    ggml_tensor* x = ggml_cont(ctx0, ggml_permute(ctx0, df_feat, 1, 2, 0, 3));
    // Reshape to (1, T, 32, 96) → GGML ne=[96, 32, T, 1]. Same flat
    // memory layout because 48*64 = 32*96 = 3072.
    x = ggml_reshape_4d(ctx0, x, 96, 32, T, 1);
    // Mul_mat batching wants (k, m, b, 1) for the input.
    // Currently k=96 at ne[0] ✓, batch=32 at ne[1] but we need T at
    // ne[1] and groups at ne[2]. Swap axes 1 and 2.
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // ne=[96, T, 32, 1]

    // Weight GGUF shape ne=[16, 96, 32, 1] (numpy (32, 96, 16)).
    // Need ne=[96, 16, 32, 1] (k=96, n=16, batch=32) for mul_mat.
    ggml_tensor* w_emb_raw = core_gguf::try_get(wl.tensors, "dfn.enc.df_fc_emb.0.weight");
    ggml_tensor* w_emb_f32 = ggml_cast(ctx0, w_emb_raw, GGML_TYPE_F32);
    ggml_tensor* w_emb = ggml_cont(ctx0, ggml_permute(ctx0, w_emb_f32, 1, 0, 2, 3));

    // Batched matmul: for each group g, out[t, g, o] = sum_i x[t, g, i] * w[g, i, o]
    x = ggml_mul_mat(ctx0, w_emb, x); // ne=[16, T, 32, 1]

    // Flatten (groups × out_per_group) into a 512-dim feature:
    // ne=[16, T, 32, 1] → ne=[16, 32, T, 1] (groups next to out) → ne=[512, T, 1, 1].
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // ne=[16, 32, T, 1]
    x = ggml_reshape_4d(ctx0, x, 512, T, 1, 1);
    x = ggml_relu(ctx0, x);
    ggml_tensor* df_emb = x;
    ggml_set_name(df_emb, "df_emb");

    // ── e3 reshape ─────────────────────────────────────────────────
    // e3 ne=[8, T, 64, 1] → ne=[64, 8, T, 1] (PyTorch (1, T, 8, 64))
    // → reshape ne=[512, T, 1, 1]. Same axis-shift as df_feat above.
    ggml_tensor* e3_flat = ggml_cont(ctx0, ggml_permute(ctx0, e3, 1, 2, 0, 3));
    e3_flat              = ggml_reshape_4d(ctx0, e3_flat, 512, T, 1, 1);
    ggml_set_name(e3_flat, "e3_flat");

    // ── combine ────────────────────────────────────────────────────
    ggml_tensor* combined = ggml_add(ctx0, e3_flat, df_emb);
    ggml_set_name(combined, "combined");

    return {df_emb, combined};
}

// GroupedLinear via batched mul_mat.
//
//   input  ne=[in_dim,  T, 1, 1]   (in_dim  = n_groups · in_per_group)
//   weight ne=[out_per_group, in_per_group, n_groups, 1]  (GGUF layout)
//   output ne=[out_dim, T, 1, 1]   (out_dim = n_groups · out_per_group)
//
// Equivalent to einsum "btgi,gih->btgh" with input reshaped to
// (T, g, i) and weight (g, i, h).
ggml_tensor* build_grouped_linear(ggml_context* ctx0, const core_gguf::WeightLoad& wl, const char* weight_name,
                                   ggml_tensor* input, int n_groups, int in_per_group, int out_per_group) {
    const int T       = (int)input->ne[1];
    const int out_dim = n_groups * out_per_group;

    // input ne=[in_dim, T, 1, 1] → ne=[in_per_group, n_groups, T, 1]
    ggml_tensor* x = ggml_reshape_4d(ctx0, input, in_per_group, n_groups, T, 1);
    // Permute to mul_mat layout (k=in_per_group, m=T, batch=n_groups):
    //   old 0 (k=in_per_group) → new 0
    //   old 1 (g=n_groups)     → new 2
    //   old 2 (T)              → new 1
    //   old 3 (1)              → new 3
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3)); // ne=[k, T, g, 1]

    // Weight GGUF ne=[out_per_group, in_per_group, n_groups, 1].
    // For mul_mat we need k=in_per_group at ne[0] and n=out_per_group at ne[1].
    ggml_tensor* w_raw = core_gguf::try_get(wl.tensors, weight_name);
    ggml_tensor* w     = ggml_cast(ctx0, w_raw, GGML_TYPE_F32);
    //   old 0 (out_per_group) → new 1
    //   old 1 (in_per_group)  → new 0
    //   old 2 (n_groups)      → new 2
    //   old 3 (1)             → new 3
    w = ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3)); // ne=[k, n, g, 1]

    ggml_tensor* y = ggml_mul_mat(ctx0, w, x); // ne=[out_per_group, T, n_groups, 1]
    // Pack (out_per_group, n_groups) into a flat out_dim along ne[0].
    //   old 0 (out_per_group) → new 0
    //   old 1 (T)             → new 2
    //   old 2 (n_groups)      → new 1
    //   old 3 (1)             → new 3
    y = ggml_cont(ctx0, ggml_permute(ctx0, y, 0, 2, 1, 3)); // ne=[out_per_group, n_groups, T, 1]
    y = ggml_reshape_4d(ctx0, y, out_dim, T, 1, 1);
    return y;
}

// One ONNX GRU step with linear_before_reset = 1.
//
//   x_t     : (768) — already W·x_t + Wb for this frame (3 gates stacked along the 768 axis: [z, r, h])
//   h_lin   : (768) — R·h_{t-1} + Rb (same gate stacking)
//   h_prev  : (256) — h_{t-1}
//   Returns h_t : (256).
//
// linear_before_reset = 1 (DFN3) means r masks (R_h·h + Rb_h) BEFORE
// being added to (W_h·x + Wb_h). See the ONNX GRU spec.
ggml_tensor* build_gru_step_lbr1(ggml_context* ctx0, ggml_tensor* x_t, ggml_tensor* h_lin, ggml_tensor* h_prev) {
    // Slice the three 256-dim gate chunks out of the 768-dim stacks.
    const size_t H = 256;
    const size_t bf = sizeof(float);
    ggml_tensor* z_x = ggml_view_1d(ctx0, x_t, H, 0);
    ggml_tensor* r_x = ggml_view_1d(ctx0, x_t, H, H * bf);
    ggml_tensor* n_x = ggml_view_1d(ctx0, x_t, H, 2 * H * bf);
    ggml_tensor* z_h = ggml_view_1d(ctx0, h_lin, H, 0);
    ggml_tensor* r_h = ggml_view_1d(ctx0, h_lin, H, H * bf);
    ggml_tensor* n_h = ggml_view_1d(ctx0, h_lin, H, 2 * H * bf);

    ggml_tensor* z = ggml_sigmoid(ctx0, ggml_add(ctx0, z_x, z_h));
    ggml_tensor* r = ggml_sigmoid(ctx0, ggml_add(ctx0, r_x, r_h));
    // LBR=1: n = tanh(W_h·x + Wb_h + r · (R_h·h + Rb_h)) = tanh(n_x + r · n_h)
    ggml_tensor* n = ggml_tanh(ctx0, ggml_add(ctx0, n_x, ggml_mul(ctx0, r, n_h)));
    // h_t = (1 - z) · n + z · h_prev  =  n + z · (h_prev - n)
    ggml_tensor* delta = ggml_sub(ctx0, h_prev, n);
    ggml_tensor* h_new = ggml_add(ctx0, n, ggml_mul(ctx0, z, delta));
    return h_new;
}

// Drive an LBR=1 GRU over a length-T input sequence. Inputs and
// output have the same time-major layout (`[hidden, T, 1, 1]` in
// GGML coords). The hidden state is unrolled into T graph ops; the
// caller supplies its initial value (typically zero for batch
// validation, persisted on `dfn_stream` for streaming inference).
//
// Tensor naming convention follows the converter's:
//   <prefix>.W : (1, 3·hidden, input_size)  → GGUF ne=[input_size, 3·H, 1, 1]
//   <prefix>.R : (1, 3·hidden, hidden)      → GGUF ne=[hidden, 3·H, 1, 1]
//   <prefix>.B : (1, 6·hidden)              → GGUF ne=[6·H, 1, 1, 1]
ggml_tensor* build_gru_sequence(ggml_context* ctx0, const core_gguf::WeightLoad& wl, const char* prefix,
                                 ggml_tensor* input_seq, ggml_tensor* h_init, int hidden_size) {
    const int    T = (int)input_seq->ne[1];
    const int    H = hidden_size;
    const size_t bf = sizeof(float);

    auto cast = [&](const char* suffix) {
        std::string name = std::string(prefix) + "." + suffix;
        return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, name.c_str()), GGML_TYPE_F32);
    };
    const int input_size = (int)input_seq->ne[0];
    ggml_tensor* W = ggml_reshape_2d(ctx0, cast("W"), input_size, 3 * H);
    ggml_tensor* R = ggml_reshape_2d(ctx0, cast("R"), H, 3 * H);
    ggml_tensor* B = cast("B");
    ggml_tensor* Wb = ggml_view_1d(ctx0, B, 3 * H, 0);
    ggml_tensor* Rb = ggml_view_1d(ctx0, B, 3 * H, (size_t)(3 * H) * bf);

    // Precompute W·X + Wb for the whole T-frame batch in one matmul.
    ggml_tensor* x_lin_all = ggml_add(ctx0, ggml_mul_mat(ctx0, W, input_seq), Wb);

    ggml_tensor* h_curr = h_init;
    std::vector<ggml_tensor*> h_seq(T, nullptr);
    for (int t = 0; t < T; ++t) {
        ggml_tensor* x_t   = ggml_view_1d(ctx0, x_lin_all, 3 * H, (size_t)t * 3 * H * bf);
        ggml_tensor* h_lin = ggml_add(ctx0, ggml_mul_mat(ctx0, R, h_curr), Rb);
        ggml_tensor* h_new = build_gru_step_lbr1(ctx0, x_t, h_lin, h_curr);
        h_seq[t] = h_new;
        h_curr   = h_new;
    }
    ggml_tensor* out = ggml_reshape_4d(ctx0, h_seq[0], H, 1, 1, 1);
    for (int t = 1; t < T; ++t) {
        ggml_tensor* slice = ggml_reshape_4d(ctx0, h_seq[t], H, 1, 1, 1);
        out                = ggml_concat(ctx0, out, slice, /*dim=*/1);
    }
    // CRITICAL streaming fix: for T=1 (streaming graph), `out` is just a
    // reshape view of h_new — no copy. ggml_backend_sched can then reuse
    // h_new's underlying memory for downstream nodes that consume `out`
    // (the next GRU's mul_mat, linear_out, etc.), and our subsequent
    // `ggml_backend_tensor_get(out, ...)` reads garbage. Force a
    // ggml_cont so the saved hidden state has its own memory that the
    // sched will keep alive because we mark it as a graph output.
    // T=N path is already safe — the concat above always produces a
    // fresh contiguous tensor.
    if (T == 1) out = ggml_cont(ctx0, out);
    return out; // ne=[H, T, 1, 1]
}

// Build the encoder's SqueezedGRU on top of `combined`:
//   combined → linear_in → Relu → GRU(hidden=256) → linear_out → Relu → emb
//
// The GRU is unrolled T times in the graph; `h_init` (typically zero
// for batch validation) seeds the recurrence. Returns both pre_gru
// (post linear_in + Relu, == golden `pre_gru`) and emb (== golden `emb`).
enc_gru_io build_enc_emb_gru(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* combined,
                              ggml_tensor* h_init) {
    // ── linear_in: GroupedLinear(16 groups, 32→16) + Relu ─────────
    ggml_tensor* x = build_grouped_linear(ctx0, wl, "dfn.enc.emb_gru.linear_in.0.weight", combined,
                                          /*n_groups=*/16, /*in_per_group=*/32, /*out_per_group=*/16);
    x = ggml_relu(ctx0, x);
    ggml_tensor* pre_gru = x;
    ggml_set_name(pre_gru, "pre_gru");

    // ── GRU(hidden=256, linear_before_reset=1) ────────────────────
    ggml_tensor* h_seq = build_gru_sequence(ctx0, wl, "dfn.enc.emb_gru.gru", pre_gru, h_init, /*hidden=*/256);
    ggml_set_name(h_seq, "enc_emb_gru_h");

    // ── linear_out: GroupedLinear(16 groups, 16→32) + Relu ────────
    ggml_tensor* y = build_grouped_linear(ctx0, wl, "dfn.enc.emb_gru.linear_out.0.weight", h_seq,
                                          /*n_groups=*/16, /*in_per_group=*/16, /*out_per_group=*/32);
    y = ggml_relu(ctx0, y);
    ggml_tensor* emb = y;
    ggml_set_name(emb, "emb");

    return {pre_gru, emb, h_seq};
}

// erb_dec's SqueezedGRU. Identical shape to the encoder's except it
// stacks TWO LBR=1 GRUs (`gru` then `gru_1`) between linear_in and
// linear_out. Each carries its own persistent hidden state on
// `dfn_stream` for streaming use.
//
//   emb     : (1, T, 512) — encoder output, the only data input
//   h_init  : (256)       — h_0 for the first stacked GRU
//   h_init2 : (256)       — h_0 for the second
// Returns the post linear_out + Relu tensor matching
// `dfn.golden.erb_dec.emb_gru_out` (1, T, 512).
// Reshape the erb_dec SqueezedGRU output (ne=[512, T, 1, 1]) into the
// (1, 64, T, 8) channel-first layout the decoder's conv backbone
// consumes. PyTorch does this as `reshape(1, T, 8, 64) → transpose
// perm=[0, 3, 1, 2]` (verified via onnxruntime: 512 = F=8 × C=64 with
// C the innermost numpy axis).
ggml_tensor* build_emb_gru_reshape_to_feat(ggml_context* ctx0, ggml_tensor* emb_gru_out) {
    const int T = (int)emb_gru_out->ne[1];
    // ne=[512, T, 1, 1] → ne=[64, 8, T, 1] (innermost C=64, then F=8, then T)
    ggml_tensor* x = ggml_reshape_4d(ctx0, emb_gru_out, 64, 8, T, 1);
    // PyTorch transpose perm=[0,3,1,2] in GGML coords: send old GGML
    // axes (0=C, 1=F, 2=T, 3=batch) to new GGML positions (2, 0, 1, 3).
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 2, 0, 1, 3)); // ne=[8, T, 64, 1]
    return x;
}

// Depthwise 1×1 conv + bias + Relu — the `convNp` skip layers on
// e0..e3. Weight shape (numpy) is (64, 1, 1, 1); a 1×1 depthwise conv
// is mathematically a per-channel scale, but staying on
// `ggml_conv_2d_dw` keeps the layout uniform with the rest of the
// stack and avoids a separate broadcasting code path.
ggml_tensor* build_skip_pointwise_dw(ggml_context* ctx0, const core_gguf::WeightLoad& wl, const char* w_name,
                                       const char* b_name, ggml_tensor* x) {
    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };
    auto B = [&](const char* n) {
        return ggml_reshape_4d(ctx0, ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32), 1, 1, 64, 1);
    };
    ggml_tensor* y = ggml_conv_2d_dw(ctx0, W(w_name), x, 1, 1, 0, 0, 1, 1);
    y              = ggml_add(ctx0, y, B(b_name));
    y              = ggml_relu(ctx0, y);
    return y;
}

// Depthwise ConvTranspose specialised to DFN's (stride=2, kernel=3,
// pad=1, output_padding=1) decoder convs. GGML has no native
// depthwise transpose conv so we decompose by hand into:
//
//   out[2·i]   = w[1] · x[i]                          (i ∈ [0, L))
//   out[2·i+1] = w[2] · x[i] + w[0] · x[i+1]          (i ∈ [0, L-1))
//   out[2·L-1] = w[2] · x[L-1]                        (output_padding tail)
//
// Implementation: precompute three per-channel scaling tensors
// (mx0 = x·w[0], mx1 = x·w[1], mx2 = x·w[2]), then interleave
// `mx1` (the even contributions) with `mx2 + left-shifted-mx0`
// (the odd contributions) along the frequency axis.
//
// Input  ne=[L,   T, C, 1]    (C must be 64 for DFN)
// Weight ne=[3,   1, 1, C]    (KW × KH × IC=1 × OC=C)
// Output ne=[2·L, T, C, 1]
ggml_tensor* build_depthwise_conv_transpose_s2k3(ggml_context* ctx0, ggml_tensor* weight, ggml_tensor* x) {
    const int L = (int)x->ne[0];
    const int T = (int)x->ne[1];
    const int C = (int)x->ne[2];

    ggml_tensor* w_f32 = ggml_cast(ctx0, weight, GGML_TYPE_F32);
    // w_f32 ne=[3, 1, 1, C]. View one kernel tap per channel: 64 values
    // along ne[3], strided by the original (3 floats × 4 bytes = 12 byte)
    // channel stride. ggml_cont materialises a packed 64-element vector;
    // we then reshape it to (1, 1, C, 1) so ggml_mul broadcasts across
    // x's (L, T) freq + time axes.
    auto extract_tap = [&](size_t tap_idx) {
        ggml_tensor* v = ggml_view_4d(ctx0, w_f32, 1, 1, 1, C, w_f32->nb[1], w_f32->nb[2], w_f32->nb[3],
                                       tap_idx * w_f32->nb[0]);
        v              = ggml_cont(ctx0, v);
        return ggml_reshape_4d(ctx0, v, 1, 1, C, 1);
    };
    ggml_tensor* w0 = extract_tap(0);
    ggml_tensor* w1 = extract_tap(1);
    ggml_tensor* w2 = extract_tap(2);

    // Per-tap scaled inputs. Each is ne=[L, T, C, 1].
    ggml_tensor* mx0 = ggml_mul(ctx0, x, w0);
    ggml_tensor* mx1 = ggml_mul(ctx0, x, w1);
    ggml_tensor* mx2 = ggml_mul(ctx0, x, w2);

    // mx0 left-shifted: take mx0[1:L] then append a single zero plane.
    // Both the shift view and the zero-source view inherit mx0's
    // (non-contiguous along ne[0]) strides; ggml_scale and ggml_concat
    // need contiguous inputs, so we materialise each with ggml_cont.
    ggml_tensor* mx0_shift_part = ggml_view_4d(ctx0, mx0, L - 1, T, C, 1, mx0->nb[1], mx0->nb[2], mx0->nb[3], mx0->nb[0]);
    mx0_shift_part              = ggml_cont(ctx0, mx0_shift_part);
    ggml_tensor* mx0_zero_one   = ggml_view_4d(ctx0, mx0, 1, T, C, 1, mx0->nb[1], mx0->nb[2], mx0->nb[3], 0);
    mx0_zero_one                = ggml_cont(ctx0, mx0_zero_one);
    mx0_zero_one                = ggml_scale(ctx0, mx0_zero_one, 0.0f);
    ggml_tensor* mx0_shifted    = ggml_concat(ctx0, mx0_shift_part, mx0_zero_one, /*dim=*/0); // ne=[L, T, C, 1]

    // odd[i] = mx2[i] + mx0_shifted[i] = w[2]·x[i] + w[0]·x[i+1] (with 0 tail at i=L-1)
    ggml_tensor* odd  = ggml_add(ctx0, mx2, mx0_shifted);
    ggml_tensor* even = mx1;

    // Interleave even and odd along the freq axis: stack as ne=[2, L,
    // T, C] (even at i0=0, odd at i0=1) then reshape to ne=[2L, T, C, 1]
    // — the resulting innermost axis walks (even[0], odd[0], even[1],
    // odd[1], …, even[L-1], odd[L-1]).
    ggml_tensor* even_4d = ggml_reshape_4d(ctx0, even, 1, L, T, C);
    ggml_tensor* odd_4d  = ggml_reshape_4d(ctx0, odd,  1, L, T, C);
    ggml_tensor* stacked = ggml_concat(ctx0, even_4d, odd_4d, /*dim=*/0); // ne=[2, L, T, C]
    return ggml_reshape_4d(ctx0, stacked, 2 * L, T, C, 1);
}

// Build the erb_dec backbone from the SqueezedGRU output up through
// `convt3` (no upsampling). The piece that adds ConvTranspose for
// convt2/convt1 lands in the next slice once the depthwise-transpose
// decomposition is in.
ggml_tensor* build_erb_dec_up_to_convt3(ggml_context* ctx0, const core_gguf::WeightLoad& wl,
                                          ggml_tensor* emb_gru_out, ggml_tensor* e3) {
    // Reshape GRU output to (1, 64, T, 8) layout.
    ggml_tensor* gru_feat = build_emb_gru_reshape_to_feat(ctx0, emb_gru_out);

    // Skip stream from e3.
    ggml_tensor* e3p = build_skip_pointwise_dw(ctx0, wl, "dfn.erb_dec.onnx::Conv_288",
                                                "dfn.erb_dec.onnx::Conv_289", e3);
    // Add: GRU output + e3-skip.
    ggml_tensor* x = ggml_add(ctx0, e3p, gru_feat);

    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };
    auto B = [&](const char* n) {
        return ggml_reshape_4d(ctx0, ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32), 1, 1, 64, 1);
    };

    // convt3: depthwise 1×3 (no stride, freq pad 1) → pointwise 1×1 → Relu.
    x = ggml_conv_2d_dw(ctx0, W("dfn.erb_dec.convt3.0.weight"), x, 1, 1, 1, 0, 1, 1);
    x = ggml_conv_2d(ctx0, W("dfn.erb_dec.onnx::Conv_291"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.erb_dec.onnx::Conv_292"));
    x = ggml_relu(ctx0, x);
    return x; // ne=[8, T, 64, 1] — matches dfn.golden.erb_dec.convt3_out layout
}

// Continue the erb_dec backbone through convt2 — the first
// ConvTranspose stage (depthwise stride-2 upsample F: 8 → 16). Input
// is `convt3_out` produced by `build_erb_dec_up_to_convt3`.
ggml_tensor* build_erb_dec_up_to_convt2(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* convt3_out,
                                         ggml_tensor* e2) {
    // Skip stream from e2 + Add.
    ggml_tensor* e2p = build_skip_pointwise_dw(ctx0, wl, "dfn.erb_dec.onnx::Conv_294",
                                                "dfn.erb_dec.onnx::Conv_295", e2);
    ggml_tensor* x   = ggml_add(ctx0, e2p, convt3_out); // ne=[8, T, 64, 1]

    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };
    auto B = [&](const char* n) {
        return ggml_reshape_4d(ctx0, ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32), 1, 1, 64, 1);
    };

    // convt2 ConvTranspose: depthwise stride-2 kernel-3 pad-1 op-1
    // → upsample F: 8 → 16. Hand-decomposed (see helper above).
    ggml_tensor* w_dw = core_gguf::try_get(wl.tensors, "dfn.erb_dec.convt2.0.weight");
    x                 = build_depthwise_conv_transpose_s2k3(ctx0, w_dw, x);
    // Pointwise 1×1 (group=1, 64→64) + bias + Relu.
    x = ggml_conv_2d(ctx0, W("dfn.erb_dec.onnx::Conv_297"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.erb_dec.onnx::Conv_298"));
    x = ggml_relu(ctx0, x);
    return x; // ne=[16, T, 64, 1] — matches dfn.golden.erb_dec.convt2_out
}

// Continue through convt1 — the second ConvTranspose stage (depthwise
// stride-2 upsample F: 16 → 32).
ggml_tensor* build_erb_dec_up_to_convt1(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* convt2_out,
                                         ggml_tensor* e1) {
    ggml_tensor* e1p = build_skip_pointwise_dw(ctx0, wl, "dfn.erb_dec.onnx::Conv_300",
                                                "dfn.erb_dec.onnx::Conv_301", e1);
    ggml_tensor* x   = ggml_add(ctx0, e1p, convt2_out);

    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };
    auto B = [&](const char* n) {
        return ggml_reshape_4d(ctx0, ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32), 1, 1, 64, 1);
    };

    ggml_tensor* w_dw = core_gguf::try_get(wl.tensors, "dfn.erb_dec.convt1.0.weight");
    x                 = build_depthwise_conv_transpose_s2k3(ctx0, w_dw, x);
    x = ggml_conv_2d(ctx0, W("dfn.erb_dec.onnx::Conv_303"), x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(ctx0, x, B("dfn.erb_dec.onnx::Conv_304"));
    x = ggml_relu(ctx0, x);
    return x; // ne=[32, T, 64, 1] — matches dfn.golden.erb_dec.convt1_out
}

struct df_dec_partial_io {
    ggml_tensor* skip_add    = nullptr; // (256, T, 1, 1) — gru + df_skip combine
    ggml_tensor* alpha       = nullptr; // (1,   T, 1, 1) — sigmoid α gate
    ggml_tensor* df_out_tanh = nullptr; // (960, T, 1, 1) — pre-reshape (1, T, 96, 10)
    ggml_tensor* h1_seq      = nullptr; // final hidden state of df_gru #1 (for streaming)
    ggml_tensor* h2_seq      = nullptr; // final hidden state of df_gru #2
};

struct df_dec_full_io {
    ggml_tensor* coefs = nullptr;       // (10, 96, T, 1) — final DF complex coefs
    ggml_tensor* alpha = nullptr;       // (1,  T,  1, 1) — α gate
};

// Build df_convp on c0 (the encoder's first DF feature output): a
// causal time-axis 5×1 conv with group=2, followed by a 1×1
// pointwise conv + bias + Relu. Output is (1, 10, T, 96) in PyTorch
// order, i.e. GGML ne=[96, T, 10, 1].
//
//   c0     : ne=[96, T, 64, 1] — encoder df-stream output, GGML layout
//   weight : df_convp.1.weight (10, 32, 5, 1) → GGUF ne=[1, 5, 32, 10]
//
// group=2 means each of the two input-channel halves (0..31 and
// 32..63) feeds 5 disjoint output channels. We split kernel + input
// along the channel axis, run two sub-convs, and concat — same
// pattern the encoder's df_conv0 uses.
ggml_tensor* build_df_convp(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* c0,
                              ggml_tensor* c0_history = nullptr) {
    const int T = (int)c0->ne[1];

    // ── Causal time-pad: prepend 4 frames of context along the time
    // axis. Three sources, in priority order:
    //
    //   1. `c0_history` (streaming): real c0[t-4..t-1] from the prior
    //      frames — gives df_convp the same 5-tap context the offline
    //      model trained against.
    //   2. Zero-pad view of c0[0..3] (T ≥ 4, e.g. batch golden runs).
    //   3. Zero-pad assembled from 4 single-frame zeros (T < 4 without
    //      history, only happens during the very first streaming
    //      frame before c0_history is wired in).
    ggml_tensor* pad_view;
    if (c0_history) {
        pad_view = c0_history; // ne=[96, 4, 64, 1] — caller-owned
    } else if (c0->ne[1] >= 4) {
        pad_view = ggml_view_4d(ctx0, c0, c0->ne[0], 4, c0->ne[2], c0->ne[3], c0->nb[1], c0->nb[2], c0->nb[3], 0);
        pad_view = ggml_cont(ctx0, pad_view);
        pad_view = ggml_scale(ctx0, pad_view, 0.0f);
    } else {
        ggml_tensor* one = ggml_view_4d(ctx0, c0, c0->ne[0], 1, c0->ne[2], c0->ne[3], c0->nb[1], c0->nb[2], c0->nb[3], 0);
        one              = ggml_cont(ctx0, one);
        one              = ggml_scale(ctx0, one, 0.0f);
        pad_view         = one;
        for (int i = 1; i < 4; ++i)
            pad_view = ggml_concat(ctx0, pad_view, one, /*dim=*/1);
    }
    ggml_tensor* c0_pad = ggml_concat(ctx0, pad_view, c0, /*dim=*/1); // ne=[96, T+4, 64, 1]

    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };

    // ── Group=2 split of the 5×1 conv kernel + input. Weight GGUF
    //    ne=[1, 5, 32, 10]: split along ne[3] (OC) into two ne=[1, 5, 32, 5].
    ggml_tensor* w_full = W("dfn.df_dec.df_convp.1.weight");
    ggml_tensor* w0 = ggml_cont(ctx0, ggml_view_4d(ctx0, w_full, 1, 5, 32, 5, w_full->nb[1], w_full->nb[2],
                                                    w_full->nb[3], 0));
    ggml_tensor* w1 = ggml_cont(ctx0, ggml_view_4d(ctx0, w_full, 1, 5, 32, 5, w_full->nb[1], w_full->nb[2],
                                                    w_full->nb[3], 5 * w_full->nb[3]));
    // Input split along channel axis (ne[2]) into two ne=[96, T+4, 32, 1].
    ggml_tensor* in0 = ggml_cont(ctx0, ggml_view_4d(ctx0, c0_pad, c0_pad->ne[0], c0_pad->ne[1], 32, 1, c0_pad->nb[1],
                                                     c0_pad->nb[2], c0_pad->nb[3], 0));
    ggml_tensor* in1 = ggml_cont(ctx0, ggml_view_4d(ctx0, c0_pad, c0_pad->ne[0], c0_pad->ne[1], 32, 1, c0_pad->nb[1],
                                                     c0_pad->nb[2], c0_pad->nb[3], 32 * c0_pad->nb[2]));

    // Sub-convs: KW=1 KH=5 IC=32 OC=5, stride 1×1, no pad (already done).
    ggml_tensor* out0 = ggml_conv_2d(ctx0, w0, in0, 1, 1, 0, 0, 1, 1); // ne=[96, T, 5, 1]
    ggml_tensor* out1 = ggml_conv_2d(ctx0, w1, in1, 1, 1, 0, 0, 1, 1);
    ggml_tensor* x    = ggml_concat(ctx0, out0, out1, /*dim=*/2);     // ne=[96, T, 10, 1]

    // Pointwise 1×1 (group=1, 10→10) + bias + Relu.
    x = ggml_conv_2d(ctx0, W("dfn.df_dec.onnx::Conv_268"), x, 1, 1, 0, 0, 1, 1);
    ggml_tensor* b_pw = ggml_cast(ctx0, core_gguf::try_get(wl.tensors, "dfn.df_dec.onnx::Conv_269"), GGML_TYPE_F32);
    x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, b_pw, 1, 1, 10, 1));
    x = ggml_relu(ctx0, x);
    return x; // ne=[96, T, 10, 1] — PyTorch (1, 10, T, 96)
}

// Forward decl — body is below `build_df_dec_no_convp`.
df_dec_partial_io build_df_dec_no_convp(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* emb,
                                          ggml_tensor* h1, ggml_tensor* h2);

// Build the entire df_dec graph: combine the no-convp path (with its
// Tanh output) and df_convp into the final (coefs, α) outputs.
//
// PyTorch flow:
//   coefs = Add( Reshape(df_out_tanh, (1, T, 96, 10)),
//                Transpose(perm=[0,2,3,1], df_convp_relu_out) )
df_dec_full_io build_df_dec(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* emb, ggml_tensor* c0,
                              ggml_tensor* h1, ggml_tensor* h2, ggml_tensor* c0_history = nullptr) {
    df_dec_partial_io part = build_df_dec_no_convp(ctx0, wl, emb, h1, h2);
    const int          T   = (int)part.df_out_tanh->ne[1];

    // df_convp branch on c0 → (1, 10, T, 96) in PyTorch coords.
    ggml_tensor* convp = build_df_convp(ctx0, wl, c0, c0_history); // ne=[96, T, 10, 1]

    // PyTorch Transpose perm=[0,2,3,1]: (1, 10, T, 96) → (1, T, 96, 10).
    // Old PyT axes (0=batch, 1=C, 2=T, 3=F) ↔ old GGML (3, 2, 1, 0).
    // New PyT shape (1, T, 96, 10): old PyT axis 2 → new pos 1, old PyT
    // axis 3 → new pos 2, old PyT axis 1 → new pos 3. In GGML coords:
    //   old GGML 0 (F=96)  → new GGML 1 ⇒ axis0=1
    //   old GGML 1 (T)     → new GGML 2 ⇒ axis1=2
    //   old GGML 2 (C=10)  → new GGML 0 ⇒ axis2=0
    //   old GGML 3 (batch) → new GGML 3 ⇒ axis3=3
    ggml_tensor* convp_t = ggml_cont(ctx0, ggml_permute(ctx0, convp, 1, 2, 0, 3)); // ne=[10, 96, T, 1]

    // Reshape df_out_tanh to match: ne=[960, T, 1, 1] → ne=[10, 96, T, 1].
    // Memory is identical (innermost 10 unchanged, next axis groups 96).
    ggml_tensor* tanh_resh = ggml_reshape_4d(ctx0, part.df_out_tanh, 10, 96, T, 1);

    ggml_tensor* coefs = ggml_add(ctx0, tanh_resh, convp_t);
    ggml_set_name(coefs, "coefs");
    return {coefs, part.alpha};
}

// df_dec sub-graph excluding the df_convp branch — covers df_gru
// (linear_in + two stacked LBR=1 GRUs, no linear_out), df_skip
// (GroupedLinear) Add, α head (Dense + Sigmoid) and df_out
// (GroupedLinear + Tanh). The df_convp+Add tail is the last
// remaining piece and is added in a separate slice once group=2 conv
// with time-axis causal pad is wired up.
df_dec_partial_io build_df_dec_no_convp(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* emb,
                                          ggml_tensor* h1, ggml_tensor* h2) {
    // ── df_gru SqueezedGRU — note: NO linear_out (the gru output is
    //    consumed directly by df_skip Add). ────────────────────────
    ggml_tensor* x = build_grouped_linear(ctx0, wl, "dfn.df_dec.df_gru.linear_in.0.weight", emb,
                                          /*n_groups=*/8, /*in_per_group=*/64, /*out_per_group=*/32);
    x = ggml_relu(ctx0, x);
    ggml_tensor* h1_seq = build_gru_sequence(ctx0, wl, "dfn.df_dec.df_gru.gru.gru",   x,      h1, /*hidden=*/256);
    ggml_tensor* h2_seq = build_gru_sequence(ctx0, wl, "dfn.df_dec.df_gru.gru.gru_1", h1_seq, h2, /*hidden=*/256);
    ggml_set_name(h1_seq, "df_dec_gru_h1");
    ggml_set_name(h2_seq, "df_dec_gru_h2");
    // h2_seq ne=[256, T, 1, 1].

    // ── df_skip (GroupedLinear 16 groups, 32→16) — no activation. ─
    ggml_tensor* skip = build_grouped_linear(ctx0, wl, "dfn.df_dec.df_skip.weight", emb,
                                              /*n_groups=*/16, /*in_per_group=*/32, /*out_per_group=*/16);

    // ── Add ─────────────────────────────────────────────────────────
    ggml_tensor* skip_add = ggml_add(ctx0, h2_seq, skip); // ne=[256, T, 1, 1]
    ggml_set_name(skip_add, "skip_add");

    // ── α head: (Add output) × MatMul(256, 1) + bias → Sigmoid ────
    // MatMul_321 ne=[1, 256, 1, 1] (numpy (256, 1)). For mul_mat we
    // need k=256 at ne[0], n=1 at ne[1]. ggml_reshape_2d does it.
    ggml_tensor* w_a_raw = core_gguf::try_get(wl.tensors, "dfn.df_dec.onnx::MatMul_321");
    ggml_tensor* w_a     = ggml_reshape_2d(ctx0, ggml_cast(ctx0, w_a_raw, GGML_TYPE_F32), 256, 1);
    ggml_tensor* alpha   = ggml_mul_mat(ctx0, w_a, skip_add); // ne=[1, T, 1, 1]
    ggml_tensor* b_a     = ggml_cast(ctx0, core_gguf::try_get(wl.tensors, "dfn.df_dec.df_fc_a.0.bias"), GGML_TYPE_F32);
    alpha                = ggml_add(ctx0, alpha, ggml_reshape_4d(ctx0, b_a, 1, 1, 1, 1));
    alpha                = ggml_sigmoid(ctx0, alpha);
    ggml_set_name(alpha, "alpha");

    // ── df_out (GroupedLinear 16 groups, 16→60) + Tanh ────────────
    ggml_tensor* y = build_grouped_linear(ctx0, wl, "dfn.df_dec.df_out.0.weight", skip_add,
                                           /*n_groups=*/16, /*in_per_group=*/16, /*out_per_group=*/60);
    y = ggml_tanh(ctx0, y); // ne=[960, T, 1, 1]
    ggml_set_name(y, "df_out_tanh");

    return {skip_add, alpha, y, h1_seq, h2_seq};
}

// Final erb_dec stage: e0-skip Add + conv0_out (1×3 conv that
// collapses 64 channels → 1) + Sigmoid → m (the ERB gain mask).
ggml_tensor* build_erb_dec_m(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* convt1_out,
                              ggml_tensor* e0) {
    ggml_tensor* e0p = build_skip_pointwise_dw(ctx0, wl, "dfn.erb_dec.onnx::Conv_306",
                                                "dfn.erb_dec.onnx::Conv_307", e0);
    ggml_tensor* x   = ggml_add(ctx0, e0p, convt1_out);

    auto W = [&](const char* n) { return ggml_cast(ctx0, core_gguf::try_get(wl.tensors, n), GGML_TYPE_F32); };

    // conv0_out: 1×3 conv (group=1), 64 input channels → 1 output channel.
    // Weight ne=[3, 1, 64, 1] (KW=3, KH=1, IC=64, OC=1). Bias is a scalar.
    x = ggml_conv_2d(ctx0, W("dfn.erb_dec.onnx::Conv_309"), x, 1, 1, 1, 0, 1, 1);
    // Bias is a 1-element tensor; broadcasts trivially.
    ggml_tensor* bias = ggml_cast(ctx0, core_gguf::try_get(wl.tensors, "dfn.erb_dec.onnx::Conv_310"), GGML_TYPE_F32);
    x = ggml_add(ctx0, x, ggml_reshape_4d(ctx0, bias, 1, 1, 1, 1));
    x = ggml_sigmoid(ctx0, x);
    return x; // ne=[32, T, 1, 1] — matches dfn.golden.erb_dec.m (1, 1, T, 32)
}

struct erb_dec_gru_io {
    ggml_tensor* emb_gru_out = nullptr; // (512, T, 1, 1) — post linear_out + Relu
    ggml_tensor* h1_seq      = nullptr; // (256, T, 1, 1) — final hidden state of GRU#1 for T=1
    ggml_tensor* h2_seq      = nullptr; // (256, T, 1, 1) — final hidden state of GRU#2 for T=1
};

erb_dec_gru_io build_erb_dec_emb_gru(ggml_context* ctx0, const core_gguf::WeightLoad& wl, ggml_tensor* emb,
                                      ggml_tensor* h_init, ggml_tensor* h_init2) {
    ggml_tensor* x = build_grouped_linear(ctx0, wl, "dfn.erb_dec.emb_gru.linear_in.0.weight", emb,
                                          /*n_groups=*/16, /*in_per_group=*/32, /*out_per_group=*/16);
    x = ggml_relu(ctx0, x);

    ggml_tensor* h1_seq = build_gru_sequence(ctx0, wl, "dfn.erb_dec.emb_gru.gru",   x,      h_init,  /*hidden=*/256);
    ggml_tensor* h2_seq = build_gru_sequence(ctx0, wl, "dfn.erb_dec.emb_gru.gru_1", h1_seq, h_init2, /*hidden=*/256);
    ggml_set_name(h1_seq, "erb_dec_emb_gru_h1");
    ggml_set_name(h2_seq, "erb_dec_emb_gru_h2");

    ggml_tensor* y = build_grouped_linear(ctx0, wl, "dfn.erb_dec.emb_gru.linear_out.0.weight", h2_seq,
                                          /*n_groups=*/16, /*in_per_group=*/16, /*out_per_group=*/32);
    y = ggml_relu(ctx0, y);
    return {y, h1_seq, h2_seq};
}

} // namespace

// Common bring-up: fresh ctx0, padded feat_erb input, ERB-branch graph
// built, backend buffer allocated. Caller adds outputs to the graph
// and computes.
namespace {

struct enc_run_ctx {
    ggml_context* ctx0              = nullptr;
    ggml_tensor*  feat_erb_padded   = nullptr; // (32, T+2, 1, 1)
    ggml_tensor*  feat_spec_padded  = nullptr; // (96, T+2, 2, 1)
};

enc_run_ctx begin_encoder_graph(const dfn_model* m, int T, bool want_spec) {
    // Scale context memory with T — the GRU loops add O(T) operator
    // nodes, and the unrolled chain at T ~= 64 needs more than the
    // 16 MB the streaming graph (T=1) lives in.
    const size_t kGraphMemBytes = (T > 1) ? (size_t)128 * 1024 * 1024 : (size_t)16 * 1024 * 1024;
    ggml_init_params gp{ kGraphMemBytes, /*mem_buffer=*/nullptr, /*no_alloc=*/true };
    enc_run_ctx r;
    r.ctx0 = ggml_init(gp);
    if (!r.ctx0) return r;
    r.feat_erb_padded = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 32, T + 2, 1, 1);
    ggml_set_name(r.feat_erb_padded, "feat_erb_padded");
    ggml_set_input(r.feat_erb_padded);
    if (want_spec) {
        r.feat_spec_padded = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 96, T + 2, 2, 1);
        ggml_set_name(r.feat_spec_padded, "feat_spec_padded");
        ggml_set_input(r.feat_spec_padded);
    }
    (void)m;
    return r;
}

// Stage host-side padded input into a ggml_tensor.
// PAD time frames of zeros prepended on the time axis.
void stage_padded(ggml_tensor* dst, const float* src, int n_chan, int T, int F) {
    constexpr int PAD = 2;
    // Memory layout per channel: F floats per frame, (T+PAD) frames.
    // GGML ne=[F, T+PAD, C, 1] with C the channel axis. C-contiguous,
    // matches numpy (1, C, T, F) after the host-side pad.
    const size_t per_chan = (size_t)F * (T + PAD);
    std::vector<float> padded((size_t)n_chan * per_chan, 0.0f);
    for (int c = 0; c < n_chan; ++c) {
        float*       dst_ch = padded.data() + (size_t)c * per_chan + (size_t)F * PAD;
        const float* src_ch = src + (size_t)c * F * T;
        std::memcpy(dst_ch, src_ch, (size_t)F * T * sizeof(float));
    }
    ggml_backend_tensor_set(dst, padded.data(), 0, padded.size() * sizeof(float));
}

void stage_feat_erb_padded(ggml_tensor* dst, const float* feat_erb, int T) {
    stage_padded(dst, feat_erb, 1, T, 32);
}
void stage_feat_spec_padded(ggml_tensor* dst, const float* feat_spec, int T) {
    stage_padded(dst, feat_spec, 2, T, 96);
}

} // namespace

// Map a frequency bin index to the ERB band it belongs to. Linear
// scan over the per-band (index, width) table — n_erb is 32, so cost
// per call is bounded; for production use a precomputed
// `freq_to_band` table would be cheaper, but at one hundred
// per-second frame rate the linear scan is negligible.
static int erb_band_of(const dfn_model* m, int freq) {
    for (int b = 0; b < (int)m->erb_widths.size(); ++b) {
        int start = m->erb_indices[b];
        int end   = start + m->erb_widths[b];
        if (freq >= start && freq < end) return b;
    }
    return (int)m->erb_widths.size() - 1;
}

// ─── DFN3 neural network — one-frame streaming inference ────────────
//
// Builds a T=1 GGML graph that runs the full DFN3 pipeline (encoder
// + erb_dec + df_dec) once for the current frame and returns the
// Stage-1 ERB-gain mask `m`, Stage-2 DF complex coefs and the
// α-gate.
//
// Persistent state carried on `dfn_stream`:
//   * 5 GRU hidden buffers (one per recurrent layer)
//   * `band_mag_ema` for normalising the input complex spectrum into
//     the model's unit-magnitude `feat_spec` domain.
//
// Causal-conv warm-up: we feed zero-padded context to the encoder
// time-axis convs at every frame (the persistent GRU state carries
// long-term temporal info — the convs only see 3- or 5-tap windows).
// First few frames after `dfn_stream_reset` have small additive
// warm-up artefacts; in practice these are below the noise floor of
// a 100-frame-per-second TTS stream.
static bool predict_dfn3(const dfn_model* m, const std::vector<float>& log_erb_feat,
                          const std::vector<float>& erb_power, const float* spec_re, const float* spec_im, int n_bins,
                          dfn_stream* stream, std::vector<float>& gains, std::vector<float>& coef_re,
                          std::vector<float>& coef_im, float* alpha_out) {
    if (!m || !stream || !alpha_out) return false;
    if (!m->has_neural_weights || !m->sched) return false;
    (void)erb_power;
    constexpr int kEncFInFeat = 32;
    constexpr int kEncFInSpec = 96;
    constexpr int kEncPadT    = 2;
    constexpr int kHidden     = dfn_stream::kGruHidden;
    const int     n_erb       = m->n_erb;

    // ── 1. Per-freq-bin unit-norm of the complex spectrum ───────────
    // Matches libdf::band_unit_norm exactly, operating on the
    // wnorm-scaled spec (= spec × wnorm) since that's what the
    // training-time state and `unit_norm` init values are calibrated
    // for. See the wnorm explanation above the erb feature path.
    //   state[f] = (1-α)·|spec_wn[f]| + α·state[f]
    //   feat[f]  = spec_wn[f] / sqrt(state[f])
    const float wnorm    = 2.0f * (float)m->frame_size / ((float)m->fft_size * (float)m->fft_size);
    const float bm_alpha = kDfnEmaAlpha;
    if (!stream->unit_norm_seeded) {
        for (int f = 0; f < kEncFInSpec; ++f) {
            float t = kEncFInSpec <= 1 ? 0.0f : (float)f / (float)(kEncFInSpec - 1);
            stream->unit_norm_state[f] = kDfnUnitNormHi + t * (kDfnUnitNormLo - kDfnUnitNormHi);
        }
        stream->unit_norm_seeded = true;
    }
    std::vector<float> feat_spec_flat((size_t)2 * kEncFInSpec, 0.0f);
    for (int f = 0; f < kEncFInSpec && f < n_bins; ++f) {
        float re_wn = spec_re[f] * wnorm;
        float im_wn = spec_im[f] * wnorm;
        float mag   = std::sqrt(re_wn * re_wn + im_wn * im_wn);
        stream->unit_norm_state[f] = (1.0f - bm_alpha) * mag + bm_alpha * stream->unit_norm_state[f];
        float denom                = std::sqrt(std::max(1e-12f, stream->unit_norm_state[f]));
        feat_spec_flat[0 * kEncFInSpec + f] = re_wn / denom;
        feat_spec_flat[1 * kEncFInSpec + f] = im_wn / denom;
    }

    // ── 2. Build (or reuse) the persistent T=1 GGML graph ──────────
    if (!stream->infer_ctx) {
        static constexpr size_t kGraphMemBytes = 16 * 1024 * 1024;
        ggml_init_params gp{kGraphMemBytes, /*mem_buffer=*/nullptr, /*no_alloc=*/true};
        ggml_context*    ctx0 = ggml_init(gp);
        if (!ctx0) return false;
        const int T = 1;

        ggml_tensor* feat_erb_padded  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kEncFInFeat, T + kEncPadT, 1, 1);
        ggml_tensor* feat_spec_padded = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kEncFInSpec, T + kEncPadT, 2, 1);
        ggml_set_input(feat_erb_padded);
        ggml_set_input(feat_spec_padded);
        ggml_set_name(feat_erb_padded, "in_feat_erb");
        ggml_set_name(feat_spec_padded, "in_feat_spec");

        ggml_tensor* h_enc  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kHidden, 1, 1, 1);
        ggml_tensor* h_erb1 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kHidden, 1, 1, 1);
        ggml_tensor* h_erb2 = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kHidden, 1, 1, 1);
        ggml_tensor* h_df1  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kHidden, 1, 1, 1);
        ggml_tensor* h_df2  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kHidden, 1, 1, 1);
        for (ggml_tensor* h : {h_enc, h_erb1, h_erb2, h_df1, h_df2}) ggml_set_input(h);

        // c0 history: 4 prior frames, fed to df_convp as causal context.
        ggml_tensor* c0_hist = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, kEncFInSpec, 4, 64, 1);
        ggml_set_input(c0_hist);
        ggml_set_name(c0_hist, "in_c0_hist");

        auto erb_io  = build_enc_erb_branch(ctx0, m->weights, feat_erb_padded);
        auto df_io   = build_enc_df_branch(ctx0, m->weights, feat_spec_padded);
        auto cb      = build_enc_combine(ctx0, m->weights, df_io.df_feat, erb_io.e3);
        auto gru     = build_enc_emb_gru(ctx0, m->weights, cb.combined, h_enc);
        auto erb_gru = build_erb_dec_emb_gru(ctx0, m->weights, gru.emb, h_erb1, h_erb2);
        ggml_tensor* ct3  = build_erb_dec_up_to_convt3(ctx0, m->weights, erb_gru.emb_gru_out, erb_io.e3);
        ggml_tensor* ct2  = build_erb_dec_up_to_convt2(ctx0, m->weights, ct3, erb_io.e2);
        ggml_tensor* ct1  = build_erb_dec_up_to_convt1(ctx0, m->weights, ct2, erb_io.e1);
        ggml_tensor* mask = build_erb_dec_m(ctx0, m->weights, ct1, erb_io.e0);
        auto dfd = build_df_dec(ctx0, m->weights, gru.emb, df_io.c0, h_df1, h_df2, c0_hist);
        if (!mask || !dfd.coefs || !dfd.alpha) { ggml_free(ctx0); return false; }

        ggml_tensor* h_df1_out = ggml_get_tensor(ctx0, "df_dec_gru_h1");
        ggml_tensor* h_df2_out = ggml_get_tensor(ctx0, "df_dec_gru_h2");

        ggml_tensor* outs[] = {mask, dfd.coefs, dfd.alpha,
                                gru.h_seq, erb_gru.h1_seq, erb_gru.h2_seq, h_df1_out, h_df2_out, df_io.c0};
        for (ggml_tensor* t : outs) {
            if (t) ggml_set_output(t);
        }
        ggml_cgraph* gf = ggml_new_graph(ctx0);
        for (ggml_tensor* t : outs) {
            if (t) ggml_build_forward_expand(gf, t);
        }

        // Persist the graph topology.
        stream->infer_ctx  = ctx0;
        stream->infer_gf   = gf;
        stream->in_feat_erb  = feat_erb_padded;
        stream->in_feat_spec = feat_spec_padded;
        stream->in_h_enc     = h_enc;
        stream->in_h_erb1    = h_erb1;
        stream->in_h_erb2    = h_erb2;
        stream->in_h_df1     = h_df1;
        stream->in_h_df2     = h_df2;
        stream->in_c0_hist   = c0_hist;
        stream->out_mask     = mask;
        stream->out_coefs    = dfd.coefs;
        stream->out_alpha    = dfd.alpha;
        stream->out_h_enc    = gru.h_seq;
        stream->out_h_erb1   = erb_gru.h1_seq;
        stream->out_h_erb2   = erb_gru.h2_seq;
        stream->out_h_df1    = h_df1_out;
        stream->out_h_df2    = h_df2_out;
        stream->out_c0_current = df_io.c0;
    }

    // ── 3. Allocate per-call sched, stage inputs, run ──────────────
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, stream->infer_gf)) return false;

    // Stage feat_erb_padded as [hist_t-2, hist_t-1, current] — the
    // encoder's 3-tap time conv then sees real history instead of
    // zeros. Same for feat_spec.
    {
        const int          T = 1;
        std::vector<float> feh((size_t)kEncFInFeat * (T + kEncPadT), 0.0f);
        std::memcpy(feh.data() + (size_t)0 * kEncFInFeat, stream->feat_erb_hist.data() + 0 * kEncFInFeat,
                    (size_t)kEncFInFeat * sizeof(float));
        std::memcpy(feh.data() + (size_t)1 * kEncFInFeat, stream->feat_erb_hist.data() + 1 * kEncFInFeat,
                    (size_t)kEncFInFeat * sizeof(float));
        std::memcpy(feh.data() + (size_t)2 * kEncFInFeat, log_erb_feat.data(),
                    (size_t)kEncFInFeat * sizeof(float));
        ggml_backend_tensor_set(stream->in_feat_erb, feh.data(), 0, feh.size() * sizeof(float));

        // feat_spec layout: 2 channels (re/im) outer × 3 time frames × 96 bins inner.
        // Memory address(c, t, f) = c * (T+pad) * F + t * F + f.
        std::vector<float> fsh((size_t)2 * kEncFInSpec * (T + kEncPadT), 0.0f);
        for (int c = 0; c < 2; ++c) {
            for (int t = 0; t < 2; ++t) {
                std::memcpy(fsh.data() + ((size_t)c * (T + kEncPadT) + t) * kEncFInSpec,
                            stream->feat_spec_hist.data() + ((size_t)c * 2 + t) * kEncFInSpec,
                            (size_t)kEncFInSpec * sizeof(float));
            }
            std::memcpy(fsh.data() + ((size_t)c * (T + kEncPadT) + 2) * kEncFInSpec,
                        feat_spec_flat.data() + (size_t)c * kEncFInSpec, (size_t)kEncFInSpec * sizeof(float));
        }
        ggml_backend_tensor_set(stream->in_feat_spec, fsh.data(), 0, fsh.size() * sizeof(float));
    }

    ggml_tensor* h_in[5] = {stream->in_h_enc, stream->in_h_erb1, stream->in_h_erb2, stream->in_h_df1,
                             stream->in_h_df2};
    for (int i = 0; i < 5; ++i) {
        ggml_backend_tensor_set(h_in[i], stream->gru_hidden.data() + (size_t)i * kHidden, 0,
                                (size_t)kHidden * sizeof(float));
    }

    // c0 history (4 frames of the encoder's c0 output) — fed as the
    // causal pad for df_convp's 5-tap time conv.
    ggml_backend_tensor_set(stream->in_c0_hist, stream->c0_hist.data(), 0,
                            stream->c0_hist.size() * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, stream->infer_gf);

    // ── 4. Extract outputs and update streaming state ──────────────
    gains.assign(n_erb, 0.0f);
    ggml_backend_tensor_get(stream->out_mask, gains.data(), 0, (size_t)n_erb * sizeof(float));

    std::vector<float> coefs_flat((size_t)10 * m->df_bins, 0.0f);
    ggml_backend_tensor_get(stream->out_coefs, coefs_flat.data(), 0, coefs_flat.size() * sizeof(float));
    coef_re.assign((size_t)m->df_order * m->df_bins, 0.0f);
    coef_im.assign((size_t)m->df_order * m->df_bins, 0.0f);
    for (int d = 0; d < m->df_order; ++d) {
        for (int f = 0; f < m->df_bins; ++f) {
            coef_re[(size_t)d * m->df_bins + f] = coefs_flat[(size_t)f * 10 + 2 * d + 0];
            coef_im[(size_t)d * m->df_bins + f] = coefs_flat[(size_t)f * 10 + 2 * d + 1];
        }
    }
    ggml_backend_tensor_get(stream->out_alpha, alpha_out, 0, sizeof(float));

    ggml_tensor* h_out[5] = {stream->out_h_enc, stream->out_h_erb1, stream->out_h_erb2, stream->out_h_df1,
                              stream->out_h_df2};
    for (int i = 0; i < 5; ++i) {
        if (h_out[i]) {
            ggml_backend_tensor_get(h_out[i], stream->gru_hidden.data() + (size_t)i * kHidden, 0,
                                    (size_t)kHidden * sizeof(float));
        }
    }

    // ── 5. Slide history buffers forward by one frame ──────────────
    // feat_erb_hist[0] ← feat_erb_hist[1];   feat_erb_hist[1] ← current
    std::memcpy(stream->feat_erb_hist.data(), stream->feat_erb_hist.data() + kEncFInFeat,
                (size_t)kEncFInFeat * sizeof(float));
    std::memcpy(stream->feat_erb_hist.data() + kEncFInFeat, log_erb_feat.data(),
                (size_t)kEncFInFeat * sizeof(float));
    // feat_spec_hist layout matches the staged buffer: (channel × frame × freq).
    for (int c = 0; c < 2; ++c) {
        float*       dst    = stream->feat_spec_hist.data() + (size_t)c * 2 * kEncFInSpec;
        const float* src1   = stream->feat_spec_hist.data() + (size_t)c * 2 * kEncFInSpec + kEncFInSpec;
        const float* srcnew = feat_spec_flat.data() + (size_t)c * kEncFInSpec;
        std::memcpy(dst, src1, (size_t)kEncFInSpec * sizeof(float));
        std::memcpy(dst + kEncFInSpec, srcnew, (size_t)kEncFInSpec * sizeof(float));
    }
    // c0 history: shift left by one frame on the time axis, then push
    // the current frame's c0 into the rightmost slot.
    // Layout in the c0_hist buffer: address(c, t, f) = (c*4 + t)*96 + f.
    // c0 tensor layout (innermost ne[0]=96 freq, ne[1]=1 time, ne[2]=64 ch):
    //   tensor[c, 0, f] == hist new-frame slot for channel c.
    {
        constexpr int F = 96;
        // Shift t=1..3 → t=0..2 per channel; we'll fill t=3 from c0_current.
        for (int c = 0; c < 64; ++c) {
            float* dst = stream->c0_hist.data() + (size_t)c * 4 * F;
            std::memmove(dst, dst + F, (size_t)3 * F * sizeof(float));
        }
        // Read current frame's c0 from the graph output.
        std::vector<float> c0_curr((size_t)64 * F, 0.0f);
        ggml_backend_tensor_get(stream->out_c0_current, c0_curr.data(), 0, c0_curr.size() * sizeof(float));
        // c0_curr layout: tensor ne=[96, 1, 64, 1] contiguous — address(c, f) = c*96 + f.
        for (int c = 0; c < 64; ++c) {
            float*       slot = stream->c0_hist.data() + ((size_t)c * 4 + 3) * F;
            const float* src  = c0_curr.data() + (size_t)c * F;
            std::memcpy(slot, src, (size_t)F * sizeof(float));
        }
    }
    return true;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_encoder_e0(const struct dfn_model* m, const float* feat_erb, int T, float* out_e0, int out_e0_cap) {
    if (!m || !feat_erb || !out_e0 || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    const int n_out = 1 * 64 * T * 32;
    if (out_e0_cap < n_out) return -3;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/false);
    if (!r.ctx0) return -5;
    auto io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    if (!io.e0) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(io.e0);
    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, io.e0);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(io.e0, out_e0, 0, (size_t)n_out * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_encoder_erb_pyramid(const struct dfn_model* m, const float* feat_erb, int T, float* out_e0, float* out_e1,
                                  float* out_e2, float* out_e3) {
    if (!m || !feat_erb || !out_e0 || !out_e1 || !out_e2 || !out_e3 || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/false);
    if (!r.ctx0) return -5;
    auto io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    if (!io.e0 || !io.e1 || !io.e2 || !io.e3) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(io.e0);
    ggml_set_output(io.e1);
    ggml_set_output(io.e2);
    ggml_set_output(io.e3);
    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, io.e0);
    ggml_build_forward_expand(gf, io.e1);
    ggml_build_forward_expand(gf, io.e2);
    ggml_build_forward_expand(gf, io.e3);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(io.e0, out_e0, 0, (size_t)1 * 64 * T * 32 * sizeof(float));
    ggml_backend_tensor_get(io.e1, out_e1, 0, (size_t)1 * 64 * T * 16 * sizeof(float));
    ggml_backend_tensor_get(io.e2, out_e2, 0, (size_t)1 * 64 * T * 8  * sizeof(float));
    ggml_backend_tensor_get(io.e3, out_e3, 0, (size_t)1 * 64 * T * 8  * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_encoder_emb(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                           float* out_pre_gru, float* out_emb) {
    if (!m || !feat_erb || !feat_spec || !out_pre_gru || !out_emb || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    // Zero-init GRU hidden state for batch validation.
    ggml_tensor* h_init = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_set_name(h_init, "gru_h_init");
    ggml_set_input(h_init);

    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    if (!erb_io.e3 || !df_io.df_feat) { ggml_free(r.ctx0); return -4; }
    auto cb = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_init);
    if (!gru.pre_gru || !gru.emb) { ggml_free(r.ctx0); return -4; }

    ggml_set_output(gru.pre_gru);
    ggml_set_output(gru.emb);
    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, gru.pre_gru);
    ggml_build_forward_expand(gf, gru.emb);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    ggml_backend_tensor_set(h_init, h_zero.data(), 0, 256 * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(gru.pre_gru, out_pre_gru, 0, (size_t)256 * T * sizeof(float));
    ggml_backend_tensor_get(gru.emb,     out_emb,     0, (size_t)512 * T * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_df_dec(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                      float* out_coefs, float* out_alpha) {
    if (!m || !feat_erb || !feat_spec || !out_coefs || !out_alpha || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    ggml_tensor* h_enc = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_df1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_df2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_df1, h_df2}) { ggml_set_input(h); }

    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    auto cb     = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru    = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    auto dfd    = build_df_dec(r.ctx0, m->weights, gru.emb, df_io.c0, h_df1, h_df2);
    if (!dfd.coefs || !dfd.alpha) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(dfd.coefs);
    ggml_set_output(dfd.alpha);

    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, dfd.coefs);
    ggml_build_forward_expand(gf, dfd.alpha);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_df1, h_df2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(dfd.coefs, out_coefs, 0, (size_t)10 * 96 * T * sizeof(float));
    ggml_backend_tensor_get(dfd.alpha, out_alpha, 0, (size_t)1  * T      * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_df_dec_partial(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                              float* out_skip_add, float* out_alpha, float* out_df_out_tanh) {
    if (!m || !feat_erb || !feat_spec || !out_skip_add || !out_alpha || !out_df_out_tanh || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    // Three zero-init hidden states: encoder emb_gru + two df_dec GRUs.
    // The erb_dec branch is intentionally skipped here so the test
    // focuses on df_dec; erb_dec hidden states would be dangling
    // (no op references them) and sched would refuse to allocate.
    ggml_tensor* h_enc = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_df1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_df2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_df1, h_df2}) { ggml_set_input(h); }

    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    auto cb     = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru    = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    auto dfd    = build_df_dec_no_convp(r.ctx0, m->weights, gru.emb, h_df1, h_df2);
    if (!dfd.skip_add || !dfd.alpha || !dfd.df_out_tanh) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(dfd.skip_add);
    ggml_set_output(dfd.alpha);
    ggml_set_output(dfd.df_out_tanh);

    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, dfd.skip_add);
    ggml_build_forward_expand(gf, dfd.alpha);
    ggml_build_forward_expand(gf, dfd.df_out_tanh);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_df1, h_df2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(dfd.skip_add,    out_skip_add,    0, (size_t)256 * T * sizeof(float));
    ggml_backend_tensor_get(dfd.alpha,       out_alpha,       0, (size_t)1   * T * sizeof(float));
    ggml_backend_tensor_get(dfd.df_out_tanh, out_df_out_tanh, 0, (size_t)960 * T * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_erb_dec_m(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                         float* out_m) {
    if (!m || !feat_erb || !feat_spec || !out_m || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    ggml_tensor* h_enc  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2}) { ggml_set_input(h); }

    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    auto cb     = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru    = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    ggml_tensor* emb_gru_out = build_erb_dec_emb_gru(r.ctx0, m->weights, gru.emb, h_erb1, h_erb2).emb_gru_out;
    ggml_tensor* ct3         = build_erb_dec_up_to_convt3(r.ctx0, m->weights, emb_gru_out, erb_io.e3);
    ggml_tensor* ct2         = build_erb_dec_up_to_convt2(r.ctx0, m->weights, ct3, erb_io.e2);
    ggml_tensor* ct1         = build_erb_dec_up_to_convt1(r.ctx0, m->weights, ct2, erb_io.e1);
    ggml_tensor* mask        = build_erb_dec_m(r.ctx0, m->weights, ct1, erb_io.e0);
    if (!mask) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(mask);

    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, mask);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(mask, out_m, 0, (size_t)32 * T * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_erb_dec_convt2(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                              float* out_convt2) {
    if (!m || !feat_erb || !feat_spec || !out_convt2 || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    ggml_tensor* h_enc  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2}) { ggml_set_input(h); }

    auto erb_io  = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io   = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    auto cb      = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru     = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    ggml_tensor* emb_gru_out = build_erb_dec_emb_gru(r.ctx0, m->weights, gru.emb, h_erb1, h_erb2).emb_gru_out;
    ggml_tensor* convt3_out  = build_erb_dec_up_to_convt3(r.ctx0, m->weights, emb_gru_out, erb_io.e3);
    ggml_tensor* convt2_out  = build_erb_dec_up_to_convt2(r.ctx0, m->weights, convt3_out, erb_io.e2);
    if (!convt2_out) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(convt2_out);

    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, convt2_out);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(convt2_out, out_convt2, 0, (size_t)64 * T * 16 * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_erb_dec_convt3(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                              float* out_convt3) {
    if (!m || !feat_erb || !feat_spec || !out_convt3 || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    ggml_tensor* h_enc  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2}) { ggml_set_input(h); }

    auto erb_io  = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io   = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    auto cb      = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru     = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    ggml_tensor* emb_gru_out = build_erb_dec_emb_gru(r.ctx0, m->weights, gru.emb, h_erb1, h_erb2).emb_gru_out;
    ggml_tensor* convt3_out  = build_erb_dec_up_to_convt3(r.ctx0, m->weights, emb_gru_out, erb_io.e3);
    if (!convt3_out) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(convt3_out);

    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, convt3_out);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(convt3_out, out_convt3, 0, (size_t)64 * T * 8 * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_erb_dec_emb_gru(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                               float* out_emb_gru_out) {
    if (!m || !feat_erb || !feat_spec || !out_emb_gru_out || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    // Three zero-init hidden states: encoder emb_gru + two erb_dec GRUs.
    ggml_tensor* h_enc  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2}) { ggml_set_input(h); }

    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    if (!erb_io.e3 || !df_io.df_feat) { ggml_free(r.ctx0); return -4; }
    auto cb  = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    if (!gru.emb) { ggml_free(r.ctx0); return -4; }

    ggml_tensor* erb_emb_out = build_erb_dec_emb_gru(r.ctx0, m->weights, gru.emb, h_erb1, h_erb2).emb_gru_out;
    if (!erb_emb_out) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(erb_emb_out);

    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, erb_emb_out);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));

    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(erb_emb_out, out_emb_gru_out, 0, (size_t)512 * T * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_encoder_combined(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                                float* out_df_emb, float* out_combined) {
    if (!m || !feat_erb || !feat_spec || !out_df_emb || !out_combined || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;
    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    if (!erb_io.e3 || !df_io.df_feat) { ggml_free(r.ctx0); return -4; }
    auto cb = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    if (!cb.df_emb || !cb.combined) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(cb.df_emb);
    ggml_set_output(cb.combined);
    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, cb.df_emb);
    ggml_build_forward_expand(gf, cb.combined);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(cb.df_emb,   out_df_emb,   0, (size_t)512 * T * sizeof(float));
    ggml_backend_tensor_get(cb.combined, out_combined, 0, (size_t)512 * T * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

// Batch-mode equivalent of `predict_dfn3()`'s neural pass: run the
// whole DFN3 pipeline (encoder → erb_dec → df_dec) for T frames in a
// single graph and return the per-frame mask, complex coefs and α.
// No streaming state — GRU hiddens are zero-initialised, c0_history
// uses the standard 4-zero-frame causal pad (same semantics as the
// model's training-time padding).
//
// Inputs:
//   feat_erb   : (T, 32) contiguous
//   feat_spec  : (2, T, 96) contiguous (channel × time × freq)
// Outputs:
//   out_mask   : (T, 32)              — final ERB-band gains m[T, 32]
//   out_coefs  : (T, 96, 10) flat     — DF complex coefs (re/im interleaved)
//   out_alpha  : (T)                  — α-gate per frame
//
// Used by `dfn-denoise-wav --batch` and Python A/B harnesses to
// isolate model-graph correctness from streaming-state correctness.
extern "C" __attribute__((visibility("default")))
int dfn_debug_full_predict(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                            float* out_mask, float* out_coefs, float* out_alpha) {
    if (!m || !feat_erb || !feat_spec || !out_mask || !out_coefs || !out_alpha || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;

    ggml_tensor* h_enc  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb1 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_erb2 = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_df1  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    ggml_tensor* h_df2  = ggml_new_tensor_4d(r.ctx0, GGML_TYPE_F32, 256, 1, 1, 1);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2, h_df1, h_df2}) { ggml_set_input(h); }

    auto erb_io = build_enc_erb_branch(r.ctx0, m->weights, r.feat_erb_padded);
    auto df_io  = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    auto cb     = build_enc_combine(r.ctx0, m->weights, df_io.df_feat, erb_io.e3);
    auto gru    = build_enc_emb_gru(r.ctx0, m->weights, cb.combined, h_enc);
    auto erb_gru = build_erb_dec_emb_gru(r.ctx0, m->weights, gru.emb, h_erb1, h_erb2);
    ggml_tensor* ct3  = build_erb_dec_up_to_convt3(r.ctx0, m->weights, erb_gru.emb_gru_out, erb_io.e3);
    ggml_tensor* ct2  = build_erb_dec_up_to_convt2(r.ctx0, m->weights, ct3, erb_io.e2);
    ggml_tensor* ct1  = build_erb_dec_up_to_convt1(r.ctx0, m->weights, ct2, erb_io.e1);
    ggml_tensor* mask = build_erb_dec_m(r.ctx0, m->weights, ct1, erb_io.e0);
    // No c0_history: build_df_dec uses T-ge-4 zero-pad fallback,
    // matching training-time behaviour for the first 4 frames.
    auto dfd = build_df_dec(r.ctx0, m->weights, gru.emb, df_io.c0, h_df1, h_df2);
    if (!mask || !dfd.coefs || !dfd.alpha) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(mask);
    ggml_set_output(dfd.coefs);
    ggml_set_output(dfd.alpha);

    // Batch mode unrolls 5 GRUs over T frames + the full encoder &
    // decoder stacks, so the per-frame node count from
    // `ggml_new_graph`'s default (2048) is far too small. 64k headroom
    // is enough for T up to a few hundred; we cap T outside.
    ggml_cgraph* gf = ggml_new_graph_custom(r.ctx0, /*size=*/65536, /*grads=*/false);
    ggml_build_forward_expand(gf, mask);
    ggml_build_forward_expand(gf, dfd.coefs);
    ggml_build_forward_expand(gf, dfd.alpha);

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_erb_padded(r.feat_erb_padded, feat_erb, T);
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    std::vector<float> h_zero(256, 0.0f);
    for (ggml_tensor* h : {h_enc, h_erb1, h_erb2, h_df1, h_df2})
        ggml_backend_tensor_set(h, h_zero.data(), 0, 256 * sizeof(float));
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(mask,      out_mask,  0, (size_t)32 * T      * sizeof(float));
    ggml_backend_tensor_get(dfd.coefs, out_coefs, 0, (size_t)10 * 96 * T * sizeof(float));
    ggml_backend_tensor_get(dfd.alpha, out_alpha, 0, (size_t)T           * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}

extern "C" __attribute__((visibility("default")))
int dfn_debug_encoder_c0(const struct dfn_model* m, const float* feat_spec, int T, float* out_c0, int out_c0_cap) {
    if (!m || !feat_spec || !out_c0 || T <= 0) return -1;
    if (!m->has_neural_weights || !m->sched) return -2;
    const int n_out = 1 * 64 * T * 96;
    if (out_c0_cap < n_out) return -3;
    enc_run_ctx r = begin_encoder_graph(m, T, /*want_spec=*/true);
    if (!r.ctx0) return -5;
    auto df = build_enc_df_branch(r.ctx0, m->weights, r.feat_spec_padded);
    if (!df.c0) { ggml_free(r.ctx0); return -4; }
    ggml_set_output(df.c0);
    ggml_cgraph* gf = ggml_new_graph(r.ctx0);
    ggml_build_forward_expand(gf, df.c0);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) { ggml_free(r.ctx0); return -6; }
    stage_feat_spec_padded(r.feat_spec_padded, feat_spec, T);
    ggml_backend_sched_graph_compute(m->sched, gf);
    ggml_backend_tensor_get(df.c0, out_c0, 0, (size_t)n_out * sizeof(float));
    ggml_free(r.ctx0);
    return 0;
}
