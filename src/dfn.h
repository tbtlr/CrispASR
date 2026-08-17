// dfn.h — DeepFilterNet3 streaming speech enhancement (GGML port).
//
// DFN3 is a low-latency 48 kHz speech denoiser. Two-stage:
//   1) ERB-band magnitude gain (32 bands)
//   2) Complex "deep filter" coefficients on the first ~5 kHz
//
// Used by the session-layer TTS post-filter
// (`crispasr_session_set_tts_postfilter`) to scrub background-music
// artefacts out of VibeVoice realtime output (and to push the sample
// rate up to 48 kHz as a side effect).
//
// Same "consume PCM → produce PCM" layering as `crispasr_enhance.h`.
// Different surface, because DFN is *stateful* — for streaming we
// need a per-utterance handle that owns the STFT ring buffer, GRU
// hidden states and the lookahead delay.
//
// All audio is 48 kHz mono float32 in [-1, 1].

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dfn_model;
struct dfn_stream;

struct dfn_params {
    int  n_threads;     // GGML thread count for the per-frame graph (default 2)
    bool use_gpu;       // Metal/CUDA offload of the conv stack (default false — model is tiny)
    int  verbosity;     // 0 silent, 1 normal, 2 verbose
    float atten_lim_db; // Max attenuation per band. <= 0 disables the cap (= raw DFN).
                        // 100 dB matches the upstream default.
};

struct dfn_params dfn_default_params(void);

// Load the GGUF produced by `models/convert-deepfilternet-to-gguf.py`.
// Returns NULL on failure. Thread-safe to share across streams.
struct dfn_model* dfn_model_load(const char* gguf_path, struct dfn_params params);
void              dfn_model_free(struct dfn_model* m);

// Native operating rate (48000). Exposed so the session layer can
// configure its resampler without baking the number in.
int dfn_model_sample_rate(const struct dfn_model* m);

// Frame size in samples (480 = 10 ms @ 48 kHz). Callers don't need
// to feed whole frames — the stream buffers internally — but knowing
// this is useful for sizing output buffers in steady state.
int dfn_model_frame_size(const struct dfn_model* m);

// Number of lookahead frames the model needs before emitting the first
// sample (2 frames ≈ 20 ms for DFN3). After draining, `final_chunk=true`
// flushes them.
int dfn_model_lookahead_frames(const struct dfn_model* m);

// Per-utterance state. Allocate one per concurrent stream — they are
// NOT thread-safe (the GGML scheduler is owned by the model, but the
// scratch buffers live on the stream).
struct dfn_stream* dfn_stream_create(const struct dfn_model* m);
void               dfn_stream_reset(struct dfn_stream* st);
void               dfn_stream_free(struct dfn_stream* st);

// Pre-warm the stream by running `n_frames` of silence through the
// full graph. Used at the start of an utterance to converge the EMA
// state (log-mag mean, per-bin unit-norm) and the recurrent GRU
// hidden state away from their zero / linspace-init values before
// real audio arrives. Without this, the first ~0.5–1 s of output
// contains audible warm-up distortion as the EMA chases real input.
//
// Recommended `n_frames` for a clean start: 100 (= 1 s of warm-up
// audio at the model's 100 fps frame rate). Lower if you can tolerate
// some residual artefact; higher buys diminishing returns. Returns 0
// on success.
int dfn_stream_warmup(struct dfn_stream* st, int n_frames);

// Push `n_in` input samples (48 kHz mono f32); write up to `out_cap`
// denoised samples into `out`. On return `*produced` holds the actual
// number of output samples. Internal lookahead means the first frames
// produce zero output until the lookahead window fills. Pass
// `final_chunk=true` on the last call to drain the tail.
//
// Returns 0 on success, negative on failure.
int dfn_stream_process(struct dfn_stream* st, const float* in, int n_in, float* out, int out_cap, int* produced,
                       bool final_chunk);

#ifdef __cplusplus
}
#endif
