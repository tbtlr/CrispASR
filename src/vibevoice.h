// vibevoice.h — Microsoft VibeVoice-ASR (σ-VAE tokenizers + Qwen2 LM).
//
// Architecture: Two ConvNeXt-style tokenizer encoders (acoustic + semantic)
// → linear connectors → Qwen2-1.5B autoregressive decoder.
// Input: raw 24kHz mono PCM. Output: structured text with timestamps.
// 1.5B params (ASR path), 4.7 GB F16, MIT license.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vibevoice_context;

struct vibevoice_context_params {
    int n_threads;
    int max_new_tokens;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    int tts_steps;   // DPM-Solver++ inference steps (default 20, min 4)
    uint32_t seed;   // RNG seed for TTS diffusion noise (0 = env/default)
    bool flash_attn; // PLAN #89 plumbing — σ-VAE encoder + Qwen2.5
                     // talker SA blocks.
    // Diffusion overrides. NaN = "unset": keep the built-in behaviour (CFG scale
    // auto-selected by model type; scaling/bias from the GGUF tensors, else the
    // 0.196 / -0.049 defaults). A non-NaN value overrides — and for scaling/bias
    // takes precedence over the GGUF tensors.
    float cfg_scale;             // classifier-free-guidance scale (>0)
    float speech_scaling_factor; // VAE latent scale (>0; latent = raw/scale - bias)
    float speech_bias_factor;    // VAE latent bias
    // CFG negative-condition anchor blend. The realtime model's negative-path
    // hidden state evolves with each speech frame, so the "reference" the CFG
    // pushes away from drifts as a reply gets long — audible as the voice
    // gradually losing energy / getting calmer. Blending in a fixed snapshot
    // (taken right after the initial IMAGE_PAD prefill) stabilises the
    // reference. Range [0, 1]: 0 = live negative path only (original
    // behaviour, drifts), 1 = fixed anchor only, 0.2 = misc/vibevoice's
    // default (live * 0.8 + anchor * 0.2). NaN = use that 0.2 default too.
    float neg_condition_anchor;
};

struct vibevoice_context_params vibevoice_context_default_params(void);

struct vibevoice_context* vibevoice_init_from_file(const char* path_model, struct vibevoice_context_params params);

void vibevoice_free(struct vibevoice_context* ctx);

// Runtime setter for the DPM-Solver++ inference step count
// (default 20). Read on every vibevoice_synthesize call (line ~3422
// in vibevoice.cpp), so post-init mutation is safe. Clamps to
// [4, 100]: below 4 the schedule is degenerate, above 100 you're
// burning latency for inaudible quality gain.
void vibevoice_set_tts_steps(struct vibevoice_context* ctx, int steps);
void vibevoice_set_seed(struct vibevoice_context* ctx, uint32_t seed);

// Runtime setters for the diffusion overrides above. Like tts_steps these are
// read on every synthesize call, so post-init mutation affects the next call.
// Pass NaN to clear an override and restore the built-in behaviour.
void vibevoice_set_cfg_scale(struct vibevoice_context* ctx, float cfg_scale);
void vibevoice_set_speech_scaling_factor(struct vibevoice_context* ctx, float factor);
void vibevoice_set_speech_bias_factor(struct vibevoice_context* ctx, float factor);
void vibevoice_set_neg_condition_anchor(struct vibevoice_context* ctx, float anchor);

// Transcribe raw 24kHz mono PCM audio.
// Returns malloc'd UTF-8 string, caller frees with free().
char* vibevoice_transcribe(struct vibevoice_context* ctx, const float* samples, int n_samples);

// Variant that additionally returns per-emitted-token ids and softmax
// probabilities. Free with vibevoice_result_free.
struct vibevoice_result {
    char* text;
    int* token_ids;
    float* token_probs;
    int n_tokens;
};

struct vibevoice_result* vibevoice_transcribe_with_probs(struct vibevoice_context* ctx, const float* samples,
                                                         int n_samples);
void vibevoice_result_free(struct vibevoice_result* r);

// Token-id → vocab piece (raw, with Qwen2/GPT-2 byte-level BPE markers
// like Ġ / Ċ — caller may need to decode). Returns empty string for
// out-of-range ids.
const char* vibevoice_token_text(struct vibevoice_context* ctx, int id);

// ── Stage-level API for differential testing ─────────────────────────────────

// Run the acoustic σ-VAE encoder. Returns a malloc'd float array of shape
// [*n_frames * *vae_dim] in row-major order (frame-major: data[t*vae_dim+c]).
// Caller frees with free(). Returns NULL on failure.
float* vibevoice_run_acoustic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                                      int* vae_dim);

// Run the semantic encoder. Same layout as acoustic. Returns NULL on failure.
float* vibevoice_run_semantic_encoder(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                                      int* vae_dim);

// Run one SpeechConnector (FC1 → RMSNorm → FC2) on pre-computed encoder mean.
// prefix: "at_conn" (acoustic) or "se_conn" (semantic).
// encoder_mean: row-major [n_frames * vae_dim].
// Returns malloc'd float [n_frames * *d_lm]. Caller frees. Returns NULL on failure.
float* vibevoice_run_connector(struct vibevoice_context* ctx, const char* prefix, const float* encoder_mean,
                               int n_frames, int vae_dim, int* d_lm);

// Run both encoders + both connectors and return the combined speech features
// (element-wise sum of acoustic and semantic connector outputs).
// Returns malloc'd float [*n_frames * *d_lm]. Caller frees. Returns NULL on failure.
float* vibevoice_encode_speech(struct vibevoice_context* ctx, const float* samples, int n_samples, int* n_frames,
                               int* d_lm);

// ── TTS API (requires GGUF converted with --include-decoder) ─────────────────

// Synthesize speech from text. Returns malloc'd 24 kHz mono PCM float array.
// n_samples is set to the number of output samples. Caller frees with free().
// Returns NULL if the model lacks decoder tensors (vibevoice.has_decoder=0).
float* vibevoice_synthesize(struct vibevoice_context* ctx, const char* text, int* out_n_samples);

// Load a voice prompt GGUF for TTS. Returns 0 on success.
// The voice prompt pre-fills KV caches with speaker identity.
int vibevoice_load_voice(struct vibevoice_context* ctx, const char* voice_path);

// ── Streaming TTS (VibeVoice-Realtime-0.5B) ──────────────────────────────────
// Push text incrementally and receive 24 kHz mono PCM chunks as they are
// synthesized, instead of waiting for the whole utterance. Requires a loaded
// voice prompt (vibevoice_load_voice). Generation runs on an internal worker
// thread; `on_audio` is invoked from that thread for each decoded chunk (the
// `pcm` pointer is valid only for the duration of the call — copy it).
struct vibevoice_tts_stream;
typedef void (*vibevoice_on_audio)(const float* pcm, int n_samples, void* user);

// Begin a streaming TTS session. Spawns the worker thread (which blocks waiting
// for the first text). Returns NULL on failure.
struct vibevoice_tts_stream* vibevoice_tts_stream_begin(struct vibevoice_context* ctx, vibevoice_on_audio on_audio,
                                                        void* user);
// Append UTF-8 text to vocalize. Non-blocking. Returns 0 on success.
int vibevoice_tts_stream_push_text(struct vibevoice_tts_stream* st, const char* utf8);
// Signal end-of-text: no more push_text calls. The worker drains remaining text,
// emits trailing audio, then finishes. Non-blocking. Returns 0 on success.
int vibevoice_tts_stream_end(struct vibevoice_tts_stream* st);
// Abort generation ASAP (user interrupt / shutdown). The worker stops emitting
// further audio; free() then joins quickly instead of synthesizing the rest.
void vibevoice_tts_stream_abort(struct vibevoice_tts_stream* st);
// Signal end (if not already), join the worker thread, and release the session.
void vibevoice_tts_stream_free(struct vibevoice_tts_stream* st);

#ifdef __cplusplus
}
#endif
