// audio_resample.h — thin stateful float32-mono polyphase resampler.
//
// Wraps miniaudio's `ma_resampler` so the rest of the codebase doesn't
// have to know about `MINIAUDIO_IMPLEMENTATION`, ma_format conversion,
// or frame-count bookkeeping. Used by the session-layer TTS post-filter
// (24 kHz engine output ↔ 48 kHz DFN) and free for any other 1-channel
// resampling need.
//
// Implementation lives in `crispasr_audio.cpp`, which already links
// `MINIAUDIO_IMPLEMENTATION` for the WAV decoder. Header is C-callable
// from dfn.cpp / crispasr_c_api.cpp without dragging miniaudio in.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct audio_resampler;

// Create a stateful resampler. `algo` chooses the underlying miniaudio
// algorithm: 0 = linear (fast, ~no quality), 1 = sinc/polyphase (default
// for the TTS post-filter; kaiser-windowed inside miniaudio). Returns
// NULL on failure.
struct audio_resampler* audio_resampler_create(int in_rate, int out_rate, int channels, int algo);

// Push up to `n_in` input frames, write up to `out_cap` output frames.
// On return `*in_used` is the number of input frames the resampler
// consumed and `*out_written` is the number of output frames produced.
// Either may be less than the corresponding capacity — call repeatedly
// until both are zero to fully drain.
// Returns 0 on success, negative on failure.
int audio_resampler_process(struct audio_resampler* r, const float* in, int n_in, float* out, int out_cap, int* in_used,
                            int* out_written);

void audio_resampler_free(struct audio_resampler* r);

#ifdef __cplusplus
}
#endif
