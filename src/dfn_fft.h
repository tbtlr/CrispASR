// dfn_fft.h — small self-contained mixed-radix FFT for DeepFilterNet.
//
// DFN3 needs a length-960 DFT (n_fft = 960, hop = 480, sqrt-Hann
// window). 960 = 2^6 · 3 · 5, so a mixed-radix Cooley-Tukey covering
// radices 2, 3 and 5 is sufficient — no Bluestein, no power-of-2
// zero padding (which would change the bin spacing the network was
// trained on).
//
// The implementation lives in `dfn_fft.cpp` and is intentionally
// independent of the rnnoise/kiss_fft.c copy already in the tree:
// that one is wrapped in opus_* allocator macros that only exist
// when compiled inside RNNoise. We need a clean, allocation-free
// version we can hand to the streaming TTS post-filter.
//
// Performance is fine for our use case: at 100 frames/s the per-frame
// FFT cost is ~30 µs unoptimised; way below the GGML graph cost.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct dfn_fft_plan;

// Create a forward + inverse FFT plan for length `n`. Returns NULL if
// `n` is not factorisable into {2, 3, 5}. Reusable across calls; share
// across threads only behind an external lock.
struct dfn_fft_plan* dfn_fft_plan_create(int n);
void                 dfn_fft_plan_free(struct dfn_fft_plan* p);

// Length the plan was created for.
int dfn_fft_plan_n(const struct dfn_fft_plan* p);

// Real → complex forward FFT (length n input, length n/2+1 complex output).
// `out_re` and `out_im` must each have capacity n/2 + 1.
void dfn_rfft(const struct dfn_fft_plan* p, const float* in, float* out_re, float* out_im);

// Complex → real inverse FFT (length n/2+1 complex input, length n output).
// Scaled so that irfft(rfft(x)) == x (i.e. division by n is applied inside).
void dfn_irfft(const struct dfn_fft_plan* p, const float* in_re, const float* in_im, float* out);

#ifdef __cplusplus
}
#endif
