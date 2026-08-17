// dfn_fft.cpp — mixed-radix FFT used by the DFN3 streaming STFT.
//
// Approach: classic Cooley–Tukey with radix-2/3/5 butterflies. The
// length is factorised once at plan creation; per-call we run a
// stride-based recursion that walks every factor exactly once. Real
// signals go through a length-n complex FFT (no half-length trick) —
// at n=960 the simpler path wins on code size and is fast enough.
//
// Twiddles are precomputed in the plan (one length-n table). Workspace
// for the recursion is preallocated as well.

#include "dfn_fft.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Mixed-radix decomposition of `n` into factors from {5, 3, 2}, in that
// order (largest first so the small radices end up in the inner loops).
// Returns false if `n` has factors outside that set.
static bool factorise(int n, std::vector<int>& factors) {
    factors.clear();
    if (n <= 0) return false;
    for (int r : {5, 3, 2}) {
        while (n % r == 0) {
            factors.push_back(r);
            n /= r;
        }
    }
    return n == 1;
}

struct cpx {
    float re, im;
};

// Recursive Cooley–Tukey. `n` is the length being transformed,
// `stride` strides through the input (and the twiddle table at step
// 1×stride between successive butterfly lobes). `sign` is -1 for
// forward, +1 for inverse.
//
// `factor_idx` indexes into the precomputed factor list — at each
// level we peel off one radix.
static void fft_recurse(const cpx* in, cpx* out, int n, int stride, int sign,
                        const std::vector<int>& factors, int factor_idx, const cpx* tw, int tw_n) {
    if (factor_idx == (int)factors.size()) {
        // Base case: n must be 1 — single point passes through.
        out[0] = in[0];
        return;
    }
    int r    = factors[factor_idx];
    int m    = n / r;
    // Recurse: r sub-transforms of length m.
    for (int q = 0; q < r; ++q)
        fft_recurse(in + q * stride, out + q * m, m, stride * r, sign, factors, factor_idx + 1, tw, tw_n);

    // Combine: standard Cooley–Tukey mixed-radix step.
    //   X[j*m + k] = Σ_q W_n^{k*q} · W_r^{j*q} · Y_q[k]
    // Implemented as: pre-twiddle Y_q[k] by W_n^{k*q}, then a plain
    // radix-r DFT over q. W_n^{k*q} is read from the global table at
    // index (k*q*stride) mod N, since at this recursion level
    // stride = N/n.
    cpx scratch[5]; // r ≤ 5
    for (int k = 0; k < m; ++k) {
        for (int q = 0; q < r; ++q) {
            cpx y = out[q * m + k];
            int idx = (int)(((int64_t)k * q * stride) % tw_n);
            if (sign > 0) idx = (tw_n - idx) % tw_n; // conj for inverse
            cpx w = tw[idx];
            scratch[q].re = y.re * w.re - y.im * w.im;
            scratch[q].im = y.re * w.im + y.im * w.re;
        }
        for (int j = 0; j < r; ++j) {
            cpx acc{0.0f, 0.0f};
            for (int q = 0; q < r; ++q) {
                double ang = (sign * kTwoPi * j * q) / r;
                float  c   = (float)std::cos(ang);
                float  s   = (float)std::sin(ang);
                acc.re += scratch[q].re * c - scratch[q].im * s;
                acc.im += scratch[q].re * s + scratch[q].im * c;
            }
            out[j * m + k] = acc;
        }
    }
}

} // namespace

struct dfn_fft_plan {
    int n = 0;
    std::vector<int> factors;
    std::vector<cpx> twiddle; // length n: W_n^k for k=0..n-1 (forward sign)
    // Per-call scratch — sized for the forward path (length n).
    mutable std::vector<cpx> scratch_in;
    mutable std::vector<cpx> scratch_out;
};

extern "C" struct dfn_fft_plan* dfn_fft_plan_create(int n) {
    auto* p = new dfn_fft_plan();
    p->n = n;
    if (!factorise(n, p->factors)) {
        delete p;
        return nullptr;
    }
    p->twiddle.resize(n);
    for (int k = 0; k < n; ++k) {
        double ang = (-kTwoPi * k) / n; // forward sign convention
        p->twiddle[k].re = (float)std::cos(ang);
        p->twiddle[k].im = (float)std::sin(ang);
    }
    p->scratch_in.resize(n);
    p->scratch_out.resize(n);
    return p;
}

extern "C" void dfn_fft_plan_free(struct dfn_fft_plan* p) {
    delete p;
}

extern "C" int dfn_fft_plan_n(const struct dfn_fft_plan* p) {
    return p ? p->n : 0;
}

extern "C" void dfn_rfft(const struct dfn_fft_plan* p, const float* in, float* out_re, float* out_im) {
    if (!p || !in || !out_re || !out_im) return;
    const int n = p->n;
    auto& sin = p->scratch_in;
    auto& sout = p->scratch_out;
    for (int i = 0; i < n; ++i) {
        sin[i].re = in[i];
        sin[i].im = 0.0f;
    }
    fft_recurse(sin.data(), sout.data(), n, 1, /*sign=*/-1, p->factors, 0, p->twiddle.data(), n);
    const int n_bins = n / 2 + 1;
    for (int k = 0; k < n_bins; ++k) {
        out_re[k] = sout[k].re;
        out_im[k] = sout[k].im;
    }
}

extern "C" void dfn_irfft(const struct dfn_fft_plan* p, const float* in_re, const float* in_im, float* out) {
    if (!p || !in_re || !in_im || !out) return;
    const int n      = p->n;
    const int n_bins = n / 2 + 1;
    auto&     sin    = p->scratch_in;
    auto&     sout   = p->scratch_out;

    // Reconstruct the full Hermitian-symmetric spectrum.
    for (int k = 0; k < n_bins; ++k) {
        sin[k].re = in_re[k];
        sin[k].im = in_im[k];
    }
    for (int k = n_bins; k < n; ++k) {
        int sym = n - k;
        sin[k].re = in_re[sym];
        sin[k].im = -in_im[sym];
    }
    fft_recurse(sin.data(), sout.data(), n, 1, /*sign=*/+1, p->factors, 0, p->twiddle.data(), n);
    const float scale = 1.0f / (float)n;
    for (int i = 0; i < n; ++i)
        out[i] = sout[i].re * scale;
}
