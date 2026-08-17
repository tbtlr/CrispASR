// test-dfn-fft.cpp — round-trip sanity for the mixed-radix FFT used by
// the DeepFilterNet3 streaming post-filter (PLAN: flickering-greeting-backus).
//
// We check three things:
//   1. rfft → irfft is the identity (within float epsilon).
//   2. The bin layout matches the standard real-FFT convention:
//      DC bin is real-valued; bin 1 of a unit-amplitude cosine at
//      frequency k=1 lands at (n/2, 0).
//   3. The same plan covers DFN3's exact length (n = 960 = 2^6·3·5).
//
// No model load, no audio — runs in microseconds; tagged [unit].

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dfn.h"
#include "dfn_fft.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Largest absolute deviation between two equal-length float arrays.
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

} // namespace

TEST_CASE("dfn_fft round-trip identity for DFN's n=960", "[unit][dfn]") {
    constexpr int n = 960;
    auto* plan = dfn_fft_plan_create(n);
    REQUIRE(plan != nullptr);

    std::mt19937            rng(42);
    std::normal_distribution<float> noise(0.0f, 0.3f);

    std::vector<float> in(n), recon(n);
    for (int i = 0; i < n; ++i) in[i] = noise(rng);

    std::vector<float> re(n / 2 + 1, 0.0f), im(n / 2 + 1, 0.0f);
    dfn_rfft(plan, in.data(), re.data(), im.data());
    dfn_irfft(plan, re.data(), im.data(), recon.data());

    // 960 = 2^6·3·5 — six radix-2 passes accumulate ~6·eps of error,
    // radix-3/5 add a few more. 1e-4 leaves comfortable headroom.
    REQUIRE(max_abs_diff(in, recon) < 1e-4f);

    dfn_fft_plan_free(plan);
}

TEST_CASE("dfn_fft delta-function spectrum", "[unit][dfn]") {
    constexpr int n = 960;
    auto* plan = dfn_fft_plan_create(n);
    REQUIRE(plan != nullptr);

    // Unit-amplitude cosine at k=1: x[i] = cos(2π·i/n).
    // Expected DFT: bin 1 = n/2 (real), all others ≈ 0.
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = std::cos(kTwoPi * (double)i / (double)n);

    std::vector<float> re(n / 2 + 1, 0.0f), im(n / 2 + 1, 0.0f);
    dfn_rfft(plan, in.data(), re.data(), im.data());

    using Catch::Matchers::WithinAbs;
    REQUIRE_THAT(re[1], WithinAbs(n / 2.0, 1e-2));
    REQUIRE_THAT(im[1], WithinAbs(0.0, 1e-2));
    // Every other bin should be near zero.
    for (int k = 0; k < (int)re.size(); ++k) {
        if (k == 1) continue;
        REQUIRE_THAT(re[k], WithinAbs(0.0, 1e-2));
        REQUIRE_THAT(im[k], WithinAbs(0.0, 1e-2));
    }

    dfn_fft_plan_free(plan);
}

TEST_CASE("dfn_fft rejects unsupported lengths", "[unit][dfn]") {
    // 7 has a prime factor outside {2,3,5}.
    REQUIRE(dfn_fft_plan_create(7) == nullptr);
    REQUIRE(dfn_fft_plan_create(0) == nullptr);
}

// End-to-end load + streaming smoke against the real ONNX-converted
// GGUF. Off by default ([.dfn-real]) so CI doesn't depend on the file;
// run locally with:
//   CRISPASR_DFN_GGUF=models/deepfilternet3.gguf ./bin/test-dfn-fft "[dfn-real]"
TEST_CASE("dfn_model_load + stream_process round-trip on a real GGUF", "[dfn][.dfn-real]") {
    const char* path = std::getenv("CRISPASR_DFN_GGUF");
    if (!path) path = "models/deepfilternet3.gguf";
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        SUCCEED("no DFN GGUF at " << path << " — skip (override via CRISPASR_DFN_GGUF)");
        return;
    }
    std::fclose(f);

    dfn_params p = dfn_default_params();
    p.verbosity  = 1; // log "neural_weights=yes/no" via stderr
    dfn_model*  m  = dfn_model_load(path, p);
    REQUIRE(m != nullptr);
    REQUIRE(dfn_model_sample_rate(m) == 48000);
    REQUIRE(dfn_model_frame_size(m) == 480);
    REQUIRE(dfn_model_lookahead_frames(m) >= 0);

    dfn_stream* st = dfn_stream_create(m);
    REQUIRE(st != nullptr);

    // 50 ms of silence — long enough to drive the streaming state
    // machine through several frames + the lookahead-prime transition.
    const int  n_in = dfn_model_sample_rate(m) / 20;
    std::vector<float> in(n_in, 0.0f);
    std::vector<float> out(n_in * 2, 0.0f);
    int produced = 0;
    REQUIRE(dfn_stream_process(st, in.data(), n_in, out.data(), (int)out.size(), &produced,
                               /*final_chunk=*/false) == 0);
    REQUIRE(dfn_stream_process(st, nullptr, 0, out.data(), (int)out.size(), &produced,
                               /*final_chunk=*/true) == 0);

    dfn_stream_free(st);
    dfn_model_free(m);
}
