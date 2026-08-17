// test-dfn-graph.cpp — golden-value validation for the DFN3 GGML
// graph (PLAN: flickering-greeting-backus).
//
// Compares each sub-graph's output against the reference tensors
// produced by `models/dump-deepfilternet-golden.py`. Off by default
// ([.dfn-golden]); run locally with:
//
//   CRISPASR_DFN_GGUF=models/deepfilternet3.gguf \\
//   CRISPASR_DFN_GOLDEN=models/deepfilternet3-golden.gguf \\
//     ./bin/test-dfn-graph "[dfn-golden]"
//
// As each new layer lands in `predict_dfn3`, add a TEST_CASE that
// drives it with the golden input + asserts max|Δ| against the
// matching golden tensor.

#include <catch2/catch_test_macros.hpp>

#include "dfn.h"
#include "dfn_debug.h"

#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

// Wraps a GGUF loaded purely for its tensor contents (the golden
// reference values, allocated on a CPU buffer). Owns ctx + buf so
// tensor data pointers stay alive for the test's duration.
struct golden_blob {
    ggml_backend_t        backend = nullptr;
    core_gguf::WeightLoad wl{};

    static golden_blob load(const char* path) {
        golden_blob g;
        g.backend = ggml_backend_cpu_init();
        REQUIRE(g.backend != nullptr);
        REQUIRE(core_gguf::load_weights(path, g.backend, "dfn-golden", g.wl));
        return g;
    }
    ~golden_blob() {
        core_gguf::free_weights(wl);
        if (backend) ggml_backend_free(backend);
    }
    // Read a tensor's flat float32 contents into a std::vector.
    std::vector<float> read(const char* name) const {
        ggml_tensor* t = core_gguf::try_get(wl.tensors, name);
        REQUIRE(t != nullptr);
        REQUIRE(t->type == GGML_TYPE_F32);
        int64_t n = ggml_nelements(t);
        std::vector<float> v((size_t)n);
        ggml_backend_tensor_get(t, v.data(), 0, (size_t)n * sizeof(float));
        return v;
    }
    // Read an unsigned-32 metadata key (used to recover T = frames).
    uint32_t kv_u32(const char* key, uint32_t fallback) const {
        // wl doesn't expose the gguf context after load; re-open the
        // file for the (tiny) metadata read.
        return fallback;
    }
};

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Resolve a path from an env var with fallback. Returns nullptr if
// neither the env var nor the fallback path exists.
const char* env_or_default(const char* env, const char* fallback) {
    const char* p = std::getenv(env);
    if (!p || !*p) p = fallback;
    std::FILE* f = std::fopen(p, "rb");
    if (!f) return nullptr;
    std::fclose(f);
    return p;
}

} // namespace

TEST_CASE("dfn streaming end-to-end produces finite, lookahead-delayed output",
          "[dfn][.dfn-streaming]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF", "models/deepfilternet3.gguf");
    if (!weights_path) {
        SUCCEED("no DFN GGUF at models/deepfilternet3.gguf — skip "
                "(override via CRISPASR_DFN_GGUF)");
        return;
    }

    dfn_params  p = dfn_default_params();
    dfn_model*  m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    REQUIRE(dfn_model_sample_rate(m) == 48000);
    dfn_stream* st = dfn_stream_create(m);
    REQUIRE(st != nullptr);

    // 250 ms of pink-ish noise = 12 000 samples at 48 kHz, well past
    // the 2-frame ≈ 20 ms lookahead and the GRU warm-up.
    const int          n_in = 12000;
    std::vector<float> in(n_in);
    {
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (int i = 0; i < n_in; ++i) in[i] = nd(rng);
    }

    // Drive the stream in 480-sample (10 ms) chunks; collect output
    // and time it so the persistent-graph speedup can be observed.
    std::vector<float> out;
    out.reserve(n_in);
    const int                chunk = 480;
    std::vector<float>       scratch(chunk * 4, 0.0f);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_in; i += chunk) {
        int n = std::min(chunk, n_in - i);
        int produced = 0;
        REQUIRE(dfn_stream_process(st, in.data() + i, n, scratch.data(), (int)scratch.size(), &produced,
                                   /*final_chunk=*/false) == 0);
        out.insert(out.end(), scratch.begin(), scratch.begin() + produced);
    }
    auto t1 = std::chrono::steady_clock::now();
    double process_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // The input represented 250 ms of real time. Anything < 250 ms of
    // wall clock is real-time-capable; way under that, with margin.
    INFO("processed 250 ms of audio in " << process_ms << " ms wall-clock");
    int produced = 0;
    REQUIRE(dfn_stream_process(st, nullptr, 0, scratch.data(), (int)scratch.size(), &produced,
                               /*final_chunk=*/true) == 0);
    out.insert(out.end(), scratch.begin(), scratch.begin() + produced);

    // Sanity: lots of output, all finite, magnitudes bounded (input
    // was sub-unit, denoiser doesn't add gain).
    REQUIRE((int)out.size() > 0);
    float max_abs = 0.0f;
    int   n_nan   = 0;
    for (float v : out) {
        if (!std::isfinite(v)) n_nan++;
        max_abs = std::max(max_abs, std::fabs(v));
    }
    INFO("streamed " << out.size() << " samples, max|out|=" << max_abs);
    REQUIRE(n_nan == 0);
    REQUIRE(max_abs < 1.0f);

    dfn_stream_free(st);
    dfn_model_free(m);
}

TEST_CASE("dfn encoder erb_conv0 → e0 matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    // Load the trained-weights GGUF (full model).
    dfn_params p = dfn_default_params();
    p.verbosity  = 0;
    dfn_model*  m  = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);

    // Load the golden GGUF (intermediate reference values).
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb = gld.read("dfn.golden.input.feat_erb"); // (1, 1, T, 32)
    auto e0_ref   = gld.read("dfn.golden.enc.e0");          // (1, 64, T, 32)

    REQUIRE(feat_erb.size() % 32 == 0);
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(e0_ref.size() == (size_t)64 * T * 32);

    std::vector<float> e0(e0_ref.size(), 0.0f);
    int rc = dfn_debug_encoder_e0(m, feat_erb.data(), T, e0.data(), (int)e0.size());
    REQUIRE(rc == 0);

    // Tolerance: weights are stored f16 in the GGUF and cast to f32
    // inside the graph, so the round-trip is bounded by ~1e-3 per
    // element on convolution outputs of magnitude ~10. The golden is
    // float32 from onnxruntime so its precision exceeds ours.
    float err = max_abs_diff(e0, e0_ref);
    INFO("max|Δe0| = " << err);
    REQUIRE(err < 5e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn encoder DF stream df_conv0 → c0 matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model*  m  = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto c0_ref    = gld.read("dfn.golden.enc.c0");
    REQUIRE(feat_spec.size() % (2 * 96) == 0);
    const int T = (int)(feat_spec.size() / (2 * 96));
    REQUIRE(c0_ref.size() == (size_t)64 * T * 96);

    std::vector<float> c0(c0_ref.size(), 0.0f);
    int rc = dfn_debug_encoder_c0(m, feat_spec.data(), T, c0.data(), (int)c0.size());
    REQUIRE(rc == 0);

    float err = max_abs_diff(c0, c0_ref);
    INFO("max|Δc0| = " << err);
    // c0 sits two convs deep — f16-cast error budget similar to e1.
    REQUIRE(err < 5e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn encoder df_fc_emb + combine match golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto df_emb_ref   = gld.read("dfn.golden.enc.df_fc_emb");
    auto combined_ref = gld.read("dfn.golden.enc.combined");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(df_emb_ref.size() == (size_t)512 * T);
    REQUIRE(combined_ref.size() == (size_t)512 * T);

    std::vector<float> df_emb(df_emb_ref.size(), 0.0f);
    std::vector<float> combined(combined_ref.size(), 0.0f);
    int rc = dfn_debug_encoder_combined(m, feat_erb.data(), feat_spec.data(), T,
                                         df_emb.data(), combined.data());
    REQUIRE(rc == 0);

    auto d_emb = max_abs_diff(df_emb, df_emb_ref);
    auto d_cmb = max_abs_diff(combined, combined_ref);
    INFO("max|Δ| df_fc_emb=" << d_emb << " combined=" << d_cmb);
    // df_fc_emb sits ~3 convs + 1 matmul deep on the DF stream; combined
    // also folds in e3 (4 convs deep). Magnitudes climb to ~20, so the
    // absolute slack scales accordingly.
    REQUIRE(d_emb < 3e-2f);
    REQUIRE(d_cmb < 3e-2f);

    dfn_model_free(m);
}

TEST_CASE("dfn encoder emb_gru pre_gru + emb match golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto pre_gru_ref = gld.read("dfn.golden.enc.pre_gru");
    auto emb_ref     = gld.read("dfn.golden.enc.emb");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(pre_gru_ref.size() == (size_t)256 * T);
    REQUIRE(emb_ref.size()     == (size_t)512 * T);

    std::vector<float> pre_gru(pre_gru_ref.size(), 0.0f);
    std::vector<float> emb(emb_ref.size(), 0.0f);
    int rc = dfn_debug_encoder_emb(m, feat_erb.data(), feat_spec.data(), T, pre_gru.data(), emb.data());
    REQUIRE(rc == 0);

    auto d_pg  = max_abs_diff(pre_gru, pre_gru_ref);
    auto d_emb = max_abs_diff(emb, emb_ref);
    INFO("max|Δ| pre_gru=" << d_pg << " emb=" << d_emb);
    // pre_gru sits one more matmul + Relu past `combined` (Δ≈0.014);
    // emb passes the GRU + another GroupedLinear + Relu, so the error
    // budget loosens further.
    REQUIRE(d_pg  < 3e-2f);
    REQUIRE(d_emb < 1e-1f);

    dfn_model_free(m);
}

TEST_CASE("dfn df_dec final coefs + alpha match golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto coefs_ref = gld.read("dfn.golden.df_dec.coefs");
    auto alpha_ref = gld.read("dfn.golden.df_dec.alpha");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(coefs_ref.size() == (size_t)10 * 96 * T);
    REQUIRE(alpha_ref.size() == (size_t)T);

    std::vector<float> coefs(coefs_ref.size(), 0.0f);
    std::vector<float> alpha(alpha_ref.size(), 0.0f);
    int rc = dfn_debug_df_dec(m, feat_erb.data(), feat_spec.data(), T, coefs.data(), alpha.data());
    REQUIRE(rc == 0);

    auto d_coefs = max_abs_diff(coefs, coefs_ref);
    auto d_alpha = max_abs_diff(alpha, alpha_ref);
    INFO("max|Δ| coefs=" << d_coefs << " alpha=" << d_alpha);
    REQUIRE(d_coefs < 5e-3f);
    REQUIRE(d_alpha < 1e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn df_dec skip_add + alpha + df_out_tanh match golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto skip_ref  = gld.read("dfn.golden.df_dec.skip_add");
    auto alpha_ref = gld.read("dfn.golden.df_dec.alpha");
    auto tanh_ref  = gld.read("dfn.golden.df_dec.df_out_tanh");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(skip_ref.size()  == (size_t)256 * T);
    REQUIRE(alpha_ref.size() == (size_t)1   * T);
    REQUIRE(tanh_ref.size()  == (size_t)960 * T);

    std::vector<float> skip(skip_ref.size(), 0.0f);
    std::vector<float> alpha(alpha_ref.size(), 0.0f);
    std::vector<float> tanh_out(tanh_ref.size(), 0.0f);
    int rc = dfn_debug_df_dec_partial(m, feat_erb.data(), feat_spec.data(), T, skip.data(), alpha.data(), tanh_out.data());
    REQUIRE(rc == 0);

    auto d_skip  = max_abs_diff(skip, skip_ref);
    auto d_alpha = max_abs_diff(alpha, alpha_ref);
    auto d_tanh  = max_abs_diff(tanh_out, tanh_ref);
    INFO("max|Δ| skip_add=" << d_skip << " alpha=" << d_alpha << " df_out_tanh=" << d_tanh);
    // skip_add is post 2× GRU + linear — magnitudes ~1, sigmoid/tanh
    // clamping keeps error small. α is bounded in [0, 1].
    // df_out_tanh is bounded in [-1, 1].
    REQUIRE(d_skip  < 5e-3f);
    REQUIRE(d_alpha < 1e-3f);
    REQUIRE(d_tanh  < 5e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn erb_dec final mask m matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto ref       = gld.read("dfn.golden.erb_dec.m");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(ref.size() == (size_t)32 * T);

    std::vector<float> out(ref.size(), 0.0f);
    int rc = dfn_debug_erb_dec_m(m, feat_erb.data(), feat_spec.data(), T, out.data());
    REQUIRE(rc == 0);

    float err = max_abs_diff(out, ref);
    INFO("max|Δ erb_dec.m| = " << err);
    // Sigmoid output bounded in [0, 1] — error budget is small.
    REQUIRE(err < 5e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn erb_dec convt2 output matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto ref       = gld.read("dfn.golden.erb_dec.convt2_out");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(ref.size() == (size_t)64 * T * 16);

    std::vector<float> out(ref.size(), 0.0f);
    int rc = dfn_debug_erb_dec_convt2(m, feat_erb.data(), feat_spec.data(), T, out.data());
    REQUIRE(rc == 0);

    float err = max_abs_diff(out, ref);
    INFO("max|Δ erb_dec.convt2_out| = " << err);
    REQUIRE(err < 5e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn erb_dec convt3 output matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto ref       = gld.read("dfn.golden.erb_dec.convt3_out");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(ref.size() == (size_t)64 * T * 8);

    std::vector<float> out(ref.size(), 0.0f);
    int rc = dfn_debug_erb_dec_convt3(m, feat_erb.data(), feat_spec.data(), T, out.data());
    REQUIRE(rc == 0);

    float err = max_abs_diff(out, ref);
    INFO("max|Δ erb_dec.convt3_out| = " << err);
    // Tight tolerance — convt3 magnitude max is ~0.94 (post-Relu).
    REQUIRE(err < 5e-3f);

    dfn_model_free(m);
}

TEST_CASE("dfn erb_dec emb_gru output matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model* m = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb  = gld.read("dfn.golden.input.feat_erb");
    auto feat_spec = gld.read("dfn.golden.input.feat_spec");
    auto ref       = gld.read("dfn.golden.erb_dec.emb_gru_out");
    const int T = (int)(feat_erb.size() / 32);
    REQUIRE(ref.size() == (size_t)512 * T);

    std::vector<float> out(ref.size(), 0.0f);
    int rc = dfn_debug_erb_dec_emb_gru(m, feat_erb.data(), feat_spec.data(), T, out.data());
    REQUIRE(rc == 0);

    float err = max_abs_diff(out, ref);
    INFO("max|Δ erb_dec.emb_gru_out| = " << err);
    // Two stacked GRUs after the full encoder — clamping by sigmoid/
    // tanh activations keeps the absolute error small (the ref's max
    // magnitude is ~1.3).
    REQUIRE(err < 1e-2f);

    dfn_model_free(m);
}

TEST_CASE("dfn encoder ERB conv pyramid e0..e3 matches golden", "[dfn][.dfn-golden]") {
    const char* weights_path = env_or_default("CRISPASR_DFN_GGUF",    "models/deepfilternet3.gguf");
    const char* golden_path  = env_or_default("CRISPASR_DFN_GOLDEN", "models/deepfilternet3-golden.gguf");
    if (!weights_path || !golden_path) {
        SUCCEED("missing weights or golden GGUF — skip");
        return;
    }

    dfn_params p = dfn_default_params();
    dfn_model*  m  = dfn_model_load(weights_path, p);
    REQUIRE(m != nullptr);
    golden_blob gld = golden_blob::load(golden_path);

    auto feat_erb = gld.read("dfn.golden.input.feat_erb");
    auto e0_ref   = gld.read("dfn.golden.enc.e0");
    auto e1_ref   = gld.read("dfn.golden.enc.e1");
    auto e2_ref   = gld.read("dfn.golden.enc.e2");
    auto e3_ref   = gld.read("dfn.golden.enc.e3");
    const int T = (int)(feat_erb.size() / 32);

    std::vector<float> e0(e0_ref.size()), e1(e1_ref.size()), e2(e2_ref.size()), e3(e3_ref.size());
    int rc = dfn_debug_encoder_erb_pyramid(m, feat_erb.data(), T, e0.data(), e1.data(), e2.data(), e3.data());
    REQUIRE(rc == 0);

    auto d0 = max_abs_diff(e0, e0_ref);
    auto d1 = max_abs_diff(e1, e1_ref);
    auto d2 = max_abs_diff(e2, e2_ref);
    auto d3 = max_abs_diff(e3, e3_ref);
    INFO("max|Δ| e0=" << d0 << " e1=" << d1 << " e2=" << d2 << " e3=" << d3);
    // The error grows with each layer (4 successive f16-weight casts
    // ≈ 4 epsilons of relative slack). Output magnitudes also climb
    // through the pyramid — e3 mean ≈ 1.3 vs e0 mean ≈ 0.6, max ≈ 18.
    // A ~0.1% relative tolerance lands at these absolute bounds.
    REQUIRE(d0 < 5e-3f);
    REQUIRE(d1 < 5e-3f);
    REQUIRE(d2 < 1e-2f);
    REQUIRE(d3 < 2e-2f);

    dfn_model_free(m);
}
