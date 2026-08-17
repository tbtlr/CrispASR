// dfn_debug.h — test-only entry points for the DFN3 GGML port.
//
// These are used by `tests/test-dfn-graph.cpp` to drive one
// sub-graph of the encoder / decoders at a time and compare against
// the golden tensors produced by
// `models/dump-deepfilternet-golden.py`.
//
// They live in `dfn.cpp` (exported with default visibility despite
// libcrispasr's hidden-by-default setting) so tests can link against
// libcrispasr instead of recompiling dfn.cpp + its GGML dependencies.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct dfn_model;

// Run the encoder's first conv (`erb_conv0`) on a T-frame batch of
// ERB log-mag features and return the resulting `e0` activations.
//
//   feat_erb   : input tensor in PyTorch / golden layout (1, 1, T, 32)
//                — caller-contiguous float32 with stride 32 between
//                frames, 32 floats per frame.
//   T          : number of input frames.
//   out_e0     : output buffer for the (1, 64, T, 32) tensor — same
//                memory layout as `dfn.golden.enc.e0`. Caller sizes it
//                to at least 64 * T * 32 floats.
//   out_e0_cap : capacity of `out_e0` in float elements.
//
// Returns 0 on success, negative on failure (-1 bad args, -2 model
// has no neural weights, -4 expected tensor missing, etc.).
int dfn_debug_encoder_e0(const struct dfn_model* m, const float* feat_erb, int T, float* out_e0, int out_e0_cap);

// Run the full ERB-stream conv pyramid (`erb_conv0..3`) on a T-frame
// batch and return all four feature levels. Output layout:
//   out_e0 : (1, 64, T, 32)
//   out_e1 : (1, 64, T, 16)
//   out_e2 : (1, 64, T,  8)
//   out_e3 : (1, 64, T,  8)
// Buffers must be sized by the caller (use the dimensions above).
int dfn_debug_encoder_erb_pyramid(const struct dfn_model* m, const float* feat_erb, int T, float* out_e0,
                                   float* out_e1, float* out_e2, float* out_e3);

// Run the DF-feature branch of the encoder up through `c0` (which is
// the output of `df_conv0` + 1×1 pointwise + Relu, before the
// stride-2 df_conv1).
//
//   feat_spec : input tensor in PyTorch / golden layout (1, 2, T, 96)
//               — caller-contiguous float32, channel-major then
//               time-major then frequency-minor.
//   out_c0    : (1, 64, T, 96) — same memory layout as
//               `dfn.golden.enc.c0`. Caller sizes to 64 * T * 96 floats.
int dfn_debug_encoder_c0(const struct dfn_model* m, const float* feat_spec, int T, float* out_c0, int out_c0_cap);

// Drive the encoder up through df_fc_emb (GroupedLinear) and the
// combine_add that fuses the ERB-stream e3 features with the DF-stream
// embedding. Output layouts:
//   out_df_emb   : (1, T, 512) — post df_fc_emb + Relu
//   out_combined : (1, T, 512) — Add(e3_reshape, df_emb) before emb_gru
// Both buffers are sized by the caller (512 * T floats each).
int dfn_debug_encoder_combined(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                                float* out_df_emb, float* out_combined);

// Drive the full encoder through emb_gru's SqueezedGRU (linear_in +
// GRU step looped over T + linear_out + Relu). The GRU's hidden state
// is zero-initialised internally — fine for batch validation against
// the golden values; the streaming wrapper will hold persistent state
// on `dfn_stream` once it's wired in.
//
//   out_pre_gru : (1, T, 256) — post linear_in + Relu (GRU input)
//   out_emb     : (1, T, 512) — final encoder output
int dfn_debug_encoder_emb(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                           float* out_pre_gru, float* out_emb);

// Drive the ERB-decoder's SqueezedGRU (two stacked LBR=1 GRUs between
// linear_in and linear_out) on top of the encoder. Hidden states are
// zero-init for batch validation. Output:
//   out_emb_gru_out : (1, T, 512) — post linear_out + Relu, matches
//                     `dfn.golden.erb_dec.emb_gru_out`.
int dfn_debug_erb_dec_emb_gru(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                               float* out_emb_gru_out);

// Drive the ERB-decoder backbone up through `convt3` (the first
// non-upsampling decoder block after the SqueezedGRU + e3-skip add).
// Output:
//   out_convt3 : (1, 64, T, 8) — matches `dfn.golden.erb_dec.convt3_out`.
// Sized 64 * T * 8 floats by the caller.
int dfn_debug_erb_dec_convt3(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                              float* out_convt3);

// Drive the ERB-decoder one stage further — through `convt2`, the
// first ConvTranspose-bearing decoder block (depthwise stride-2
// upsample F: 8 → 16). Output:
//   out_convt2 : (1, 64, T, 16) — matches `dfn.golden.erb_dec.convt2_out`.
int dfn_debug_erb_dec_convt2(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                              float* out_convt2);

// Drive the entire ERB decoder end-to-end through the final
// sigmoid-gated mask `m`. Output:
//   out_m : (1, 1, T, 32) — matches `dfn.golden.erb_dec.m`. Caller sizes
//           the buffer to T * 32 floats.
int dfn_debug_erb_dec_m(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                         float* out_m);

// Drive the DF decoder *minus the df_convp branch*: df_gru
// (linear_in + 2× LBR=1 GRU, no linear_out), df_skip + Add, α head,
// and df_out + Tanh. Returns three intermediates so each gate
// (skip_add, alpha, df_out_tanh) can be validated against the
// matching golden. df_convp's contribution to `coefs` is added in
// the next slice — empirically zero on this synthetic input
// (Relu clamps the whole branch to 0), so reshape(df_out_tanh) is
// numerically equal to `coefs` for the existing golden file.
//
//   out_skip_add    : (1, T, 256)
//   out_alpha       : (1, T, 1)
//   out_df_out_tanh : (1, T, 960)  -- caller can reshape to (1, T, 96, 10)
int dfn_debug_df_dec_partial(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                              float* out_skip_add, float* out_alpha, float* out_df_out_tanh);

// Drive the entire DF decoder end-to-end through the final
// `coefs` + `α` outputs (df_convp + 2× GRU + Add'd Tanh head).
// Output:
//   out_coefs : (1, T, 96, 10) — matches `dfn.golden.df_dec.coefs`.
//   out_alpha : (1, T, 1)      — matches `dfn.golden.df_dec.alpha`.
int dfn_debug_df_dec(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                      float* out_coefs, float* out_alpha);

// Batch-mode equivalent of streaming `predict_dfn3`: run encoder +
// erb_dec + df_dec for T frames in a single graph. GRU hiddens
// zero-init; no `c0_history` (uses the no-history fallback). Output:
//   out_mask  : (T, 32)
//   out_coefs : (T, 96, 10)  flat
//   out_alpha : (T)
// Used by `dfn-denoise-wav --batch` to A/B against upstream ONNX
// without streaming-state divergence.
int dfn_debug_full_predict(const struct dfn_model* m, const float* feat_erb, const float* feat_spec, int T,
                            float* out_mask, float* out_coefs, float* out_alpha);

#ifdef __cplusplus
}
#endif
