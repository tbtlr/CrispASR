// dfn-denoise-wav.cpp — run our GGML DFN3 over a mono WAV.
//
// Mirrors the path the CLI's `--tts-postfilter` flag takes after the
// VibeVoice synthesis, but skips the TTS step: read a 16-bit mono WAV
// at any sample rate, resample to the DFN3 native rate, push through
// `dfn_stream_process`, write a 16-bit mono WAV at 48 kHz.
//
// Used to A/B against the upstream's `df.enhance.enhance()` (driven
// by the same ONNX weights) on real signals — see the matching
// Python reference at `/tmp/denoise_onnx_ref.py`.
//
// Built unconditionally when CRISPASR_DFN is on; binary lands at
// `build/bin/dfn-denoise-wav`.

#include "audio_resample.h"
#include "dfn.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Bare-minimum mono 16-bit WAV I/O — keeps the tool free of any
// external audio library so it stays portable across the
// xcframework / Windows / Linux configurations.
bool read_wav_16_mono(const char* path, std::vector<float>& out, int& sr) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return false;
    }
    char     riff[12];
    if (std::fread(riff, 1, 12, f) != 12 || std::memcmp(riff, "RIFF", 4) || std::memcmp(riff + 8, "WAVE", 4)) {
        std::fprintf(stderr, "%s: not a RIFF/WAVE file\n", path);
        std::fclose(f);
        return false;
    }
    int     channels = 0, bits = 0;
    int32_t data_size = 0;
    int32_t fmt_sr = 0;
    while (!std::feof(f)) {
        char     id[4];
        uint32_t sz;
        if (std::fread(id, 1, 4, f) != 4) break;
        if (std::fread(&sz, 4, 1, f) != 1) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            int16_t fmt_tag, ch_, bits_;
            int32_t sr_, byte_rate;
            int16_t block_align;
            std::fread(&fmt_tag, 2, 1, f);
            std::fread(&ch_, 2, 1, f);
            std::fread(&sr_, 4, 1, f);
            std::fread(&byte_rate, 4, 1, f);
            std::fread(&block_align, 2, 1, f);
            std::fread(&bits_, 2, 1, f);
            if (sz > 16) std::fseek(f, sz - 16, SEEK_CUR);
            channels = ch_;
            bits     = bits_;
            fmt_sr   = sr_;
            if (fmt_tag != 1 || bits_ != 16) {
                std::fprintf(stderr, "%s: only 16-bit PCM supported (fmt_tag=%d bits=%d)\n", path, fmt_tag, bits_);
                std::fclose(f);
                return false;
            }
        } else if (!std::memcmp(id, "data", 4)) {
            data_size = (int32_t)sz;
            std::vector<int16_t> samples(data_size / 2);
            if ((int32_t)std::fread(samples.data(), 1, data_size, f) != data_size) {
                std::fprintf(stderr, "%s: short read in data chunk\n", path);
                std::fclose(f);
                return false;
            }
            // Downmix to mono if needed.
            int n_frames = (int)samples.size() / channels;
            out.assign(n_frames, 0.0f);
            for (int i = 0; i < n_frames; ++i) {
                int32_t acc = 0;
                for (int c = 0; c < channels; ++c) acc += samples[i * channels + c];
                out[i] = (float)(acc / channels) / 32768.0f;
            }
            sr = fmt_sr;
            std::fclose(f);
            return true;
        } else {
            std::fseek(f, sz, SEEK_CUR);
        }
    }
    std::fprintf(stderr, "%s: no data chunk found\n", path);
    std::fclose(f);
    return false;
}

bool write_wav_16_mono(const char* path, const std::vector<float>& audio, int sr) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    int32_t data_size = (int32_t)audio.size() * 2;
    int32_t file_size = 36 + data_size;
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&file_size, 4, 1, f);
    std::fwrite("WAVEfmt ", 1, 8, f);
    int32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, f);
    int16_t fmt_tag = 1, channels = 1, bits = 16;
    std::fwrite(&fmt_tag, 2, 1, f);
    std::fwrite(&channels, 2, 1, f);
    std::fwrite(&sr, 4, 1, f);
    int32_t byte_rate   = sr * channels * (bits / 8);
    std::fwrite(&byte_rate, 4, 1, f);
    int16_t block_align = channels * (bits / 8);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_size, 4, 1, f);
    for (float v : audio) {
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        std::fwrite(&s, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <input.wav> <output.wav> <dfn-model.gguf> [warmup_frames]\n", argv[0]);
        return 2;
    }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    const char* gguf     = argv[3];
    int         warmup   = (argc > 4) ? std::atoi(argv[4]) : 0;

    std::vector<float> audio;
    int                sr_in = 0;
    if (!read_wav_16_mono(in_path, audio, sr_in)) return 3;
    std::fprintf(stderr, "  input: %zu samples @ %d Hz, %.3f s\n", audio.size(), sr_in, audio.size() / (double)sr_in);

    dfn_params p = dfn_default_params();
    p.verbosity  = 1;
    dfn_model*  m = dfn_model_load(gguf, p);
    if (!m) {
        std::fprintf(stderr, "failed to load %s\n", gguf);
        return 4;
    }
    const int sr_dfn = dfn_model_sample_rate(m);

    // 24 → 48 kHz upsample (linear + miniaudio's LPF — same as the CLI).
    audio_resampler* up = audio_resampler_create(sr_in, sr_dfn, 1, /*algo=*/1);
    dfn_stream*      st = dfn_stream_create(m);
    if (warmup > 0) {
        std::fprintf(stderr, "  warmup: %d frames\n", warmup);
        dfn_stream_warmup(st, warmup);
    }

    std::vector<float> out;
    out.reserve(audio.size() * sr_dfn / sr_in + 16384);

    const int             in_chunk = 4096;
    std::vector<float>    up_buf(in_chunk * 4 + 64);
    std::vector<float>    dfn_buf(up_buf.size() * 2);
    for (size_t i = 0; i < audio.size(); i += (size_t)in_chunk) {
        int n_in = (int)std::min((size_t)in_chunk, audio.size() - i);
        int in_used = 0, up_written = 0;
        audio_resampler_process(up, audio.data() + i, n_in, up_buf.data(), (int)up_buf.size(), &in_used, &up_written);
        int produced = 0;
        dfn_stream_process(st, up_buf.data(), up_written, dfn_buf.data(), (int)dfn_buf.size(), &produced,
                           /*final_chunk=*/false);
        out.insert(out.end(), dfn_buf.begin(), dfn_buf.begin() + produced);
    }
    // Drain.
    int in_used = 0, up_written = 0;
    audio_resampler_process(up, nullptr, 0, up_buf.data(), (int)up_buf.size(), &in_used, &up_written);
    int produced = 0;
    dfn_stream_process(st, up_buf.data(), up_written, dfn_buf.data(), (int)dfn_buf.size(), &produced,
                       /*final_chunk=*/true);
    out.insert(out.end(), dfn_buf.begin(), dfn_buf.begin() + produced);

    audio_resampler_free(up);
    dfn_stream_free(st);
    dfn_model_free(m);

    if (!write_wav_16_mono(out_path, out, sr_dfn)) {
        std::fprintf(stderr, "failed to write %s\n", out_path);
        return 5;
    }
    std::fprintf(stderr, "  output: %zu samples @ %d Hz, wrote %s\n", out.size(), sr_dfn, out_path);
    return 0;
}
