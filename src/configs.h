#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace qwen3_asr {

struct GgmlSessionConfig {
    std::string device = "CPU";
    std::string params;
    int graph_size = 4096;
};

struct Qwen3AsrConfig {
    std::filesystem::path encoder_gguf;
    std::filesystem::path lm_gguf;
    GgmlSessionConfig session;
    std::optional<int> text_num_ctx;
};

struct AudioFrontendConfig {
    int feature_size = 128;
    int sampling_rate = 16000;
    int n_fft = 400;
    int hop_length = 160;
    float dither = 0.0F;
    float mel_min_frequency = 0.0F;
    float mel_max_frequency = 8000.0F;
};

struct AutEncoderConfig {
    int d_model = 0;
    int encoder_layers = 0;
    int encoder_attention_heads = 0;
    int encoder_ffn_dim = 0;
    int num_mel_bins = 0;
    int n_window = 0;
    int n_window_infer = 0;
    int max_source_positions = 0;
    int output_dim = 0;
    int downsample_hidden_size = 0;
    int conv_chunksize = 0;
    std::string activation_function;
    float layer_norm_eps = 1.0e-5F;
};

struct Qwen3LmConfig {
    int hidden_size = 0;
    int layers = 0;
    int attention_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int max_position_embeddings = 0;
    int num_ctx = 4096;
    float rope_theta = 0.0F;
    float rms_norm_eps = 0.0F;
    int intermediate_size = 0;
    bool tie_word_embeddings = false;
    std::vector<int> mrope_section;
};

struct AudioFrontendPreprocessorConfig {
    int sample_rate = 0;
    int feature_size = 0;
    int n_fft = 0;
    int hop_length = 0;
    int chunk_length = 0;
    int n_samples = 0;
    int nb_max_frames = 0;
    float dither = 0.0F;
    float padding_value = 0.0F;
    float mel_min_frequency = 0.0F;
    float mel_max_frequency = 0.0F;
    std::string mel_scale;
    std::string mel_norm;
    bool return_attention_mask = false;
};

}  // namespace qwen3_asr
