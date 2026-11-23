#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "audio_frontend.h"
#include "configs.h"
#include "text_tokenizer.h"

namespace qwen3_asr {

struct GgufTensorDescriptor {
    std::string name;
    std::int32_t type = 0;
    std::array<std::int64_t, 4> ne{1, 1, 1, 1};
    std::uint64_t file_offset = 0;
    std::uint64_t byte_count = 0;
};

struct GgufModel {
    int format_version = 0;
    std::filesystem::path path;
    std::string part;
    std::string gf_name;
    std::string model_family;
    std::string model_type;
    std::string architecture;
    std::vector<std::string> supported_languages;
    std::string bundle_id;
    std::string weight_layout;
    std::string weight_dtype;
    int weight_tensor_count = 0;
    int audio_weight_tensor_count = 0;
    int text_weight_tensor_count = 0;
    int lm_head_weight_tensor_count = 0;
    std::int64_t weight_byte_count = 0;
    std::vector<GgufTensorDescriptor> tensors;

    AutEncoderConfig audio;
    Qwen3LmConfig text;
    AudioFrontendPreprocessorConfig preprocessor;

    std::string tokenizer_model;
    std::string tokenizer_pre;
    int tokenizer_model_max_length = 0;
    std::vector<std::int32_t> generation_eos_token_ids;

    std::string prompt_version;

    AudioFrontendConfig frontend_config;
    MelFilterBank mel_filters;
    TextTokenizerData tokenizer_data;
};

struct GgufModelPair {
    GgufModel encoder;
    GgufModel lm;
};

GgufModel load_qwen3_asr_gguf(
    const std::filesystem::path & path
);

GgufModelPair load_qwen3_asr_gguf_pair(
    const std::filesystem::path & encoder_path,
    const std::filesystem::path & lm_path
);

}  // namespace qwen3_asr
