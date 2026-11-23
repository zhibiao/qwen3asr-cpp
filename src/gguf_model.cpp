#include "gguf_model.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <utility>

#include "ggml.h"
#include "gguf.h"

#include "errors.h"

namespace qwen3_asr {
namespace {

constexpr std::int32_t kGgufFormatVersion = 1;
constexpr const char * kGgufArchitecture = "qwen3_asr";
constexpr const char * kTokenizerModel = "gpt2";
constexpr const char * kTokenizerPre = "qwen2";
constexpr const char * kPromptVersion =
    "qwen3_asr_transcription_v1";
constexpr const char * kWeightsLayout =
    "huggingface_safetensors_v1";
constexpr const char * kWeightsDtype = "BF16";

class GgufContextGuard {
public:
    explicit GgufContextGuard(gguf_context * context)
        : context_(context) {}

    ~GgufContextGuard() {
        if (context_ != nullptr) {
            gguf_free(context_);
        }
    }

    gguf_context * get() const noexcept {
        return context_;
    }

private:
    gguf_context * context_;
};

std::int64_t require_key(
    const gguf_context * context,
    std::string_view key,
    gguf_type expected_type
) {
    const std::int64_t id =
        gguf_find_key(context, std::string(key).c_str());
    if (id < 0) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF key is missing: " + std::string(key)
        );
    }
    if (gguf_get_kv_type(context, id) != expected_type) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF key has the wrong type: " + std::string(key)
        );
    }
    return id;
}

std::int32_t read_i32(
    const gguf_context * context,
    std::string_view key
) {
    return gguf_get_val_i32(
        context,
        require_key(context, key, GGUF_TYPE_INT32)
    );
}

std::int32_t read_i32_or(
    const gguf_context * context,
    std::string_view key,
    std::int32_t default_value
) {
    const std::int64_t id =
        gguf_find_key(context, std::string(key).c_str());
    if (id < 0) {
        return default_value;
    }
    if (gguf_get_kv_type(context, id) != GGUF_TYPE_INT32) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF key has the wrong type: " + std::string(key)
        );
    }
    return gguf_get_val_i32(context, id);
}

std::int64_t read_i64(
    const gguf_context * context,
    std::string_view key
) {
    return gguf_get_val_i64(
        context,
        require_key(context, key, GGUF_TYPE_INT64)
    );
}

float read_f32(
    const gguf_context * context,
    std::string_view key
) {
    return gguf_get_val_f32(
        context,
        require_key(context, key, GGUF_TYPE_FLOAT32)
    );
}

bool read_bool(
    const gguf_context * context,
    std::string_view key
) {
    return gguf_get_val_bool(
        context,
        require_key(context, key, GGUF_TYPE_BOOL)
    );
}

std::string read_string(
    const gguf_context * context,
    std::string_view key
) {
    return gguf_get_val_str(
        context,
        require_key(context, key, GGUF_TYPE_STRING)
    );
}

std::int64_t require_array(
    const gguf_context * context,
    std::string_view key,
    gguf_type element_type
) {
    const std::int64_t id =
        require_key(context, key, GGUF_TYPE_ARRAY);
    if (gguf_get_arr_type(context, id) != element_type) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF array has the wrong element type: " +
                std::string(key)
        );
    }
    return id;
}

void require_tensor(
    const gguf_context * context,
    std::string_view tensor_name,
    ggml_type expected_type,
    std::initializer_list<std::int64_t> expected_ne
) {
    const std::int64_t tensor_id =
        gguf_find_tensor(context, std::string(tensor_name).c_str());
    if (tensor_id < 0) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF tensor is missing: " + std::string(tensor_name)
        );
    }
    if (expected_type != GGML_TYPE_COUNT &&
        gguf_get_tensor_type(context, tensor_id) != expected_type) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF tensor has the wrong dtype: " +
                std::string(tensor_name)
        );
    }

    const std::int64_t * actual_ne =
        gguf_get_tensor_ne(context, tensor_id);
    std::size_t dimension = 0;
    for (const std::int64_t expected : expected_ne) {
        if (actual_ne[dimension] != expected) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF tensor shape mismatch: " +
                    std::string(tensor_name)
            );
        }
        ++dimension;
    }
    for (; dimension < GGML_MAX_DIMS; ++dimension) {
        if (actual_ne[dimension] != 1) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF tensor rank mismatch: " +
                    std::string(tensor_name)
            );
        }
    }
}

std::vector<std::string> read_string_array(
    const gguf_context * context,
    std::string_view key
) {
    const std::int64_t id = require_array(
        context,
        key,
        GGUF_TYPE_STRING
    );
    const std::size_t count = gguf_get_arr_n(context, id);
    std::vector<std::string> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const char * value = gguf_get_arr_str(context, id, index);
        if (value == nullptr) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF string array contains null: " +
                    std::string(key)
            );
        }
        values.emplace_back(value);
    }
    return values;
}

std::vector<std::int32_t> read_i32_array(
    const gguf_context * context,
    std::string_view key
) {
    const std::int64_t id = require_array(
        context,
        key,
        GGUF_TYPE_INT32
    );
    const std::size_t count = gguf_get_arr_n(context, id);
    const void * data = gguf_get_arr_data(context, id);
    if (count > 0 && data == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF int32 array data is null: " + std::string(key)
        );
    }
    std::vector<std::int32_t> values(count);
    if (count > 0) {
        std::memcpy(
            values.data(),
            data,
            count * sizeof(std::int32_t)
        );
    }
    return values;
}

std::vector<GgufTensorDescriptor> collect_tensor_descriptors(
    const gguf_context * context,
    std::string_view name_prefix
) {
    std::vector<GgufTensorDescriptor> descriptors;
    const std::int64_t tensor_count = gguf_get_n_tensors(context);
    const std::size_t data_offset = gguf_get_data_offset(context);
    descriptors.reserve(static_cast<std::size_t>(tensor_count));
    for (std::int64_t index = 0; index < tensor_count; ++index) {
        const char * raw_name = gguf_get_tensor_name(context, index);
        if (raw_name == nullptr) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF contains an unnamed tensor"
            );
        }
        const std::string_view name = raw_name;
        if (!name_prefix.empty() &&
            name.find(name_prefix) != 0) {
            continue;
        }
        GgufTensorDescriptor descriptor;
        descriptor.name = name;
        descriptor.type = static_cast<std::int32_t>(
            gguf_get_tensor_type(context, index)
        );
        const int64_t * ne = gguf_get_tensor_ne(context, index);
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            descriptor.ne[dimension] = ne[dimension];
        }
        descriptor.file_offset =
            static_cast<std::uint64_t>(data_offset) +
            gguf_get_tensor_offset(context, index);
        descriptor.byte_count = gguf_get_tensor_size(context, index);
        descriptors.push_back(std::move(descriptor));
    }
    return descriptors;
}

std::vector<float> read_f32_tensor(
    const std::filesystem::path & path,
    const gguf_context * context,
    std::string_view tensor_name,
    const int64_t * expected_ne
) {
    const std::int64_t tensor_id =
        gguf_find_tensor(context, std::string(tensor_name).c_str());
    if (tensor_id < 0) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF tensor is missing: " + std::string(tensor_name)
        );
    }
    if (gguf_get_tensor_type(context, tensor_id) != GGML_TYPE_F32) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF tensor must be F32: " + std::string(tensor_name)
        );
    }

    const int64_t * ne =
        gguf_get_tensor_ne(context, tensor_id);
    for (int index = 0; index < GGML_MAX_DIMS; ++index) {
        if (ne[index] != expected_ne[index]) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF tensor shape mismatch: " +
                    std::string(tensor_name)
            );
        }
    }

    const size_t byte_count =
        gguf_get_tensor_size(context, tensor_id);
    const size_t value_count = byte_count / sizeof(float);
    const size_t file_offset =
        gguf_get_data_offset(context) +
        gguf_get_tensor_offset(context, tensor_id);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to open GGUF data: " + path.string()
        );
    }
    input.seekg(
        static_cast<std::streamoff>(file_offset),
        std::ios::beg
    );
    if (!input) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to seek GGUF tensor data"
        );
    }

    std::vector<float> values(value_count);
    input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(byte_count)
    );
    if (!input) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to read GGUF tensor data"
        );
    }
    return values;
}

}  // namespace

GgufModel load_qwen3_asr_gguf(
    const std::filesystem::path & path
) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    GgufContextGuard context_guard(
        gguf_init_from_file(path.string().c_str(), params)
    );
    gguf_context * context = context_guard.get();
    if (context == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to open GGUF file: " + path.string()
        );
    }

    GgufModel model;
    const std::string general_architecture = read_string(
        context,
        "general.architecture"
    );
    if (general_architecture != kGgufArchitecture) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unexpected GGUF architecture: " + general_architecture
        );
    }
    model.format_version = read_i32(
        context,
        "qwen3_asr.format_version"
    );
    if (model.format_version != kGgufFormatVersion) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported Qwen3-ASR GGUF format version"
        );
    }

    model.gf_name = read_string(context, "general.name");
    model.model_family = read_string(
        context,
        "qwen3_asr.model.family"
    );
    model.model_type = read_string(
        context,
        "qwen3_asr.model.model_type"
    );
    model.architecture = read_string(
        context,
        "qwen3_asr.model.architecture"
    );
    model.supported_languages = read_string_array(
        context,
        "qwen3_asr.model.supported_languages"
    );
    model.weight_layout = read_string(
        context,
        "qwen3_asr.weights.layout"
    );
    model.weight_dtype = read_string(
        context,
        "qwen3_asr.weights.dtype"
    );
    model.weight_tensor_count = read_i32(
        context,
        "qwen3_asr.weights.tensor_count"
    );
    model.audio_weight_tensor_count = read_i32(
        context,
        "qwen3_asr.weights.audio_tensor_count"
    );
    model.text_weight_tensor_count = read_i32(
        context,
        "qwen3_asr.weights.text_tensor_count"
    );
    model.lm_head_weight_tensor_count = read_i32(
        context,
        "qwen3_asr.weights.lm_head_tensor_count"
    );
    model.weight_byte_count = read_i64(
        context,
        "qwen3_asr.weights.byte_count"
    );
    if (model.weight_layout != kWeightsLayout ||
        model.weight_dtype != kWeightsDtype) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported GGUF weight layout or dtype"
        );
    }
    if (model.weight_tensor_count <= 0 ||
        model.audio_weight_tensor_count <= 0 ||
        model.text_weight_tensor_count <= 0 ||
        model.lm_head_weight_tensor_count != 1 ||
        model.audio_weight_tensor_count +
                model.text_weight_tensor_count +
                model.lm_head_weight_tensor_count !=
            model.weight_tensor_count ||
        model.weight_byte_count <= 0) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF weight manifest is invalid"
        );
    }
    if (gguf_get_n_tensors(context) !=
        model.weight_tensor_count + 1) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF tensor count does not match weight manifest"
        );
    }

    model.audio.d_model = read_i32(context, "qwen3_asr.audio.d_model");
    model.audio.encoder_layers = read_i32(
        context,
        "qwen3_asr.audio.encoder_layers"
    );
    model.audio.encoder_attention_heads = read_i32(
        context,
        "qwen3_asr.audio.encoder_attention_heads"
    );
    model.audio.encoder_ffn_dim = read_i32(
        context,
        "qwen3_asr.audio.encoder_ffn_dim"
    );
    model.audio.num_mel_bins = read_i32(
        context,
        "qwen3_asr.audio.num_mel_bins"
    );
    model.audio.n_window = read_i32(
        context,
        "qwen3_asr.audio.n_window"
    );
    model.audio.n_window_infer = read_i32(
        context,
        "qwen3_asr.audio.n_window_infer"
    );
    model.audio.max_source_positions = read_i32(
        context,
        "qwen3_asr.audio.max_source_positions"
    );
    model.audio.output_dim = read_i32(
        context,
        "qwen3_asr.audio.output_dim"
    );

    model.text.hidden_size = read_i32(
        context,
        "qwen3_asr.text.hidden_size"
    );
    model.text.layers = read_i32(
        context,
        "qwen3_asr.text.layers"
    );
    model.text.attention_heads = read_i32(
        context,
        "qwen3_asr.text.attention_heads"
    );
    model.text.kv_heads = read_i32(
        context,
        "qwen3_asr.text.kv_heads"
    );
    model.text.head_dim = read_i32(
        context,
        "qwen3_asr.text.head_dim"
    );
    model.text.vocab_size = read_i32(
        context,
        "qwen3_asr.text.vocab_size"
    );
    model.text.max_position_embeddings = read_i32(
        context,
        "qwen3_asr.text.max_position_embeddings"
    );
    model.text.num_ctx = read_i32_or(
        context,
        "qwen3_asr.text.num_ctx",
        Qwen3LmConfig{}.num_ctx
    );
    model.text.rope_theta = read_f32(
        context,
        "qwen3_asr.text.rope_theta"
    );

    require_tensor(
        context,
        "thinker.audio_tower.conv2d1.weight",
        GGML_TYPE_BF16,
        {3, 3, 1, 480}
    );
    require_tensor(
        context,
        "thinker.audio_tower.conv_out.weight",
        GGML_TYPE_BF16,
        {7680, 896}
    );
    require_tensor(
        context,
        "thinker.audio_tower.layers.17.fc2.weight",
        GGML_TYPE_BF16,
        {3584, 896}
    );
    require_tensor(
        context,
        "thinker.audio_tower.proj2.weight",
        GGML_TYPE_BF16,
        {896, 1024}
    );
    require_tensor(
        context,
        "thinker.model.embed_tokens.weight",
        GGML_TYPE_BF16,
        {1024, 151936}
    );
    require_tensor(
        context,
        "thinker.model.layers.27.self_attn.q_proj.weight",
        GGML_TYPE_BF16,
        {1024, 2048}
    );
    require_tensor(
        context,
        "thinker.model.layers.27.mlp.down_proj.weight",
        GGML_TYPE_BF16,
        {3072, 1024}
    );
    require_tensor(
        context,
        "thinker.lm_head.weight",
        GGML_TYPE_BF16,
        {1024, 151936}
    );

    model.preprocessor.sample_rate = read_i32(
        context,
        "qwen3_asr.preprocessor.sample_rate"
    );
    if (model.preprocessor.sample_rate !=
        16000) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported sample rate in GGUF"
        );
    }
    model.preprocessor.feature_size = read_i32(
        context,
        "qwen3_asr.preprocessor.feature_size"
    );
    model.preprocessor.n_fft = read_i32(
        context,
        "qwen3_asr.preprocessor.n_fft"
    );
    model.preprocessor.hop_length = read_i32(
        context,
        "qwen3_asr.preprocessor.hop_length"
    );
    model.preprocessor.chunk_length = read_i32(
        context,
        "qwen3_asr.preprocessor.chunk_length"
    );
    model.preprocessor.n_samples = read_i32(
        context,
        "qwen3_asr.preprocessor.n_samples"
    );
    model.preprocessor.nb_max_frames = read_i32(
        context,
        "qwen3_asr.preprocessor.nb_max_frames"
    );
    model.preprocessor.dither = read_f32(
        context,
        "qwen3_asr.preprocessor.dither"
    );
    model.preprocessor.padding_value = read_f32(
        context,
        "qwen3_asr.preprocessor.padding_value"
    );
    model.preprocessor.mel_min_frequency = read_f32(
        context,
        "qwen3_asr.preprocessor.mel_min_frequency"
    );
    model.preprocessor.mel_max_frequency = read_f32(
        context,
        "qwen3_asr.preprocessor.mel_max_frequency"
    );
    model.preprocessor.mel_scale = read_string(
        context,
        "qwen3_asr.preprocessor.mel_scale"
    );
    model.preprocessor.mel_norm = read_string(
        context,
        "qwen3_asr.preprocessor.mel_norm"
    );
    model.preprocessor.return_attention_mask = read_bool(
        context,
        "qwen3_asr.preprocessor.return_attention_mask"
    );

    model.frontend_config.feature_size =
        model.preprocessor.feature_size;
    model.frontend_config.sampling_rate =
        model.preprocessor.sample_rate;
    model.frontend_config.n_fft = model.preprocessor.n_fft;
    model.frontend_config.hop_length =
        model.preprocessor.hop_length;
    model.frontend_config.dither = model.preprocessor.dither;
    model.frontend_config.mel_min_frequency =
        model.preprocessor.mel_min_frequency;
    model.frontend_config.mel_max_frequency =
        model.preprocessor.mel_max_frequency;

    const std::vector<std::string> tokens = read_string_array(
        context,
        "tokenizer.ggml.tokens"
    );
    const std::vector<std::string> merges = read_string_array(
        context,
        "tokenizer.ggml.merges"
    );
    model.tokenizer_model = read_string(
        context,
        "tokenizer.ggml.model"
    );
    model.tokenizer_pre = read_string(
        context,
        "tokenizer.ggml.pre"
    );
    if (model.tokenizer_model != kTokenizerModel ||
        model.tokenizer_pre != kTokenizerPre) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported tokenizer rules in GGUF"
        );
    }
    model.tokenizer_model_max_length = read_i32(
        context,
        "qwen3_asr.tokenizer.model_max_length"
    );

    model.tokenizer_data.vocab.reserve(tokens.size());
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        if (tokens[id].empty()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF tokenizer contains an empty token"
            );
        }
        model.tokenizer_data.vocab.emplace(
            tokens[id],
            static_cast<std::int32_t>(id)
        );
    }

    model.tokenizer_data.merges.reserve(merges.size());
    for (const std::string & merge : merges) {
        const std::size_t separator = merge.find(' ');
        if (separator == std::string::npos ||
            separator + 1 >= merge.size()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "invalid GGUF tokenizer merge"
            );
        }
        model.tokenizer_data.merges.emplace_back(
            merge.substr(0, separator),
            merge.substr(separator + 1)
        );
    }

    model.tokenizer_data.eos_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.eos_token_id"
    );
    model.tokenizer_data.pad_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.pad_token_id"
    );
    model.tokenizer_data.im_start_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.im_start_token_id"
    );
    model.tokenizer_data.im_end_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.im_end_token_id"
    );
    model.tokenizer_data.audio_start_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.audio_start_token_id"
    );
    model.tokenizer_data.audio_pad_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.audio_pad_token_id"
    );
    model.tokenizer_data.audio_end_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.audio_end_token_id"
    );
    model.tokenizer_data.asr_text_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.asr_text_token_id"
    );

    const std::vector<std::int32_t> added_ids = read_i32_array(
        context,
        "qwen3_asr.tokenizer.added_token_ids"
    );
    const std::vector<std::int32_t> added_special = read_i32_array(
        context,
        "qwen3_asr.tokenizer.added_token_special"
    );
    if (added_ids.size() != added_special.size()) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF added-token arrays have different sizes"
        );
    }
    model.tokenizer_data.added_tokens.reserve(added_ids.size());
    for (std::size_t index = 0; index < added_ids.size(); ++index) {
        const std::int32_t id = added_ids[index];
        if (id < 0 ||
            static_cast<std::size_t>(id) >= tokens.size()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "GGUF added token ID is out of range"
            );
        }
        model.tokenizer_data.added_tokens.push_back(
            TextTokenizerAddedToken{
                tokens[static_cast<std::size_t>(id)],
                id,
                added_special[index] != 0,
            }
        );
    }

    model.generation_eos_token_ids = read_i32_array(
        context,
        "qwen3_asr.generation.eos_token_ids"
    );
    model.prompt_version = read_string(
        context,
        "qwen3_asr.prompt.version"
    );
    if (model.prompt_version != kPromptVersion) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported prompt version in GGUF"
        );
    }

    const std::int64_t tensor_id = gguf_find_tensor(
        context,
        "preprocessor.mel_filters"
    );
    if (tensor_id < 0) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF Mel filter tensor is missing"
        );
    }
    const int64_t * ne = gguf_get_tensor_ne(context, tensor_id);
    if (ne[0] != model.frontend_config.feature_size ||
        ne[1] != 1 + model.frontend_config.n_fft / 2 ||
        ne[2] != 1 ||
        ne[3] != 1) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF Mel filter shape does not match preprocessor config"
        );
    }
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne{
        ne[0], ne[1], 1, 1,
    };
    model.mel_filters.n_frequency_bins =
        static_cast<int>(ne[1]);
    model.mel_filters.n_mel_bins = static_cast<int>(ne[0]);
    model.mel_filters.data = read_f32_tensor(
        path,
        context,
        "preprocessor.mel_filters",
        expected_ne.data()
    );

    if (model.mel_filters.data.size() !=
        static_cast<std::size_t>(
            model.mel_filters.n_frequency_bins
        ) *
        static_cast<std::size_t>(model.mel_filters.n_mel_bins)) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "GGUF Mel filter data size is invalid"
        );
    }

    model.path = path;
    model.part = "complete";
    model.tensors = collect_tensor_descriptors(context, "");
    return model;
}

namespace {

constexpr std::int32_t kSplitFormatVersion = 2;
constexpr int kEncoderWeightTensorCount = 301;
constexpr int kLmTextTensorCount = 310;
constexpr int kLmHeadTensorCount = 1;
constexpr int kLmWeightTensorCount =
    kLmTextTensorCount + kLmHeadTensorCount;

struct SplitCommonInfo {
    std::int32_t format_version = 0;
    std::string gf_name;
    std::string model_family;
    std::string model_type;
    std::string architecture;
    std::vector<std::string> supported_languages;
    std::string bundle_id;
    std::string part;
    std::string weight_layout;
    std::string weight_dtype;
    int weight_tensor_count = 0;
    std::int64_t weight_byte_count = 0;
};

ggml_type parse_weight_type(std::string_view dtype) {
    if (dtype == "F32") {
        return GGML_TYPE_F32;
    }
    if (dtype == "F16") {
        return GGML_TYPE_F16;
    }
    if (dtype == "BF16") {
        return GGML_TYPE_BF16;
    }
    if (dtype == "Q8_0") {
        return GGML_TYPE_Q8_0;
    }
    throw Error(
        ErrorCode::ModelLoadFailed,
        "unsupported split GGUF weight dtype: " + std::string(dtype)
    );
}

bool tensor_belongs_to_part(std::string_view name, std::string_view part) {
    if (part == "encoder") {
        return name.find("thinker.audio_tower.") == 0;
    }
    if (part == "lm") {
        return name.find("thinker.model.") == 0 ||
               name.find("thinker.lm_head.") == 0;
    }
    return false;
}

void verify_split_part(
    const gguf_context * context,
    const SplitCommonInfo & info
) {
    if (info.part != "encoder" && info.part != "lm") {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported split GGUF part"
        );
    }
    if (info.weight_layout != kWeightsLayout) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported split GGUF weight layout"
        );
    }

    const int expected_weight_count =
        info.part == "encoder" ?
        kEncoderWeightTensorCount : kLmWeightTensorCount;
    if (info.weight_tensor_count != expected_weight_count) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF weight tensor count is invalid"
        );
    }

    const std::vector<std::string> type_names = read_string_array(
        context,
        "qwen3_asr.weights.type_names"
    );
    const std::vector<std::int32_t> type_counts = read_i32_array(
        context,
        "qwen3_asr.weights.type_counts"
    );
    if (type_names.empty() ||
        type_names.size() != type_counts.size()) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF weight dtype manifest is invalid"
        );
    }

    std::vector<ggml_type> declared_types;
    declared_types.reserve(type_names.size());
    std::vector<std::int64_t> observed_counts(type_names.size(), 0);
    std::int64_t declared_count = 0;
    bool has_dtype = false;
    for (std::size_t index = 0; index < type_names.size(); ++index) {
        if (type_counts[index] <= 0) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "split GGUF weight dtype count is invalid"
            );
        }
        declared_types.push_back(parse_weight_type(type_names[index]));
        declared_count += type_counts[index];
        has_dtype = has_dtype || type_names[index] == info.weight_dtype;
    }
    if (!has_dtype || declared_count != info.weight_tensor_count) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF weight dtype totals are invalid"
        );
    }

    int actual_weight_count = 0;
    std::int64_t actual_byte_count = 0;
    const std::int64_t tensor_count = gguf_get_n_tensors(context);
    for (std::int64_t index = 0; index < tensor_count; ++index) {
        const char * raw_name = gguf_get_tensor_name(context, index);
        if (raw_name == nullptr) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "split GGUF contains an unnamed tensor"
            );
        }
        const std::string_view name = raw_name;
        if (name == "preprocessor.mel_filters") {
            if (info.part != "encoder" ||
                gguf_get_tensor_type(context, index) != GGML_TYPE_F32) {
                throw Error(
                    ErrorCode::ModelLoadFailed,
                    "split GGUF Mel tensor is invalid"
                );
            }
            continue;
        }
        if (!tensor_belongs_to_part(name, info.part)) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "split GGUF contains a tensor from the wrong part: " +
                    std::string(name)
            );
        }
        const ggml_type actual_type =
            gguf_get_tensor_type(context, index);
        const auto declared = std::find(
            declared_types.begin(),
            declared_types.end(),
            actual_type
        );
        if (declared == declared_types.end()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "split GGUF tensor has undeclared dtype: " +
                    std::string(name)
            );
        }
        ++observed_counts[static_cast<std::size_t>(
            declared - declared_types.begin()
        )];
        ++actual_weight_count;
        actual_byte_count += gguf_get_tensor_size(context, index);
    }
    const bool type_counts_match = std::equal(
        observed_counts.begin(),
        observed_counts.end(),
        type_counts.begin(),
        type_counts.end(),
        [](std::int64_t actual, std::int32_t declared) {
            return actual == declared;
        }
    );
    if (actual_weight_count != info.weight_tensor_count ||
        actual_byte_count != info.weight_byte_count ||
        !type_counts_match ||
        tensor_count !=
            info.weight_tensor_count +
                (info.part == "encoder" ? 1 : 0)) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF tensor totals do not match metadata"
        );
    }

    if (info.part == "encoder") {
        require_tensor(
            context,
            "thinker.audio_tower.conv2d1.weight",
            GGML_TYPE_COUNT,
            {3, 3, 1, 480}
        );
        require_tensor(
            context,
            "thinker.audio_tower.conv_out.weight",
            GGML_TYPE_COUNT,
            {7680, 896}
        );
        require_tensor(
            context,
            "thinker.audio_tower.layers.17.fc2.weight",
            GGML_TYPE_COUNT,
            {3584, 896}
        );
        require_tensor(
            context,
            "thinker.audio_tower.proj2.weight",
            GGML_TYPE_COUNT,
            {896, 1024}
        );
    } else {
        require_tensor(
            context,
            "thinker.model.embed_tokens.weight",
            GGML_TYPE_COUNT,
            {1024, 151936}
        );
        require_tensor(
            context,
            "thinker.model.layers.27.self_attn.q_proj.weight",
            GGML_TYPE_COUNT,
            {1024, 2048}
        );
        require_tensor(
            context,
            "thinker.model.layers.27.mlp.down_proj.weight",
            GGML_TYPE_COUNT,
            {3072, 1024}
        );
        require_tensor(
            context,
            "thinker.lm_head.weight",
            GGML_TYPE_COUNT,
            {1024, 151936}
        );
    }
}

SplitCommonInfo read_split_common(
    const gguf_context * context,
    std::string_view expected_part
) {
    const std::string architecture = read_string(
        context,
        "general.architecture"
    );
    if (architecture != kGgufArchitecture) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unexpected split GGUF architecture: " + architecture
        );
    }

    SplitCommonInfo info;
    info.format_version = read_i32(
        context,
        "qwen3_asr.format_version"
    );
    if (info.format_version != kSplitFormatVersion) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported split GGUF format version"
        );
    }
    info.gf_name = read_string(context, "general.name");
    info.model_family = read_string(
        context,
        "qwen3_asr.model.family"
    );
    info.model_type = read_string(
        context,
        "qwen3_asr.model.model_type"
    );
    info.architecture = read_string(
        context,
        "qwen3_asr.model.architecture"
    );
    info.supported_languages = read_string_array(
        context,
        "qwen3_asr.model.supported_languages"
    );
    info.bundle_id = read_string(context, "qwen3_asr.bundle.id");
    info.part = read_string(context, "qwen3_asr.part");
    info.weight_layout = read_string(
        context,
        "qwen3_asr.weights.layout"
    );
    info.weight_dtype = read_string(
        context,
        "qwen3_asr.weights.dtype"
    );
    info.weight_tensor_count = read_i32(
        context,
        "qwen3_asr.weights.tensor_count"
    );
    info.weight_byte_count = read_i64(
        context,
        "qwen3_asr.weights.byte_count"
    );
    if (info.part != expected_part || info.bundle_id.empty()) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF part or bundle ID is invalid"
        );
    }
    verify_split_part(context, info);
    return info;
}

void apply_split_common(
    GgufModel & model,
    const SplitCommonInfo & info
) {
    model.format_version = info.format_version;
    model.gf_name = info.gf_name;
    model.model_family = info.model_family;
    model.model_type = info.model_type;
    model.architecture = info.architecture;
    model.supported_languages = info.supported_languages;
    model.bundle_id = info.bundle_id;
    model.weight_layout = info.weight_layout;
}

void read_split_audio(
    const gguf_context * context,
    GgufModel & model
) {
    model.audio.d_model = read_i32(context, "qwen3_asr.audio.d_model");
    model.audio.encoder_layers = read_i32(
        context,
        "qwen3_asr.audio.encoder_layers"
    );
    model.audio.encoder_attention_heads = read_i32(
        context,
        "qwen3_asr.audio.encoder_attention_heads"
    );
    model.audio.encoder_ffn_dim = read_i32(
        context,
        "qwen3_asr.audio.encoder_ffn_dim"
    );
    model.audio.num_mel_bins = read_i32(
        context,
        "qwen3_asr.audio.num_mel_bins"
    );
    model.audio.n_window = read_i32(
        context,
        "qwen3_asr.audio.n_window"
    );
    model.audio.n_window_infer = read_i32(
        context,
        "qwen3_asr.audio.n_window_infer"
    );
    model.audio.max_source_positions = read_i32(
        context,
        "qwen3_asr.audio.max_source_positions"
    );
    model.audio.output_dim = read_i32(
        context,
        "qwen3_asr.audio.output_dim"
    );
    model.audio.downsample_hidden_size = read_i32(
        context,
        "qwen3_asr.audio.downsample_hidden_size"
    );
    model.audio.conv_chunksize = read_i32(
        context,
        "qwen3_asr.audio.conv_chunksize"
    );
    model.audio.activation_function = read_string(
        context,
        "qwen3_asr.audio.activation_function"
    );
    model.audio.layer_norm_eps = read_f32(
        context,
        "qwen3_asr.audio.layer_norm_eps"
    );

    model.preprocessor.sample_rate = read_i32(
        context,
        "qwen3_asr.preprocessor.sample_rate"
    );
    model.preprocessor.feature_size = read_i32(
        context,
        "qwen3_asr.preprocessor.feature_size"
    );
    model.preprocessor.n_fft = read_i32(
        context,
        "qwen3_asr.preprocessor.n_fft"
    );
    model.preprocessor.hop_length = read_i32(
        context,
        "qwen3_asr.preprocessor.hop_length"
    );
    model.preprocessor.chunk_length = read_i32(
        context,
        "qwen3_asr.preprocessor.chunk_length"
    );
    model.preprocessor.n_samples = read_i32(
        context,
        "qwen3_asr.preprocessor.n_samples"
    );
    model.preprocessor.nb_max_frames = read_i32(
        context,
        "qwen3_asr.preprocessor.nb_max_frames"
    );
    model.preprocessor.dither = read_f32(
        context,
        "qwen3_asr.preprocessor.dither"
    );
    model.preprocessor.padding_value = read_f32(
        context,
        "qwen3_asr.preprocessor.padding_value"
    );
    model.preprocessor.mel_min_frequency = read_f32(
        context,
        "qwen3_asr.preprocessor.mel_min_frequency"
    );
    model.preprocessor.mel_max_frequency = read_f32(
        context,
        "qwen3_asr.preprocessor.mel_max_frequency"
    );
    model.preprocessor.mel_scale = read_string(
        context,
        "qwen3_asr.preprocessor.mel_scale"
    );
    model.preprocessor.mel_norm = read_string(
        context,
        "qwen3_asr.preprocessor.mel_norm"
    );
    model.preprocessor.return_attention_mask = read_bool(
        context,
        "qwen3_asr.preprocessor.return_attention_mask"
    );
    if (model.preprocessor.sample_rate != 16000) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported split GGUF sample rate"
        );
    }
    model.frontend_config.feature_size =
        model.preprocessor.feature_size;
    model.frontend_config.sampling_rate =
        model.preprocessor.sample_rate;
    model.frontend_config.n_fft = model.preprocessor.n_fft;
    model.frontend_config.hop_length =
        model.preprocessor.hop_length;
    model.frontend_config.dither = model.preprocessor.dither;
    model.frontend_config.mel_min_frequency =
        model.preprocessor.mel_min_frequency;
    model.frontend_config.mel_max_frequency =
        model.preprocessor.mel_max_frequency;
}

void read_split_mel(
    const std::filesystem::path & path,
    const gguf_context * context,
    GgufModel & model
) {
    const std::int64_t tensor_id = gguf_find_tensor(
        context,
        "preprocessor.mel_filters"
    );
    if (tensor_id < 0) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF Mel filter tensor is missing"
        );
    }
    const int64_t * ne = gguf_get_tensor_ne(context, tensor_id);
    if (ne[0] != model.frontend_config.feature_size ||
        ne[1] != 1 + model.frontend_config.n_fft / 2 ||
        ne[2] != 1 ||
        ne[3] != 1) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF Mel filter shape is invalid"
        );
    }
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne{
        ne[0], ne[1], 1, 1,
    };
    model.mel_filters.n_frequency_bins = static_cast<int>(ne[1]);
    model.mel_filters.n_mel_bins = static_cast<int>(ne[0]);
    model.mel_filters.data = read_f32_tensor(
        path,
        context,
        "preprocessor.mel_filters",
        expected_ne.data()
    );
}

void read_split_lm(
    const gguf_context * context,
    GgufModel & model
) {
    model.text.hidden_size = read_i32(
        context,
        "qwen3_asr.text.hidden_size"
    );
    model.text.layers = read_i32(
        context,
        "qwen3_asr.text.layers"
    );
    model.text.attention_heads = read_i32(
        context,
        "qwen3_asr.text.attention_heads"
    );
    model.text.kv_heads = read_i32(
        context,
        "qwen3_asr.text.kv_heads"
    );
    model.text.head_dim = read_i32(
        context,
        "qwen3_asr.text.head_dim"
    );
    model.text.vocab_size = read_i32(
        context,
        "qwen3_asr.text.vocab_size"
    );
    model.text.max_position_embeddings = read_i32(
        context,
        "qwen3_asr.text.max_position_embeddings"
    );
    model.text.num_ctx = read_i32_or(
        context,
        "qwen3_asr.text.num_ctx",
        Qwen3LmConfig{}.num_ctx
    );
    model.text.rope_theta = read_f32(
        context,
        "qwen3_asr.text.rope_theta"
    );
    model.text.rms_norm_eps = read_f32(
        context,
        "qwen3_asr.text.rms_norm_eps"
    );
    model.text.intermediate_size = read_i32(
        context,
        "qwen3_asr.text.intermediate_size"
    );
    model.text.tie_word_embeddings = read_bool(
        context,
        "qwen3_asr.text.tie_word_embeddings"
    );
    model.text.mrope_section = read_i32_array(
        context,
        "qwen3_asr.text.mrope_section"
    );

    const std::vector<std::string> tokens = read_string_array(
        context,
        "tokenizer.ggml.tokens"
    );
    const std::vector<std::string> merges = read_string_array(
        context,
        "tokenizer.ggml.merges"
    );
    model.tokenizer_model = read_string(
        context,
        "tokenizer.ggml.model"
    );
    model.tokenizer_pre = read_string(
        context,
        "tokenizer.ggml.pre"
    );
    if (model.tokenizer_model != kTokenizerModel ||
        model.tokenizer_pre != kTokenizerPre) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported split GGUF tokenizer rules"
        );
    }
    model.tokenizer_model_max_length = read_i32(
        context,
        "qwen3_asr.tokenizer.model_max_length"
    );
    model.tokenizer_data.vocab.reserve(tokens.size());
    for (std::size_t id = 0; id < tokens.size(); ++id) {
        if (tokens[id].empty()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "split GGUF tokenizer contains an empty token"
            );
        }
        model.tokenizer_data.vocab.emplace(
            tokens[id],
            static_cast<std::int32_t>(id)
        );
    }
    model.tokenizer_data.merges.reserve(merges.size());
    for (const std::string & merge : merges) {
        const std::size_t separator = merge.find(' ');
        if (separator == std::string::npos ||
            separator + 1 >= merge.size()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "invalid split GGUF tokenizer merge"
            );
        }
        model.tokenizer_data.merges.emplace_back(
            merge.substr(0, separator),
            merge.substr(separator + 1)
        );
    }
    model.tokenizer_data.eos_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.eos_token_id"
    );
    model.tokenizer_data.pad_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.pad_token_id"
    );
    model.tokenizer_data.im_start_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.im_start_token_id"
    );
    model.tokenizer_data.im_end_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.im_end_token_id"
    );
    model.tokenizer_data.audio_start_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.audio_start_token_id"
    );
    model.tokenizer_data.audio_pad_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.audio_pad_token_id"
    );
    model.tokenizer_data.audio_end_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.audio_end_token_id"
    );
    model.tokenizer_data.asr_text_token_id = read_i32(
        context,
        "qwen3_asr.tokenizer.asr_text_token_id"
    );
    const std::vector<std::int32_t> added_ids = read_i32_array(
        context,
        "qwen3_asr.tokenizer.added_token_ids"
    );
    const std::vector<std::int32_t> added_special = read_i32_array(
        context,
        "qwen3_asr.tokenizer.added_token_special"
    );
    if (added_ids.size() != added_special.size()) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF added-token arrays have different sizes"
        );
    }
    model.tokenizer_data.added_tokens.reserve(added_ids.size());
    for (std::size_t index = 0; index < added_ids.size(); ++index) {
        const std::int32_t id = added_ids[index];
        if (id < 0 || static_cast<std::size_t>(id) >= tokens.size()) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "split GGUF added token ID is out of range"
            );
        }
        model.tokenizer_data.added_tokens.push_back(
            TextTokenizerAddedToken{
                tokens[static_cast<std::size_t>(id)],
                id,
                added_special[index] != 0,
            }
        );
    }

    model.generation_eos_token_ids = read_i32_array(
        context,
        "qwen3_asr.generation.eos_token_ids"
    );
    model.prompt_version = read_string(
        context,
        "qwen3_asr.prompt.version"
    );
    if (model.prompt_version != kPromptVersion) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "unsupported split GGUF prompt version"
        );
    }
}

void require_same_common(
    const SplitCommonInfo & encoder,
    const SplitCommonInfo & lm
) {
    if (encoder.bundle_id != lm.bundle_id ||
        encoder.gf_name != lm.gf_name ||
        encoder.model_family != lm.model_family ||
        encoder.model_type != lm.model_type ||
        encoder.architecture != lm.architecture ||
        encoder.supported_languages != lm.supported_languages) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "encoder and LM GGUF files do not belong to the same bundle"
        );
    }
}

}  // namespace

GgufModelPair load_qwen3_asr_gguf_pair(
    const std::filesystem::path & encoder_path,
    const std::filesystem::path & lm_path
) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    GgufContextGuard encoder_guard(
        gguf_init_from_file(encoder_path.string().c_str(), params)
    );
    GgufContextGuard lm_guard(
        gguf_init_from_file(lm_path.string().c_str(), params)
    );
    if (encoder_guard.get() == nullptr || lm_guard.get() == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to open split GGUF files"
        );
    }

    const SplitCommonInfo encoder_info = read_split_common(
        encoder_guard.get(),
        "encoder"
    );
    const SplitCommonInfo lm_info = read_split_common(
        lm_guard.get(),
        "lm"
    );
    require_same_common(encoder_info, lm_info);

    GgufModelPair pair;
    apply_split_common(pair.encoder, encoder_info);
    apply_split_common(pair.lm, lm_info);

    pair.encoder.path = encoder_path;
    pair.encoder.part = encoder_info.part;
    pair.encoder.weight_layout = encoder_info.weight_layout;
    pair.encoder.weight_dtype = encoder_info.weight_dtype;
    pair.encoder.weight_tensor_count =
        encoder_info.weight_tensor_count;
    pair.encoder.audio_weight_tensor_count =
        encoder_info.weight_tensor_count;
    pair.encoder.weight_byte_count = encoder_info.weight_byte_count;

    pair.lm.path = lm_path;
    pair.lm.part = lm_info.part;
    pair.lm.weight_layout = lm_info.weight_layout;
    pair.lm.weight_dtype = lm_info.weight_dtype;
    pair.lm.weight_tensor_count = lm_info.weight_tensor_count;
    pair.lm.text_weight_tensor_count = kLmTextTensorCount;
    pair.lm.lm_head_weight_tensor_count = kLmHeadTensorCount;
    pair.lm.weight_byte_count = lm_info.weight_byte_count;

    read_split_audio(encoder_guard.get(), pair.encoder);
    read_split_mel(encoder_path, encoder_guard.get(), pair.encoder);
    read_split_lm(lm_guard.get(), pair.lm);
    pair.encoder.tensors = collect_tensor_descriptors(
        encoder_guard.get(),
        ""
    );
    pair.lm.tensors = collect_tensor_descriptors(
        lm_guard.get(),
        ""
    );
    if (pair.encoder.audio.output_dim != pair.lm.text.hidden_size) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "split GGUF audio output and text hidden size differ"
        );
    }
    return pair;
}

}  // namespace qwen3_asr
