#include "aut_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "errors.h"

namespace qwen3_asr {
namespace {

constexpr const char * kAudioPrefix = "thinker.audio_tower.";
constexpr int kMaxGraphNodes = 4096;

std::vector<float> make_positional_embedding(
    int positions,
    int channels
) {
    if (positions <= 0 || channels <= 0 || channels % 2 != 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid Audio Encoder positional embedding shape"
        );
    }
    const int half = channels / 2;
    const double log_timescale_increment =
        std::log(10000.0) / static_cast<double>(half - 1);
    std::vector<double> inverse_timescales(
        static_cast<std::size_t>(half)
    );
    for (int index = 0; index < half; ++index) {
        inverse_timescales[static_cast<std::size_t>(index)] =
            std::exp(
                -log_timescale_increment *
                static_cast<double>(index)
            );
    }

    std::vector<float> values(
        static_cast<std::size_t>(positions) *
        static_cast<std::size_t>(channels)
    );
    for (int position = 0; position < positions; ++position) {
        float * row = values.data() +
            static_cast<std::size_t>(position) * channels;
        for (int index = 0; index < half; ++index) {
            const double scaled =
                static_cast<double>(position) *
                inverse_timescales[static_cast<std::size_t>(index)];
            row[index] = static_cast<float>(std::sin(scaled));
            row[half + index] = static_cast<float>(std::cos(scaled));
        }
    }
    return values;
}

int tensor_dimensions(const GgufTensorDescriptor & descriptor) {
    int dimensions = 1;
    for (int index = 1; index < 4; ++index) {
        if (descriptor.ne[index] > 1) {
            dimensions = index + 1;
        }
    }
    return dimensions;
}

void validate_tensor_type(const GgufTensorDescriptor & descriptor) {
    switch (static_cast<ggml_type>(descriptor.type)) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
            return;
        default:
            throw Error(
                ErrorCode::ModelLoadFailed,
                "unsupported Audio Encoder tensor dtype: " +
                    descriptor.name
            );
    }
}

int convolution_output_length(int length) {
    for (int stage = 0; stage < 3; ++stage) {
        length = (length + 1) / 2;
    }
    return length;
}

}  // namespace

AutEncoder::AutEncoder() = default;

AutEncoder::~AutEncoder() {
    unload();
}

void AutEncoder::load(
    const AutEncoderConfig & config,
    const GgufModel & gguf_model,
    GgmlSession & session
) {
    unload();
    load_weights(config, gguf_model, session);
}

void AutEncoder::load_weights(
    const AutEncoderConfig & config,
    const GgufModel & gguf_model,
    GgmlSession & session
) {
    if (config.encoder_layers <= 0 ||
        config.d_model <= 0 ||
        config.encoder_attention_heads <= 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid Audio Encoder configuration"
        );
    }
    if (!session.initialized() ||
        session.native_backend() == nullptr) {
        throw Error(
            ErrorCode::BackendUnavailable,
            "Audio Encoder requires an initialized ggml backend"
        );
    }
    std::vector<const GgufTensorDescriptor *> weight_descriptors;
    weight_descriptors.reserve(gguf_model.tensors.size());
    for (const GgufTensorDescriptor & descriptor : gguf_model.tensors) {
        if (descriptor.name.find(kAudioPrefix) == 0) {
            weight_descriptors.push_back(&descriptor);
        }
    }
    if (gguf_model.path.empty() || weight_descriptors.empty()) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Audio Encoder GGUF tensors are unavailable"
        );
    }
    const int expected_layers = config.encoder_layers;
    if (gguf_model.audio.encoder_layers != expected_layers) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Audio Encoder configuration does not match GGUF metadata"
        );
    }

    config_ = config;
    session_ = &session;
    scheduler_ = session.create_scheduler(session.graph_size());
    if (scheduler_ == nullptr) {
        throw Error(
            ErrorCode::BackendInitFailed,
            "failed to initialize Audio Encoder scheduler"
        );
    }
    const std::size_t context_size =
        ggml_tensor_overhead() *
        (weight_descriptors.size() + 2);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to create Audio Encoder weight context"
        );
    }

    positional_embedding_ = ggml_new_tensor_2d(
        weight_context_,
        GGML_TYPE_F32,
        config.d_model,
        config.max_source_positions
    );
    if (positional_embedding_ == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to create Audio Encoder positional embedding"
        );
    }
    ggml_set_name(
        positional_embedding_,
        "encoder.positional_embedding"
    );

    weights_.reserve(weight_descriptors.size());
    for (const GgufTensorDescriptor * descriptor_pointer :
         weight_descriptors) {
        const GgufTensorDescriptor & descriptor = *descriptor_pointer;
        validate_tensor_type(descriptor);
        ggml_tensor * tensor = ggml_new_tensor(
            weight_context_,
            static_cast<ggml_type>(descriptor.type),
            tensor_dimensions(descriptor),
            descriptor.ne.data()
        );
        if (tensor == nullptr) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "failed to create Audio Encoder tensor: " +
                    descriptor.name
            );
        }
        if (ggml_nbytes(tensor) != descriptor.byte_count) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "Audio Encoder tensor byte size mismatch: " +
                    descriptor.name
            );
        }
        ggml_set_name(tensor, descriptor.name.c_str());
        const auto inserted = weights_.emplace(
            descriptor.name,
            tensor
        );
        if (!inserted.second) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "duplicate Audio Encoder tensor: " +
                    descriptor.name
            );
        }
    }

    weight_buffer_ = ggml_backend_alloc_ctx_tensors(
        weight_context_,
        session.native_backend()
    );
    if (weight_buffer_ == nullptr) {
        throw Error(
            ErrorCode::BackendInitFailed,
            "failed to allocate Audio Encoder weight buffer"
        );
    }
    ggml_backend_buffer_set_usage(
        weight_buffer_,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS
    );
    const std::vector<float> positional_values =
        make_positional_embedding(
            config.max_source_positions,
            config.d_model
        );
    ggml_backend_tensor_set(
        positional_embedding_,
        positional_values.data(),
        0,
        positional_values.size() * sizeof(float)
    );

    std::ifstream input(gguf_model.path, std::ios::binary);
    if (!input) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to open Audio Encoder GGUF data"
        );
    }
    for (const GgufTensorDescriptor * descriptor_pointer :
         weight_descriptors) {
        const GgufTensorDescriptor & descriptor = *descriptor_pointer;
        std::vector<std::uint8_t> data(
            static_cast<std::size_t>(descriptor.byte_count)
        );
        input.seekg(
            static_cast<std::streamoff>(descriptor.file_offset),
            std::ios::beg
        );
        input.read(
            reinterpret_cast<char *>(data.data()),
            static_cast<std::streamsize>(data.size())
        );
        if (!input) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "failed to read Audio Encoder tensor: " +
                    descriptor.name
            );
        }
        ggml_backend_tensor_set(
            weights_.at(descriptor.name),
            data.data(),
            0,
            data.size()
        );
    }

    const auto bind = [this](const std::string & name) {
        return require_weight(name);
    };
    conv_weights_.conv2d1_weight = bind(
        "thinker.audio_tower.conv2d1.weight"
    );
    conv_weights_.conv2d1_bias = bind(
        "thinker.audio_tower.conv2d1.bias"
    );
    conv_weights_.conv2d2_weight = bind(
        "thinker.audio_tower.conv2d2.weight"
    );
    conv_weights_.conv2d2_bias = bind(
        "thinker.audio_tower.conv2d2.bias"
    );
    conv_weights_.conv2d3_weight = bind(
        "thinker.audio_tower.conv2d3.weight"
    );
    conv_weights_.conv2d3_bias = bind(
        "thinker.audio_tower.conv2d3.bias"
    );
    conv_weights_.conv_out_weight = bind(
        "thinker.audio_tower.conv_out.weight"
    );

    layer_weights_.resize(
        static_cast<std::size_t>(config.encoder_layers)
    );
    for (int layer = 0; layer < config.encoder_layers; ++layer) {
        const std::string prefix =
            "thinker.audio_tower.layers." + std::to_string(layer) + ".";
        AutEncoderLayerWeights & weights = layer_weights_[
            static_cast<std::size_t>(layer)
        ];
        weights.fc1_weight = bind(prefix + "fc1.weight");
        weights.fc1_bias = bind(prefix + "fc1.bias");
        weights.fc2_weight = bind(prefix + "fc2.weight");
        weights.fc2_bias = bind(prefix + "fc2.bias");
        weights.final_layer_norm_weight = bind(
            prefix + "final_layer_norm.weight"
        );
        weights.final_layer_norm_bias = bind(
            prefix + "final_layer_norm.bias"
        );
        weights.self_attn_layer_norm_weight = bind(
            prefix + "self_attn_layer_norm.weight"
        );
        weights.self_attn_layer_norm_bias = bind(
            prefix + "self_attn_layer_norm.bias"
        );
        weights.self_attn_q_proj_weight = bind(
            prefix + "self_attn.q_proj.weight"
        );
        weights.self_attn_q_proj_bias = bind(
            prefix + "self_attn.q_proj.bias"
        );
        weights.self_attn_k_proj_weight = bind(
            prefix + "self_attn.k_proj.weight"
        );
        weights.self_attn_k_proj_bias = bind(
            prefix + "self_attn.k_proj.bias"
        );
        weights.self_attn_v_proj_weight = bind(
            prefix + "self_attn.v_proj.weight"
        );
        weights.self_attn_v_proj_bias = bind(
            prefix + "self_attn.v_proj.bias"
        );
        weights.self_attn_out_proj_weight = bind(
            prefix + "self_attn.out_proj.weight"
        );
        weights.self_attn_out_proj_bias = bind(
            prefix + "self_attn.out_proj.bias"
        );
    }

    output_weights_.ln_post_weight = bind(
        "thinker.audio_tower.ln_post.weight"
    );
    output_weights_.ln_post_bias = bind(
        "thinker.audio_tower.ln_post.bias"
    );
    output_weights_.proj1_weight = bind(
        "thinker.audio_tower.proj1.weight"
    );
    output_weights_.proj1_bias = bind(
        "thinker.audio_tower.proj1.bias"
    );
    output_weights_.proj2_weight = bind(
        "thinker.audio_tower.proj2.weight"
    );
    output_weights_.proj2_bias = bind(
        "thinker.audio_tower.proj2.bias"
    );

    const std::size_t expected_weight_count =
        13 + 16 * static_cast<std::size_t>(config.encoder_layers);
    if (weights_.size() != expected_weight_count) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Audio Encoder weight count is invalid"
        );
    }
}

AutEncoderShape AutEncoder::output_shape(int64_t feature_length) const {
    if (feature_length <= 0 ||
        feature_length > std::numeric_limits<int>::max()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Audio Encoder feature length is invalid"
        );
    }
    if (config_.n_window <= 0 || config_.n_window_infer <= 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Audio Encoder window configuration is invalid"
        );
    }

    const int chunk_frames = config_.n_window * 2;
    const int feature_count = static_cast<int>(feature_length);
    const int full_chunks = feature_count / chunk_frames;
    const int tail_frames = feature_count % chunk_frames;
    const int chunks = full_chunks + (tail_frames != 0 ? 1 : 0);
    const int frames_per_chunk = convolution_output_length(chunk_frames);

    AutEncoderShape shape;
    shape.input_padded_frames = chunks * chunk_frames;
    shape.padded_frames = chunks * frames_per_chunk;
    shape.valid_frames =
        full_chunks * frames_per_chunk +
        (tail_frames != 0 ?
            convolution_output_length(tail_frames) : 0);
    shape.attention_window_size =
        frames_per_chunk * (config_.n_window_infer / chunk_frames);
    if (shape.attention_window_size <= 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Audio Encoder attention window is invalid"
        );
    }
    shape.attention_windows =
        (shape.valid_frames + shape.attention_window_size - 1) /
        shape.attention_window_size;
    return shape;
}

AutEncoderGraph AutEncoder::build_graph(
    ggml_context * context,
    int64_t feature_length
) const {
    if (!loaded() || context == nullptr) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Audio Encoder graph requires loaded weights and a context"
        );
    }
    if (config_.n_window <= 0 ||
        config_.encoder_attention_heads <= 0 ||
        config_.d_model % config_.encoder_attention_heads != 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Audio Encoder window or head configuration is invalid"
        );
    }
    if (config_.activation_function != "gelu") {
        throw Error(
            ErrorCode::InvalidArgument,
            "unsupported Audio Encoder activation function"
        );
    }

    const AutEncoderShape shape = output_shape(feature_length);
    const int chunk_frames = config_.n_window * 2;
    const int chunks = shape.input_padded_frames / chunk_frames;
    const int input_padded_frames = chunks * chunk_frames;
    const int frames_per_chunk = convolution_output_length(
        chunk_frames
    );
    const int padded_frames = shape.padded_frames;
    const int valid_frames = shape.valid_frames;

    AutEncoderGraph result;
    result.input_padded_frames = input_padded_frames;
    result.padded_frames = padded_frames;
    result.valid_frames = valid_frames;
    result.attention_window_size = shape.attention_window_size;
    result.attention_windows = shape.attention_windows;
    const int window_tokens =
        result.attention_window_size * result.attention_windows;

    result.input_features = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        input_padded_frames,
        config_.num_mel_bins
    );
    result.attention_mask = ggml_new_tensor_4d(
        context,
        GGML_TYPE_F16,
        result.attention_window_size,
        result.attention_window_size,
        1,
        result.attention_windows
    );
    result.valid_frame_indices = ggml_new_tensor_1d(
        context,
        GGML_TYPE_I32,
        valid_frames
    );
    result.window_frame_indices = ggml_new_tensor_1d(
        context,
        GGML_TYPE_I32,
        window_tokens
    );
    if (result.input_features == nullptr ||
        result.attention_mask == nullptr ||
        result.valid_frame_indices == nullptr ||
        result.window_frame_indices == nullptr) {
        throw Error(
            ErrorCode::InferenceFailed,
            "failed to create Audio Encoder inputs"
        );
    }
    ggml_set_name(result.input_features, "encoder.input_features");
    ggml_set_name(result.attention_mask, "encoder.attention_mask");
    ggml_set_name(
        result.valid_frame_indices,
        "encoder.valid_frame_indices"
    );
    ggml_set_name(
        result.window_frame_indices,
        "encoder.window_frame_indices"
    );
    ggml_set_input(result.input_features);
    ggml_set_input(result.attention_mask);
    ggml_set_input(result.valid_frame_indices);
    ggml_set_input(result.window_frame_indices);

    const auto cast_f32 = [context](ggml_tensor * tensor) {
        return tensor->type == GGML_TYPE_F32 ?
            tensor : ggml_cast(context, tensor, GGML_TYPE_F32);
    };
    const auto add_bias = [context, &cast_f32](
        ggml_tensor * value,
        ggml_tensor * bias
    ) {
        ggml_tensor * repeated = ggml_repeat(
            context,
            cast_f32(bias),
            value
        );
        return ggml_add(context, value, repeated);
    };
    const auto linear = [context, &add_bias](
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * bias
    ) {
        ggml_tensor * output = ggml_mul_mat(context, weight, input);
        return bias != nullptr ? add_bias(output, bias) : output;
    };
    const auto layer_norm = [
        context,
        &cast_f32,
        this
    ](
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * bias
    ) {
        ggml_tensor * normalized = ggml_norm(
            context,
            input,
            config_.layer_norm_eps
        );
        ggml_tensor * gamma = ggml_reshape_2d(
            context,
            cast_f32(weight),
            config_.d_model,
            1
        );
        ggml_tensor * beta = ggml_reshape_2d(
            context,
            cast_f32(bias),
            config_.d_model,
            1
        );
        normalized = ggml_mul(
            context,
            normalized,
            ggml_repeat(context, gamma, normalized)
        );
        return ggml_add(
            context,
            normalized,
            ggml_repeat(context, beta, normalized)
        );
    };

    const auto require_element_count = [](
        const ggml_tensor * tensor,
        int64_t expected,
        const char * label
    ) {
        if (ggml_nelements(tensor) != expected) {
            throw Error(
                ErrorCode::InferenceFailed,
                std::string("Audio Encoder reshape mismatch: ") +
                    label
            );
        }
    };
    require_element_count(
        result.input_features,
        static_cast<int64_t>(input_padded_frames) *
            config_.num_mel_bins,
        "padded log-mel"
    );
    ggml_tensor * hidden = ggml_reshape_4d(
        context,
        result.input_features,
        chunk_frames,
        config_.num_mel_bins,
        1,
        chunks
    );
    const auto conv_layer = [
        context,
        &cast_f32,
        &add_bias,
        &require_element_count
    ](
        ggml_tensor * input,
        ggml_tensor * weight,
        ggml_tensor * bias
    ) {
        ggml_tensor * output = ggml_conv_2d(
            context,
            cast_f32(weight),
            input,
            2,
            2,
            1,
            1,
            1,
            1
        );
        require_element_count(
            bias,
            bias->ne[0],
            "convolution bias"
        );
        output = add_bias(
            output,
            ggml_reshape_4d(
                context,
                bias,
                1,
                1,
                bias->ne[0],
                1
            )
        );
        return ggml_gelu_erf(context, output);
    };

    hidden = conv_layer(
        hidden,
        conv_weights_.conv2d1_weight,
        conv_weights_.conv2d1_bias
    );
    hidden = conv_layer(
        hidden,
        conv_weights_.conv2d2_weight,
        conv_weights_.conv2d2_bias
    );
    hidden = conv_layer(
        hidden,
        conv_weights_.conv2d3_weight,
        conv_weights_.conv2d3_bias
    );

    const int frequency_bins = convolution_output_length(
        config_.num_mel_bins
    );
    const int conv_hidden =
        config_.downsample_hidden_size * frequency_bins;
    // ggml_permute() maps source axis i to destination axis axis_i.
    // Convert [time, frequency, channel, chunk] to
    // [frequency, channel, time, chunk], matching PyTorch's conv_out flatten.
    ggml_tensor * conv_features = ggml_cont(
        context,
        ggml_permute(context, hidden, 2, 0, 1, 3)
    );
    conv_features = ggml_reshape_3d(
        context,
        conv_features,
        conv_hidden,
        frames_per_chunk,
        chunks
    );
    hidden = ggml_mul_mat(
        context,
        conv_weights_.conv_out_weight,
        conv_features
    );

    ggml_tensor * positions = ggml_view_2d(
        context,
        positional_embedding_,
        config_.d_model,
        frames_per_chunk,
        positional_embedding_->nb[1],
        0
    );
    positions = ggml_reshape_3d(
        context,
        positions,
        config_.d_model,
        frames_per_chunk,
        1
    );
    hidden = ggml_add(
        context,
        hidden,
        ggml_repeat(context, positions, hidden)
    );
    hidden = ggml_reshape_2d(
        context,
        hidden,
        config_.d_model,
        padded_frames
    );
    // The valid frames form a prefix after chunk-major flattening. Gather them
    // into a compact [hidden, sequence] tensor before Transformer attention.
    hidden = ggml_get_rows(
        context,
        hidden,
        result.valid_frame_indices
    );
    hidden = ggml_get_rows(
        context,
        hidden,
        result.window_frame_indices
    );
    const int window_size = result.attention_window_size;
    const int window_count = result.attention_windows;
    hidden = ggml_reshape_3d(
        context,
        hidden,
        config_.d_model,
        window_size,
        window_count
    );
    const int head_dim = config_.d_model /
        config_.encoder_attention_heads;
    const float attention_scale =
        1.0F / std::sqrt(static_cast<float>(head_dim));
    for (std::size_t layer_index = 0;
         layer_index < layer_weights_.size();
         ++layer_index) {
        const AutEncoderLayerWeights & layer =
            layer_weights_[layer_index];
        ggml_tensor * residual = hidden;
        ggml_tensor * normalized = layer_norm(
            hidden,
            layer.self_attn_layer_norm_weight,
            layer.self_attn_layer_norm_bias
        );

        ggml_tensor * query = linear(
            normalized,
            layer.self_attn_q_proj_weight,
            layer.self_attn_q_proj_bias
        );
        ggml_tensor * key = linear(
            normalized,
            layer.self_attn_k_proj_weight,
            layer.self_attn_k_proj_bias
        );
        ggml_tensor * value = linear(
            normalized,
            layer.self_attn_v_proj_weight,
            layer.self_attn_v_proj_bias
        );

        query = ggml_reshape_4d(
            context,
            query,
            head_dim,
            config_.encoder_attention_heads,
            window_size,
            window_count
        );
        key = ggml_reshape_4d(
            context,
            key,
            head_dim,
            config_.encoder_attention_heads,
            window_size,
            window_count
        );
        value = ggml_reshape_4d(
            context,
            value,
            head_dim,
            config_.encoder_attention_heads,
            window_size,
            window_count
        );
        query = ggml_permute(context, query, 0, 2, 1, 3);
        key = ggml_permute(context, key, 0, 2, 1, 3);
        value = ggml_permute(context, value, 0, 2, 1, 3);

        ggml_tensor * attended = ggml_cont_3d(
            context,
            ggml_flash_attn_ext(
                context,
                query,
                key,
                value,
                result.attention_mask,
                attention_scale,
                0.0F,
                0.0F
            ),
            config_.d_model,
            window_size,
            window_count
        );
        attended = linear(
            attended,
            layer.self_attn_out_proj_weight,
            layer.self_attn_out_proj_bias
        );
        hidden = ggml_add(context, residual, attended);

        residual = hidden;
        normalized = layer_norm(
            hidden,
            layer.final_layer_norm_weight,
            layer.final_layer_norm_bias
        );
        hidden = linear(
            normalized,
            layer.fc1_weight,
            layer.fc1_bias
        );
        hidden = ggml_gelu_erf(context, hidden);
        hidden = linear(
            hidden,
            layer.fc2_weight,
            layer.fc2_bias
        );
        hidden = ggml_add(context, residual, hidden);
    }

    hidden = ggml_reshape_2d(
        context,
        hidden,
        config_.d_model,
        window_tokens
    );
    hidden = ggml_get_rows(
        context,
        hidden,
        result.valid_frame_indices
    );
    hidden = layer_norm(
        hidden,
        output_weights_.ln_post_weight,
        output_weights_.ln_post_bias
    );
    hidden = linear(
        hidden,
        output_weights_.proj1_weight,
        output_weights_.proj1_bias
    );
    hidden = ggml_gelu_erf(context, hidden);
    hidden = linear(
        hidden,
        output_weights_.proj2_weight,
        output_weights_.proj2_bias
    );

    result.output = hidden;
    ggml_set_name(result.output, "encoder.audio_embeddings");
    result.graph = ggml_new_graph_custom(
        context,
        kMaxGraphNodes,
        false
    );
    if (result.graph == nullptr) {
        throw Error(
            ErrorCode::InferenceFailed,
            "failed to create Audio Encoder graph"
        );
    }
    ggml_build_forward_expand(result.graph, result.output);
    if (ggml_graph_n_nodes(result.graph) >= kMaxGraphNodes) {
        throw Error(
            ErrorCode::InferenceFailed,
            "Audio Encoder graph exceeds node capacity"
        );
    }
    return result;
}

void AutEncoder::release_compute_graph() noexcept {
    if (compute_context_ != nullptr) {
        ggml_free(compute_context_);
        compute_context_ = nullptr;
    }
}

ggml_tensor * AutEncoder::forward(
    const LogMelFeatures & features
) {
    if (!loaded() || session_ == nullptr) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Audio Encoder is not loaded"
        );
    }
    if (features.n_mel_bins != config_.num_mel_bins ||
        features.n_frames <= 0 ||
        features.data.size() !=
            static_cast<std::size_t>(features.n_mel_bins) *
            static_cast<std::size_t>(features.n_frames)) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid log-mel features for Audio Encoder"
        );
    }

    if (scheduler_ != nullptr) {
        ggml_backend_sched_reset(scheduler_);
    }
    release_compute_graph();

    const std::size_t context_size =
        ggml_tensor_overhead() * kMaxGraphNodes +
        ggml_graph_overhead_custom(kMaxGraphNodes, false);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        throw Error(
            ErrorCode::InferenceFailed,
            "failed to create Audio Encoder compute context"
        );
    }

    ggml_tensor * output = nullptr;
    try {
        const AutEncoderGraph built = build_graph(
            context,
            features.n_frames
        );
        if (!ggml_backend_sched_alloc_graph(
                scheduler_,
                built.graph)) {
            throw Error(
                ErrorCode::BackendComputeFailed,
                "failed to allocate Audio Encoder graph"
            );
        }

        std::vector<float> padded_features(
            static_cast<std::size_t>(built.input_padded_frames) *
            static_cast<std::size_t>(features.n_mel_bins),
            0.0F
        );
        // ggml_conv_2d consumes [time, mel, channel, chunk].
        const int chunk_frames = config_.n_window * 2;
        const int chunks = built.input_padded_frames / chunk_frames;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int source_offset = chunk * chunk_frames;
            const int frame_count = std::min(
                chunk_frames,
                features.n_frames - source_offset
            );
            for (int mel = 0; mel < features.n_mel_bins; ++mel) {
                for (int frame = 0; frame < frame_count; ++frame) {
                    padded_features[
                        static_cast<std::size_t>(
                            (chunk * features.n_mel_bins + mel) *
                                chunk_frames +
                            frame
                        )
                    ] = features.data[
                        static_cast<std::size_t>(
                            mel * features.n_frames +
                            source_offset +
                            frame
                        )
                    ];
                }
            }
        }
        ggml_backend_tensor_set(
            built.input_features,
            padded_features.data(),
            0,
            padded_features.size() * sizeof(float)
        );
        std::vector<std::int32_t> valid_frame_indices(
            built.valid_frames
        );
        for (int index = 0; index < built.valid_frames; ++index) {
            valid_frame_indices[static_cast<std::size_t>(index)] = index;
        }
        ggml_backend_tensor_set(
            built.valid_frame_indices,
            valid_frame_indices.data(),
            0,
            valid_frame_indices.size() * sizeof(std::int32_t)
        );

        const int valid = built.valid_frames;
        const int window_size = built.attention_window_size;
        const int window_count = built.attention_windows;
        const int window_tokens = window_size * window_count;
        std::vector<std::int32_t> window_frame_indices(
            static_cast<std::size_t>(window_tokens)
        );
        for (int index = 0; index < window_tokens; ++index) {
            window_frame_indices[static_cast<std::size_t>(index)] =
                index < valid ? index : 0;
        }
        ggml_backend_tensor_set(
            built.window_frame_indices,
            window_frame_indices.data(),
            0,
            window_frame_indices.size() * sizeof(std::int32_t)
        );

        std::vector<float> mask(
            static_cast<std::size_t>(window_size) *
            static_cast<std::size_t>(window_size) *
            static_cast<std::size_t>(window_count),
            -std::numeric_limits<float>::infinity()
        );
        for (int window = 0; window < window_count; ++window) {
            const int window_start = window * window_size;
            const int valid_keys = std::min(
                window_size,
                valid - window_start
            );
            for (int query = 0; query < window_size; ++query) {
                for (int key = 0; key < valid_keys; ++key) {
                    mask[
                        static_cast<std::size_t>(
                            (window * window_size + query) *
                                window_size +
                            key
                        )
                    ] = 0.0F;
                }
            }
        }
        std::vector<ggml_fp16_t> mask_half(mask.size());
        ggml_fp32_to_fp16_row(
            mask.data(),
            mask_half.data(),
            static_cast<int64_t>(mask.size())
        );
        ggml_backend_tensor_set(
            built.attention_mask,
            mask_half.data(),
            0,
            mask_half.size() * sizeof(ggml_fp16_t)
        );
        const ggml_status status = ggml_backend_sched_graph_compute(
            scheduler_,
            built.graph
        );
        if (status != GGML_STATUS_SUCCESS) {
            throw Error(
                ErrorCode::BackendComputeFailed,
                "Audio Encoder graph computation failed"
            );
        }
        output = built.output;
    } catch (...) {
        ggml_free(context);
        throw;
    }

    compute_context_ = context;
    return output;
}

void AutEncoder::unload() noexcept {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_free(scheduler_);
        scheduler_ = nullptr;
    }
    release_compute_graph();
    if (weight_buffer_ != nullptr) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_context_ != nullptr) {
        ggml_free(weight_context_);
        weight_context_ = nullptr;
    }
    weights_.clear();
    positional_embedding_ = nullptr;
    conv_weights_ = {};
    layer_weights_.clear();
    output_weights_ = {};
    config_ = AutEncoderConfig{};
    session_ = nullptr;
}

ggml_tensor * AutEncoder::weight(std::string_view name) const {
    const auto found = weights_.find(std::string(name));
    return found != weights_.end() ? found->second : nullptr;
}

ggml_tensor * AutEncoder::require_weight(std::string_view name) const {
    ggml_tensor * tensor = weight(name);
    if (tensor == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Audio Encoder weight is missing: " +
                std::string(name)
        );
    }
    return tensor;
}

}  // namespace qwen3_asr
