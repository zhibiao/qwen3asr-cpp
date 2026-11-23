#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "audio_frontend.h"
#include "configs.h"
#include "ggml_session.h"
#include "gguf_model.h"

namespace qwen3_asr {

struct AutEncoderGraph {
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input_features = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * valid_frame_indices = nullptr;
    ggml_tensor * window_frame_indices = nullptr;
    ggml_tensor * output = nullptr;
    int input_padded_frames = 0;
    int padded_frames = 0;
    int valid_frames = 0;
    int attention_window_size = 0;
    int attention_windows = 0;
};

struct AutEncoderShape {
    int input_padded_frames = 0;
    int padded_frames = 0;
    int valid_frames = 0;
    int attention_window_size = 0;
    int attention_windows = 0;
};

struct AutEncoderConvWeights {
    ggml_tensor * conv2d1_weight = nullptr;
    ggml_tensor * conv2d1_bias = nullptr;
    ggml_tensor * conv2d2_weight = nullptr;
    ggml_tensor * conv2d2_bias = nullptr;
    ggml_tensor * conv2d3_weight = nullptr;
    ggml_tensor * conv2d3_bias = nullptr;
    ggml_tensor * conv_out_weight = nullptr;
};

struct AutEncoderLayerWeights {
    ggml_tensor * fc1_weight = nullptr;
    ggml_tensor * fc1_bias = nullptr;
    ggml_tensor * fc2_weight = nullptr;
    ggml_tensor * fc2_bias = nullptr;
    ggml_tensor * final_layer_norm_weight = nullptr;
    ggml_tensor * final_layer_norm_bias = nullptr;
    ggml_tensor * self_attn_layer_norm_weight = nullptr;
    ggml_tensor * self_attn_layer_norm_bias = nullptr;
    ggml_tensor * self_attn_q_proj_weight = nullptr;
    ggml_tensor * self_attn_q_proj_bias = nullptr;
    ggml_tensor * self_attn_k_proj_weight = nullptr;
    ggml_tensor * self_attn_k_proj_bias = nullptr;
    ggml_tensor * self_attn_v_proj_weight = nullptr;
    ggml_tensor * self_attn_v_proj_bias = nullptr;
    ggml_tensor * self_attn_out_proj_weight = nullptr;
    ggml_tensor * self_attn_out_proj_bias = nullptr;
};

struct AutEncoderOutputWeights {
    ggml_tensor * ln_post_weight = nullptr;
    ggml_tensor * ln_post_bias = nullptr;
    ggml_tensor * proj1_weight = nullptr;
    ggml_tensor * proj1_bias = nullptr;
    ggml_tensor * proj2_weight = nullptr;
    ggml_tensor * proj2_bias = nullptr;
};

class AutEncoder {
public:
    AutEncoder();
    ~AutEncoder();

    void load(
        const AutEncoderConfig & config,
        const GgufModel & gguf_model,
        GgmlSession & session
    );
    void unload() noexcept;

    AutEncoderGraph build_graph(
        ggml_context * context,
        int64_t feature_length
    ) const;
    AutEncoderShape output_shape(int64_t feature_length) const;
    // The returned graph tensor is [hidden_size, valid_frames] and remains
    // valid until the next forward(), unload(), or destruction.
    ggml_tensor * forward(
        const LogMelFeatures & features
    );

    bool loaded() const noexcept {
        return weight_context_ != nullptr;
    }

    std::size_t weight_count() const noexcept {
        return weights_.size();
    }

    ggml_backend_sched_t scheduler() const noexcept {
        return scheduler_;
    }

    ggml_tensor * weight(std::string_view name) const;
    ggml_tensor * require_weight(std::string_view name) const;

    const AutEncoderConvWeights & conv_weights() const noexcept {
        return conv_weights_;
    }

    const std::vector<AutEncoderLayerWeights> & layer_weights() const noexcept {
        return layer_weights_;
    }

    const AutEncoderOutputWeights & output_weights() const noexcept {
        return output_weights_;
    }

    ggml_context * native_context() const noexcept {
        return weight_context_;
    }

    ggml_backend_buffer_t native_weight_buffer() const noexcept {
        return weight_buffer_;
    }

    ggml_tensor * positional_embedding() const noexcept {
        return positional_embedding_;
    }

private:
    void load_weights(
        const AutEncoderConfig & config,
        const GgufModel & gguf_model,
        GgmlSession & session
    );
    void release_compute_graph() noexcept;

    AutEncoderConfig config_;
    GgmlSession * session_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    ggml_context * compute_context_ = nullptr;
    ggml_context * weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ggml_tensor * positional_embedding_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> weights_;
    AutEncoderConvWeights conv_weights_;
    std::vector<AutEncoderLayerWeights> layer_weights_;
    AutEncoderOutputWeights output_weights_;
};

}  // namespace qwen3_asr
