#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

#include "configs.h"
#include "ggml_session.h"
#include "gguf_model.h"
#include "sampler.h"

namespace qwen3_asr {

struct Qwen3LmLayerWeights {
    ggml_tensor * input_layernorm_weight = nullptr;
    ggml_tensor * post_attention_layernorm_weight = nullptr;
    ggml_tensor * self_attn_q_norm_weight = nullptr;
    ggml_tensor * self_attn_k_norm_weight = nullptr;
    ggml_tensor * self_attn_q_proj_weight = nullptr;
    ggml_tensor * self_attn_k_proj_weight = nullptr;
    ggml_tensor * self_attn_v_proj_weight = nullptr;
    ggml_tensor * self_attn_o_proj_weight = nullptr;
    ggml_tensor * mlp_gate_proj_weight = nullptr;
    ggml_tensor * mlp_up_proj_weight = nullptr;
    ggml_tensor * mlp_down_proj_weight = nullptr;
};

struct Qwen3LmOutputWeights {
    ggml_tensor * embed_tokens_weight = nullptr;
    ggml_tensor * final_norm_weight = nullptr;
    ggml_tensor * lm_head_weight = nullptr;
};

struct Qwen3LmGraph {
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input_ids = nullptr;
    ggml_tensor * position_ids = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * output = nullptr;
    int sequence_length = 0;
    int64_t past_length = 0;
};

struct Qwen3LmGenerationConfig {
    int max_new_tokens = 1024;
    // Empty uses the EOS IDs embedded in the LM GGUF.
    std::vector<std::int32_t> eos_token_ids;
    float temperature = 0.0F;
    int top_k = 0;
    float top_p = 1.0F;
    std::uint64_t seed = 0;
};

struct Qwen3LmGenerationStats {
    std::int64_t prompt_tokens = 0;
    std::int64_t decode_calls = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

class Qwen3Lm {
public:
    Qwen3Lm();
    ~Qwen3Lm();

    void load(
        const Qwen3LmConfig & config,
        const GgufModel & gguf_model,
        GgmlSession & session
    );
    void unload() noexcept;

    // Starts a new sequence using the initialized KV cache. Audio Encoder
    // output is a borrowed backend tensor [hidden_size, audio_token_count].
    // Returns CPU logits for the last prompt token.
    std::vector<float> prefill(
        const std::vector<std::int32_t> & input_ids,
        const ggml_tensor * audio_embeddings,
        std::size_t audio_start_index,
        std::size_t audio_token_count
    );

    // Appends one token to the current KV cache.
    // Returns CPU logits for the next token.
    std::vector<float> decode(
        std::int32_t input_id,
        std::int32_t position_id
    );

    // Prefills the prompt, samples the next token, then repeatedly appends
    // that token and decodes until EOS or max_new_tokens is reached.
    // EOS and the prompt are not included in the returned token IDs.
    std::vector<std::int32_t> generate(
        const std::vector<std::int32_t> & input_ids,
        const ggml_tensor * audio_embeddings = nullptr,
        std::size_t audio_start_index = 0,
        std::size_t audio_token_count = 0,
        const Qwen3LmGenerationConfig & generation = {},
        Qwen3LmGenerationStats * stats = nullptr
    );

    void reset_cache();
    int64_t cache_length() const noexcept;

    bool loaded() const noexcept {
        return weight_context_ != nullptr;
    }

    std::size_t weight_count() const noexcept {
        return weights_.size();
    }

    const Qwen3LmConfig & config() const noexcept {
        return config_;
    }

    ggml_backend_sched_t scheduler() const noexcept {
        return scheduler_;
    }

    ggml_tensor * weight(std::string_view name) const;
    ggml_tensor * require_weight(std::string_view name) const;

    const std::vector<Qwen3LmLayerWeights> & layer_weights() const noexcept {
        return layer_weights_;
    }

    const Qwen3LmOutputWeights & output_weights() const noexcept {
        return output_weights_;
    }

    ggml_context * native_context() const noexcept {
        return weight_context_;
    }

    ggml_backend_buffer_t native_weight_buffer() const noexcept {
        return weight_buffer_;
    }

private:
    void load_weights(
        const Qwen3LmConfig & config,
        const GgufModel & gguf_model,
        GgmlSession & session
    );
    Qwen3LmGraph build_graph(
        ggml_context * context,
        const std::vector<std::int32_t> & input_ids,
        const ggml_tensor * audio_embeddings,
        std::size_t audio_start_index,
        std::size_t audio_token_count,
        int64_t past_length
    ) const;
    std::vector<float> run_graph(
        const std::vector<std::int32_t> & input_ids,
        const ggml_tensor * audio_embeddings,
        std::size_t audio_start_index,
        std::size_t audio_token_count,
        int64_t past_length
    );
    void allocate_kv_cache();
    void release_kv_cache() noexcept;
    bool is_generation_eos(
        std::int32_t token_id,
        const std::vector<std::int32_t> & eos_token_ids
    ) const;

    Qwen3LmConfig config_;
    GgmlSession * session_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
    ggml_context * weight_context_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ggml_context * cache_context_ = nullptr;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::vector<ggml_tensor *> cache_keys_;
    std::vector<ggml_tensor *> cache_values_;
    int64_t cache_length_ = 0;
    std::vector<std::int32_t> generation_eos_token_ids_;
    std::unordered_map<std::string, ggml_tensor *> weights_;
    std::vector<Qwen3LmLayerWeights> layer_weights_;
    Qwen3LmOutputWeights output_weights_;
};

}  // namespace qwen3_asr
