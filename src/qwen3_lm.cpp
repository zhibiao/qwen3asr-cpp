#include "qwen3_lm.h"

#include <algorithm>
#include <chrono>
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

constexpr int kMaxGraphNodes = 4096;

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
                "unsupported Qwen3 LM tensor dtype: " +
                    descriptor.name
            );
    }
}

}  // namespace

Qwen3Lm::Qwen3Lm() = default;

Qwen3Lm::~Qwen3Lm() {
    unload();
}

void Qwen3Lm::load(
    const Qwen3LmConfig & config,
    const GgufModel & gguf_model,
    GgmlSession & session
) {
    unload();
    load_weights(config, gguf_model, session);
    allocate_kv_cache();
}

void Qwen3Lm::load_weights(
    const Qwen3LmConfig & config,
    const GgufModel & gguf_model,
    GgmlSession & session
) {
    if (config.layers <= 0 ||
        config.hidden_size <= 0 ||
        config.attention_heads <= 0 ||
        config.kv_heads <= 0 ||
        config.head_dim <= 0 ||
        config.vocab_size <= 0 ||
        config.max_position_embeddings <= 0 ||
        config.num_ctx <= 0 ||
        config.num_ctx > config.max_position_embeddings) {
        throw Error(
            ErrorCode::InvalidArgument,
            "invalid Qwen3 LM configuration"
        );
    }
    if (!session.initialized() ||
        session.native_backend() == nullptr) {
        throw Error(
            ErrorCode::BackendUnavailable,
            "Qwen3 LM requires an initialized ggml backend"
        );
    }
    if (gguf_model.path.empty() ||
        gguf_model.tensors.empty()) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Qwen3 LM GGUF tensors are unavailable"
        );
    }
    if (gguf_model.text.layers != config.layers) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Qwen3 LM configuration does not match GGUF metadata"
        );
    }

    config_ = config;
    session_ = &session;
    generation_eos_token_ids_ = gguf_model.generation_eos_token_ids;
    if (generation_eos_token_ids_.empty() &&
        gguf_model.tokenizer_data.eos_token_id >= 0) {
        generation_eos_token_ids_.push_back(
            gguf_model.tokenizer_data.eos_token_id
        );
    }
    scheduler_ = session.create_scheduler(session.graph_size());
    if (scheduler_ == nullptr) {
        throw Error(
            ErrorCode::BackendInitFailed,
            "failed to initialize Qwen3 LM scheduler"
        );
    }
    const std::size_t context_size =
        ggml_tensor_overhead() *
        (gguf_model.tensors.size() + 1);
    ggml_init_params params{};
    params.mem_size = context_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    weight_context_ = ggml_init(params);
    if (weight_context_ == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to create Qwen3 LM weight context"
        );
    }

    weights_.reserve(gguf_model.tensors.size());
    for (const GgufTensorDescriptor & descriptor :
         gguf_model.tensors) {
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
                "failed to create Qwen3 LM tensor: " +
                    descriptor.name
            );
        }
        if (ggml_nbytes(tensor) != descriptor.byte_count) {
            throw Error(
                ErrorCode::ModelLoadFailed,
                "Qwen3 LM tensor byte size mismatch: " +
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
                "duplicate Qwen3 LM tensor: " +
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
            "failed to allocate Qwen3 LM weight buffer"
        );
    }
    ggml_backend_buffer_set_usage(
        weight_buffer_,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS
    );

    std::ifstream input(gguf_model.path, std::ios::binary);
    if (!input) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "failed to open Qwen3 LM GGUF data"
        );
    }
    for (const GgufTensorDescriptor & descriptor :
         gguf_model.tensors) {
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
                "failed to read Qwen3 LM tensor: " +
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
    output_weights_.embed_tokens_weight = bind(
        "thinker.model.embed_tokens.weight"
    );

    layer_weights_.resize(
        static_cast<std::size_t>(config.layers)
    );
    for (int layer = 0; layer < config.layers; ++layer) {
        const std::string prefix =
            "thinker.model.layers." + std::to_string(layer) + ".";
        Qwen3LmLayerWeights & weights = layer_weights_[
            static_cast<std::size_t>(layer)
        ];
        weights.input_layernorm_weight = bind(
            prefix + "input_layernorm.weight"
        );
        weights.post_attention_layernorm_weight = bind(
            prefix + "post_attention_layernorm.weight"
        );
        weights.self_attn_q_norm_weight = bind(
            prefix + "self_attn.q_norm.weight"
        );
        weights.self_attn_k_norm_weight = bind(
            prefix + "self_attn.k_norm.weight"
        );
        weights.self_attn_q_proj_weight = bind(
            prefix + "self_attn.q_proj.weight"
        );
        weights.self_attn_k_proj_weight = bind(
            prefix + "self_attn.k_proj.weight"
        );
        weights.self_attn_v_proj_weight = bind(
            prefix + "self_attn.v_proj.weight"
        );
        weights.self_attn_o_proj_weight = bind(
            prefix + "self_attn.o_proj.weight"
        );
        weights.mlp_gate_proj_weight = bind(
            prefix + "mlp.gate_proj.weight"
        );
        weights.mlp_up_proj_weight = bind(
            prefix + "mlp.up_proj.weight"
        );
        weights.mlp_down_proj_weight = bind(
            prefix + "mlp.down_proj.weight"
        );
    }

    output_weights_.final_norm_weight = bind(
        "thinker.model.norm.weight"
    );
    output_weights_.lm_head_weight = bind(
        "thinker.lm_head.weight"
    );

    const std::size_t expected_weight_count =
        3 + 11 * static_cast<std::size_t>(config.layers);
    if (weights_.size() != expected_weight_count) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Qwen3 LM weight count is invalid"
        );
    }
}

void Qwen3Lm::allocate_kv_cache() {
    const std::size_t tensor_count =
        static_cast<std::size_t>(config_.layers) * 2;
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (tensor_count + 2);
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    cache_context_ = ggml_init(params);
    if (cache_context_ == nullptr) {
        throw Error(
            ErrorCode::InferenceFailed,
            "failed to create Qwen3 LM cache context"
        );
    }

    cache_keys_.resize(static_cast<std::size_t>(config_.layers));
    cache_values_.resize(static_cast<std::size_t>(config_.layers));
    for (int layer = 0; layer < config_.layers; ++layer) {
        const std::string prefix =
            "thinker.model.layers." + std::to_string(layer) +
            ".self_attn.cache.";
        cache_keys_[static_cast<std::size_t>(layer)] =
            ggml_new_tensor_3d(
                cache_context_,
                GGML_TYPE_F16,
                config_.head_dim,
                config_.kv_heads,
                config_.num_ctx
            );
        cache_values_[static_cast<std::size_t>(layer)] =
            ggml_new_tensor_3d(
                cache_context_,
                GGML_TYPE_F16,
                config_.head_dim,
                config_.kv_heads,
                config_.num_ctx
            );
        if (cache_keys_[static_cast<std::size_t>(layer)] == nullptr ||
            cache_values_[static_cast<std::size_t>(layer)] == nullptr) {
            release_kv_cache();
            throw Error(
                ErrorCode::InferenceFailed,
                "failed to create Qwen3 LM KV cache tensors"
            );
        }
        ggml_set_name(
            cache_keys_[static_cast<std::size_t>(layer)],
            (prefix + "k").c_str()
        );
        ggml_set_name(
            cache_values_[static_cast<std::size_t>(layer)],
            (prefix + "v").c_str()
        );
    }

    cache_buffer_ = ggml_backend_alloc_ctx_tensors(
        cache_context_,
        session_->native_backend()
    );
    if (cache_buffer_ == nullptr) {
        release_kv_cache();
        throw Error(
            ErrorCode::BackendInitFailed,
            "failed to allocate Qwen3 LM KV cache"
        );
    }
    cache_length_ = 0;
}

Qwen3LmGraph Qwen3Lm::build_graph(
    ggml_context * context,
    const std::vector<std::int32_t> & input_ids,
    const ggml_tensor * audio_embeddings,
    std::size_t audio_start_index,
    std::size_t audio_token_count,
    int64_t past_length
) const {
    if (!loaded() || session_ == nullptr || context == nullptr) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3 LM graph requires loaded weights and a session"
        );
    }
    if (input_ids.empty()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3 LM input token IDs are empty"
        );
    }
    const int sequence_length = static_cast<int>(input_ids.size());
    const int64_t total_length = past_length + sequence_length;
    if (past_length < 0 || total_length > config_.num_ctx) {
        throw Error(
            ErrorCode::ContextLengthExceeded,
            "Qwen3 LM context length exceeds num_ctx"
        );
    }
    if (total_length > config_.max_position_embeddings) {
        throw Error(
            ErrorCode::ContextLengthExceeded,
            "Qwen3 LM context length exceeds model positions"
        );
    }

    Qwen3LmGraph result;
    result.sequence_length = sequence_length;
    result.past_length = past_length;
    std::size_t suffix_start = 0;
    if (audio_embeddings != nullptr) {
        suffix_start = audio_start_index + audio_token_count;
        if (audio_start_index == 0 ||
            audio_token_count == 0 ||
            suffix_start >= input_ids.size() ||
            audio_embeddings->ne[0] != config_.hidden_size ||
            audio_embeddings->ne[1] !=
                static_cast<int64_t>(audio_token_count)) {
            throw Error(
                ErrorCode::InvalidArgument,
                "Qwen3 LM audio embedding span is invalid"
            );
        }
    }

    result.input_ids = ggml_new_tensor_1d(
        context,
        GGML_TYPE_I32,
        sequence_length
    );
    result.position_ids = ggml_new_tensor_1d(
        context,
        GGML_TYPE_I32,
        sequence_length
    );
    result.attention_mask = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F16,
        total_length,
        sequence_length
    );
    ggml_set_name(result.input_ids, "lm.input_ids");
    ggml_set_name(result.position_ids, "lm.position_ids");
    ggml_set_name(result.attention_mask, "lm.attention_mask");
    ggml_set_input(result.input_ids);
    ggml_set_input(result.position_ids);
    ggml_set_input(result.attention_mask);

    const auto cast_f32 = [context](ggml_tensor * tensor) {
        return tensor->type == GGML_TYPE_F32 ?
            tensor : ggml_cast(context, tensor, GGML_TYPE_F32);
    };
    const auto linear = [context, &cast_f32](
        ggml_tensor * input,
        ggml_tensor * weight
    ) {
        return ggml_mul_mat(
            context,
            weight->type == GGML_TYPE_Q8_0 ?
                weight : cast_f32(weight),
            input
        );
    };
    const auto rms_norm = [context, &cast_f32, this](
        ggml_tensor * input,
        ggml_tensor * weight
    ) {
        ggml_tensor * normalized = ggml_rms_norm(
            context,
            input,
            config_.rms_norm_eps
        );
        ggml_tensor * scale = ggml_reshape_3d(
            context,
            cast_f32(weight),
            weight->ne[0],
            1,
            1
        );
        return ggml_mul(
            context,
            normalized,
            ggml_repeat(context, scale, normalized)
        );
    };
    const auto token_embeddings = [context, &cast_f32, this](
        ggml_tensor * ids
    ) {
        return cast_f32(ggml_get_rows(
            context,
            output_weights_.embed_tokens_weight,
            ids
        ));
    };

    ggml_tensor * hidden = token_embeddings(result.input_ids);
    if (audio_embeddings != nullptr) {
        if (audio_embeddings->buffer == nullptr) {
            throw Error(
                ErrorCode::InvalidArgument,
                "Qwen3 LM audio embeddings are not allocated"
            );
        }
        ggml_tensor * audio_input = ggml_new_tensor_2d(
            context,
            GGML_TYPE_F32,
            config_.hidden_size,
            audio_token_count
        );
        const ggml_status alloc_status = ggml_backend_tensor_alloc(
            audio_embeddings->buffer,
            audio_input,
            audio_embeddings->data
        );
        if (alloc_status != GGML_STATUS_SUCCESS) {
            throw Error(
                ErrorCode::BackendComputeFailed,
                "failed to bind Qwen3 LM audio embeddings"
            );
        }
        ggml_set_name(audio_input, "lm.audio_embeddings");
        ggml_tensor * prefix = ggml_view_2d(
            context,
            hidden,
            config_.hidden_size,
            audio_start_index,
            hidden->nb[1],
            0
        );
        ggml_tensor * suffix = ggml_view_2d(
            context,
            hidden,
            config_.hidden_size,
            input_ids.size() - suffix_start,
            hidden->nb[1],
            suffix_start * hidden->nb[1]
        );
        prefix = ggml_cont_2d(
            context,
            prefix,
            config_.hidden_size,
            audio_start_index
        );
        suffix = ggml_cont_2d(
            context,
            suffix,
            config_.hidden_size,
            input_ids.size() - suffix_start
        );
        hidden = ggml_concat(
            context,
            ggml_concat(
                context,
                prefix,
                cast_f32(audio_input),
                1
            ),
            suffix,
            1
        );
    }

    result.graph = ggml_new_graph_custom(
        context,
        kMaxGraphNodes,
        false
    );
    if (result.graph == nullptr) {
        throw Error(
            ErrorCode::InferenceFailed,
            "failed to create Qwen3 LM graph"
        );
    }

    const int q_heads = config_.attention_heads;
    const int kv_heads = config_.kv_heads;
    const int head_dim = config_.head_dim;
    const int q_dim = q_heads * head_dim;
    const float attention_scale =
        1.0F / std::sqrt(static_cast<float>(head_dim));

    for (int layer_index = 0;
         layer_index < config_.layers;
         ++layer_index) {
        const Qwen3LmLayerWeights & layer =
            layer_weights_[static_cast<std::size_t>(layer_index)];
        ggml_tensor * residual = hidden;
        ggml_tensor * normalized = rms_norm(
            hidden,
            layer.input_layernorm_weight
        );
        ggml_tensor * query = linear(
            normalized,
            layer.self_attn_q_proj_weight
        );
        ggml_tensor * key = linear(
            normalized,
            layer.self_attn_k_proj_weight
        );
        ggml_tensor * value = linear(
            normalized,
            layer.self_attn_v_proj_weight
        );
        query = ggml_reshape_3d(
            context,
            query,
            head_dim,
            q_heads,
            sequence_length
        );
        key = ggml_reshape_3d(
            context,
            key,
            head_dim,
            kv_heads,
            sequence_length
        );
        value = ggml_reshape_3d(
            context,
            value,
            head_dim,
            kv_heads,
            sequence_length
        );
        query = rms_norm(query, layer.self_attn_q_norm_weight);
        key = rms_norm(key, layer.self_attn_k_norm_weight);
        query = ggml_rope_ext(
            context,
            query,
            result.position_ids,
            nullptr,
            head_dim,
            GGML_ROPE_TYPE_NEOX,
            config_.max_position_embeddings,
            config_.rope_theta,
            1.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F
        );
        key = ggml_rope_ext(
            context,
            key,
            result.position_ids,
            nullptr,
            head_dim,
            GGML_ROPE_TYPE_NEOX,
            config_.max_position_embeddings,
            config_.rope_theta,
            1.0F,
            0.0F,
            1.0F,
            0.0F,
            0.0F
        );

        ggml_tensor * key_cache = cache_keys_[
            static_cast<std::size_t>(layer_index)
        ];
        ggml_tensor * value_cache = cache_values_[
            static_cast<std::size_t>(layer_index)
        ];
        ggml_tensor * key_write = ggml_view_3d(
            context,
            key_cache,
            head_dim,
            kv_heads,
            sequence_length,
            key_cache->nb[1],
            key_cache->nb[2],
            past_length * key_cache->nb[2]
        );
        ggml_tensor * value_write = ggml_view_3d(
            context,
            value_cache,
            head_dim,
            kv_heads,
            sequence_length,
            value_cache->nb[1],
            value_cache->nb[2],
            past_length * value_cache->nb[2]
        );
        ggml_build_forward_expand(
            result.graph,
            ggml_cpy(context, key, key_write)
        );
        ggml_build_forward_expand(
            result.graph,
            ggml_cpy(context, value, value_write)
        );

        ggml_tensor * key_view = ggml_view_3d(
            context,
            key_cache,
            head_dim,
            kv_heads,
            total_length,
            key_cache->nb[1],
            key_cache->nb[2],
            0
        );
        ggml_tensor * value_view = ggml_view_3d(
            context,
            value_cache,
            head_dim,
            kv_heads,
            total_length,
            value_cache->nb[1],
            value_cache->nb[2],
            0
        );
        key_view = ggml_permute(context, key_view, 0, 2, 1, 3);
        value_view = ggml_permute(context, value_view, 0, 2, 1, 3);
        query = ggml_permute(context, query, 0, 2, 1, 3);
        ggml_tensor * attended = ggml_cont_2d(
            context,
            ggml_flash_attn_ext(
                context,
                query,
                key_view,
                value_view,
                result.attention_mask,
                attention_scale,
                0.0F,
                0.0F
            ),
            q_dim,
            sequence_length
        );
        attended = linear(
            attended,
            layer.self_attn_o_proj_weight
        );
        hidden = ggml_add(context, residual, attended);

        residual = hidden;
        normalized = rms_norm(
            hidden,
            layer.post_attention_layernorm_weight
        );
        ggml_tensor * gate = linear(
            normalized,
            layer.mlp_gate_proj_weight
        );
        ggml_tensor * up = linear(
            normalized,
            layer.mlp_up_proj_weight
        );
        hidden = ggml_mul(
            context,
            ggml_silu(context, gate),
            up
        );
        hidden = linear(hidden, layer.mlp_down_proj_weight);
        hidden = ggml_add(context, residual, hidden);
    }

    hidden = rms_norm(hidden, output_weights_.final_norm_weight);
    ggml_tensor * last = ggml_view_2d(
        context,
        hidden,
        config_.hidden_size,
        1,
        hidden->nb[1],
        static_cast<std::size_t>(sequence_length - 1) * hidden->nb[1]
    );
    result.output = linear(last, output_weights_.lm_head_weight);
    ggml_set_name(result.output, "lm.logits");
    ggml_build_forward_expand(result.graph, result.output);
    if (ggml_graph_n_nodes(result.graph) >= kMaxGraphNodes) {
        throw Error(
            ErrorCode::InferenceFailed,
            "Qwen3 LM graph exceeds node capacity"
        );
    }
    return result;
}

std::vector<float> Qwen3Lm::run_graph(
    const std::vector<std::int32_t> & input_ids,
    const ggml_tensor * audio_embeddings,
    std::size_t audio_start_index,
    std::size_t audio_token_count,
    int64_t past_length
) {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_reset(scheduler_);
    }

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
            "failed to create Qwen3 LM compute context"
        );
    }
    std::vector<float> logits(
        static_cast<std::size_t>(config_.vocab_size)
    );
    try {
        const Qwen3LmGraph built = build_graph(
            context,
            input_ids,
            audio_embeddings,
            audio_start_index,
            audio_token_count,
            past_length
        );
        if (!ggml_backend_sched_alloc_graph(scheduler_, built.graph)) {
            throw Error(
                ErrorCode::BackendComputeFailed,
                "failed to allocate Qwen3 LM graph"
            );
        }

        ggml_backend_tensor_set(
            built.input_ids,
            input_ids.data(),
            0,
            input_ids.size() * sizeof(std::int32_t)
        );
        std::vector<std::int32_t> positions(
            static_cast<std::size_t>(built.sequence_length)
        );
        for (int index = 0; index < built.sequence_length; ++index) {
            positions[static_cast<std::size_t>(index)] =
                static_cast<std::int32_t>(past_length + index);
        }
        ggml_backend_tensor_set(
            built.position_ids,
            positions.data(),
            0,
            positions.size() * sizeof(std::int32_t)
        );

        const int total_length =
            static_cast<int>(past_length) + built.sequence_length;
        std::vector<float> mask(
            static_cast<std::size_t>(total_length) *
            static_cast<std::size_t>(built.sequence_length),
            -std::numeric_limits<float>::infinity()
        );
        for (int query = 0; query < built.sequence_length; ++query) {
            const int last_key = static_cast<int>(past_length) + query;
            for (int key = 0; key <= last_key; ++key) {
                mask[
                    static_cast<std::size_t>(query) * total_length +
                    key
                ] = 0.0F;
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
                "Qwen3 LM graph computation failed"
            );
        }
        if (built.output->ne[0] != config_.vocab_size ||
            built.output->ne[1] != 1 ||
            ggml_nelements(built.output) !=
                static_cast<int64_t>(logits.size())) {
            throw Error(
                ErrorCode::InferenceFailed,
                "Qwen3 LM logits shape is invalid"
            );
        }
        ggml_backend_tensor_get(
            built.output,
            logits.data(),
            0,
            logits.size() * sizeof(float)
        );
        cache_length_ = past_length + built.sequence_length;
    } catch (...) {
        ggml_free(context);
        throw;
    }

    ggml_free(context);
    return logits;
}

std::vector<float> Qwen3Lm::prefill(
    const std::vector<std::int32_t> & input_ids,
    const ggml_tensor * audio_embeddings,
    std::size_t audio_start_index,
    std::size_t audio_token_count
) {
    reset_cache();
    if (input_ids.size() > static_cast<std::size_t>(config_.num_ctx)) {
        throw Error(
            ErrorCode::ContextLengthExceeded,
            "Qwen3 LM prompt exceeds num_ctx"
        );
    }
    return run_graph(
        input_ids,
        audio_embeddings,
        audio_start_index,
        audio_token_count,
        0
    );
}

std::vector<float> Qwen3Lm::decode(
    std::int32_t input_id,
    std::int32_t position_id
) {
    if (position_id != cache_length_) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3 LM decode position does not match cache length"
        );
    }
    if (cache_length_ >= config_.num_ctx) {
        throw Error(
            ErrorCode::ContextLengthExceeded,
            "Qwen3 LM KV cache is full"
        );
    }
    const std::vector<std::int32_t> input_ids{input_id};
    return run_graph(input_ids, nullptr, 0, 0, cache_length_);
}

std::vector<std::int32_t> Qwen3Lm::generate(
    const std::vector<std::int32_t> & input_ids,
    const ggml_tensor * audio_embeddings,
    std::size_t audio_start_index,
    std::size_t audio_token_count,
    const Qwen3LmGenerationConfig & generation,
    Qwen3LmGenerationStats * stats
) {
    using Clock = std::chrono::steady_clock;
    if (stats != nullptr) {
        *stats = Qwen3LmGenerationStats{};
        stats->prompt_tokens = static_cast<std::int64_t>(input_ids.size());
    }
    if (!loaded()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3 LM is not loaded"
        );
    }
    if (input_ids.empty()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3 LM generation prompt is empty"
        );
    }
    if (generation.max_new_tokens < 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3 LM max_new_tokens must be non-negative"
        );
    }
    SamplerConfig sampler_config;
    sampler_config.temperature = generation.temperature;
    sampler_config.top_k = generation.top_k;
    sampler_config.top_p = generation.top_p;
    sampler_config.seed = generation.seed;
    Sampler sampler(sampler_config);
    reset_cache();
    if (generation.max_new_tokens == 0) {
        return {};
    }

    const std::uint64_t required_length =
        static_cast<std::uint64_t>(input_ids.size()) +
        static_cast<std::uint64_t>(generation.max_new_tokens);
    const std::uint64_t context_capacity = static_cast<std::uint64_t>(
        std::min(config_.num_ctx, config_.max_position_embeddings)
    );
    if (required_length > context_capacity) {
        throw Error(
            ErrorCode::ContextLengthExceeded,
            "Qwen3 LM generation exceeds the available context"
        );
    }

    const std::vector<std::int32_t> & eos_token_ids =
        generation.eos_token_ids.empty()
            ? generation_eos_token_ids_
            : generation.eos_token_ids;
    const Clock::time_point prefill_start = Clock::now();
    std::vector<float> logits = prefill(
        input_ids,
        audio_embeddings,
        audio_start_index,
        audio_token_count
    );
    if (stats != nullptr) {
        stats->prefill_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - prefill_start
            ).count();
    }
    std::vector<std::int32_t> generated;
    generated.reserve(
        static_cast<std::size_t>(generation.max_new_tokens)
    );

    for (int step = 0; step < generation.max_new_tokens; ++step) {
        const std::int32_t token_id = sampler.sample(logits);
        if (is_generation_eos(token_id, eos_token_ids)) {
            break;
        }
        generated.push_back(token_id);
        if (step + 1 == generation.max_new_tokens) {
            break;
        }
        const Clock::time_point decode_start = Clock::now();
        logits = decode(token_id, static_cast<std::int32_t>(cache_length_));
        if (stats != nullptr) {
            stats->decode_ms +=
                std::chrono::duration<double, std::milli>(
                    Clock::now() - decode_start
                ).count();
            ++stats->decode_calls;
        }
    }

    return generated;
}

void Qwen3Lm::reset_cache() {
    cache_length_ = 0;
}

int64_t Qwen3Lm::cache_length() const noexcept {
    return cache_length_;
}

void Qwen3Lm::release_kv_cache() noexcept {
    if (cache_buffer_ != nullptr) {
        ggml_backend_buffer_free(cache_buffer_);
        cache_buffer_ = nullptr;
    }
    if (cache_context_ != nullptr) {
        ggml_free(cache_context_);
        cache_context_ = nullptr;
    }
    cache_keys_.clear();
    cache_values_.clear();
    cache_length_ = 0;
}

void Qwen3Lm::unload() noexcept {
    if (scheduler_ != nullptr) {
        ggml_backend_sched_reset(scheduler_);
        ggml_backend_sched_free(scheduler_);
        scheduler_ = nullptr;
    }
    release_kv_cache();
    if (weight_buffer_ != nullptr) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_context_ != nullptr) {
        ggml_free(weight_context_);
        weight_context_ = nullptr;
    }
    weights_.clear();
    layer_weights_.clear();
    output_weights_ = {};
    generation_eos_token_ids_.clear();
    config_ = Qwen3LmConfig{};
    session_ = nullptr;
}

bool Qwen3Lm::is_generation_eos(
    std::int32_t token_id,
    const std::vector<std::int32_t> & eos_token_ids
) const {
    return std::find(
        eos_token_ids.begin(),
        eos_token_ids.end(),
        token_id
    ) != eos_token_ids.end();
}

ggml_tensor * Qwen3Lm::weight(std::string_view name) const {
    const auto found = weights_.find(std::string(name));
    return found != weights_.end() ? found->second : nullptr;
}

ggml_tensor * Qwen3Lm::require_weight(std::string_view name) const {
    ggml_tensor * tensor = weight(name);
    if (tensor == nullptr) {
        throw Error(
            ErrorCode::ModelLoadFailed,
            "Qwen3 LM weight is missing: " + std::string(name)
        );
    }
    return tensor;
}

}  // namespace qwen3_asr
