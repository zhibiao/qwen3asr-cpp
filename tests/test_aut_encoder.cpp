#include "test_framework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aut_encoder.h"
#include "errors.h"
#include "ggml.h"
#include "ggml_session.h"
#include "gguf_model.h"

namespace {

std::vector<float> read_float_binary(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open Audio Encoder golden: " + path.string()
        );
    }
    const std::streamsize byte_count = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<float> values(
        static_cast<std::size_t>(byte_count) / sizeof(float)
    );
    input.read(
        reinterpret_cast<char *>(values.data()),
        byte_count
    );
    if (!input) {
        throw std::runtime_error("failed to read Audio Encoder golden");
    }
    return values;
}

void check_float_golden(
    ggml_tensor * actual,
    const std::filesystem::path & expected_path,
    const std::string & label,
    double max_tolerance = 5.0e-3
) {
    REQUIRE(actual != nullptr);
    const std::vector<float> expected =
        read_float_binary(expected_path);
    std::vector<float> actual_data(ggml_nelements(actual));
    ggml_backend_tensor_get(
        actual,
        actual_data.data(),
        0,
        actual_data.size() * sizeof(float)
    );
    REQUIRE(actual_data.size() == expected.size());

    double mean_error = 0.0;
    double max_error = 0.0;
    std::size_t max_error_index = 0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = std::fabs(
            static_cast<double>(actual_data[index]) -
            static_cast<double>(expected[index])
        );
        mean_error += error;
        if (error > max_error) {
            max_error = error;
            max_error_index = index;
        }
        dot += static_cast<double>(actual_data[index]) *
            static_cast<double>(expected[index]);
        actual_norm += static_cast<double>(actual_data[index]) *
            static_cast<double>(actual_data[index]);
        expected_norm += static_cast<double>(expected[index]) *
            static_cast<double>(expected[index]);
    }
    mean_error /= static_cast<double>(expected.size());
    const double cosine = dot / std::sqrt(actual_norm * expected_norm);
    double actual_mean = 0.0;
    for (float value : actual_data) {
        actual_mean += value;
    }
    actual_mean /= static_cast<double>(actual_data.size());
    double actual_variance = 0.0;
    for (float value : actual_data) {
        const double delta = value - actual_mean;
        actual_variance += delta * delta;
    }
    actual_variance /= static_cast<double>(actual_data.size());
    std::cerr << label << " mean=" << mean_error
              << " max=" << max_error
              << " max_index=" << max_error_index
              << " cosine=" << cosine
              << " actual_mean=" << actual_mean
              << " actual_std=" << std::sqrt(actual_variance) << '\n';
    CHECK(mean_error <= 5.0e-4);
    CHECK(max_error <= max_tolerance);
    CHECK(cosine >= 0.9999);
}

QWEN3ASR_TEST_CASE(aut_encoder_frontend_matches_pytorch_golden) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    ggml_backend_t backends[1] = {session.native_backend()};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
        backends,
        nullptr,
        1,
        session.graph_size(),
        false,
        true
    );
    REQUIRE(scheduler != nullptr);
    qwen3_asr::AutEncoder encoder;
    encoder.load(pair.encoder.audio, pair.encoder, session);

    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const std::vector<float> features = read_float_binary(
        fixture_dir / "logmel_ar1_128x525.f32"
    );
    std::vector<float> padded_features(6 * 128 * 100, 0.0F);
    for (int chunk = 0; chunk < 6; ++chunk) {
        const int frame_offset = chunk * 100;
        const int frame_count = std::min(100, 525 - frame_offset);
        for (int mel = 0; mel < 128; ++mel) {
            for (int frame = 0; frame < frame_count; ++frame) {
                padded_features[
                    static_cast<std::size_t>(
                        (chunk * 128 + mel) * 100 + frame
                    )
                ] = features[
                    static_cast<std::size_t>(mel * 525 +
                        frame_offset + frame)
                ];
            }
        }
    }

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor * input_features = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        600,
        128
    );
    ggml_set_input(input_features);
    ggml_tensor * input = ggml_reshape_4d(
        context,
        input_features,
        100,
        128,
        1,
        6
    );
    ggml_tensor * output = nullptr;
    const auto conv_layer = [
        context
    ](
        ggml_tensor * value,
        ggml_tensor * weight,
        ggml_tensor * bias
    ) {
        value = ggml_conv_2d(
            context,
            ggml_cast(context, weight, GGML_TYPE_F32),
            value,
            2,
            2,
            1,
            1,
            1,
            1
        );
        bias = ggml_reshape_4d(context, bias, 1, 1, 480, 1);
        bias = ggml_cast(context, bias, GGML_TYPE_F32);
        value = ggml_add(
            context,
            value,
            ggml_repeat(context, bias, value)
        );
        return ggml_gelu_erf(context, value);
    };
    output = conv_layer(
        input,
        encoder.conv_weights().conv2d1_weight,
        encoder.conv_weights().conv2d1_bias
    );
    output = conv_layer(
        output,
        encoder.conv_weights().conv2d2_weight,
        encoder.conv_weights().conv2d2_bias
    );
    output = conv_layer(
        output,
        encoder.conv_weights().conv2d3_weight,
        encoder.conv_weights().conv2d3_bias
    );
    ggml_tensor * conv_features = ggml_cont(
        context,
        ggml_permute(context, output, 2, 0, 1, 3)
    );
    output = ggml_reshape_3d(context, conv_features, 7680, 13, 6);
    output = ggml_mul_mat(
        context,
        encoder.conv_weights().conv_out_weight,
        output
    );
    ggml_tensor * positions = ggml_view_2d(
        context,
        encoder.positional_embedding(),
        896,
        13,
        encoder.positional_embedding()->nb[1],
        0
    );
    positions = ggml_reshape_3d(context, positions, 896, 13, 1);
    output = ggml_add(
        context,
        output,
        ggml_repeat(context, positions, output)
    );
    output = ggml_reshape_2d(context, output, 896, 78);
    ggml_tensor * valid_indices = ggml_new_tensor_1d(
        context,
        GGML_TYPE_I32,
        69
    );
    ggml_set_input(valid_indices);
    output = ggml_get_rows(context, output, valid_indices);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_reset(scheduler);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));
    ggml_backend_tensor_set(
        input_features,
        padded_features.data(),
        0,
        padded_features.size() * sizeof(float)
    );
    std::vector<std::int32_t> indices(69);
    for (int index = 0; index < 69; ++index) {
        indices[static_cast<std::size_t>(index)] = index;
    }
    ggml_backend_tensor_set(
        valid_indices,
        indices.data(),
        0,
        indices.size() * sizeof(std::int32_t)
    );
    REQUIRE(
        ggml_backend_sched_graph_compute(
            scheduler,
            graph
        ) == GGML_STATUS_SUCCESS
    );
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(
        output,
        actual.data(),
        0,
        actual.size() * sizeof(float)
    );
    const std::vector<float> expected = read_float_binary(
        fixture_dir / "audio_conv_ar1_896x69.f32"
    );
    REQUIRE(actual.size() == 896 * 69);
    REQUIRE(expected.size() == 896 * 69);
    double mean_error = 0.0;
    double max_error = 0.0;
    for (int frame = 0; frame < 69; ++frame) {
        for (int channel = 0; channel < 896; ++channel) {
            const double actual_value = actual[
                static_cast<std::size_t>(frame) * 896 + channel
            ];
            const double expected_value = expected[
                static_cast<std::size_t>(frame) * 896 + channel
            ];
        const double error = std::fabs(
                actual_value - expected_value
        );
        mean_error += error;
        max_error = std::max(max_error, error);
        }
    }
    mean_error /= static_cast<double>(expected.size());
    std::cerr << "Audio Encoder frontend error: mean=" << mean_error
              << " max=" << max_error << '\n';
    CHECK(mean_error <= 1.0e-3);
    CHECK(max_error <= 5.0e-3);
    ggml_backend_sched_reset(scheduler);
    ggml_backend_sched_free(scheduler);
    ggml_free(context);
}

QWEN3ASR_TEST_CASE(aut_encoder_transformer_matches_pytorch_golden) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    ggml_backend_t backends[1] = {session.native_backend()};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
        backends,
        nullptr,
        1,
        session.graph_size(),
        false,
        true
    );
    REQUIRE(scheduler != nullptr);
    qwen3_asr::AutEncoder encoder;
    encoder.load(pair.encoder.audio, pair.encoder, session);

    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const std::vector<float> source = read_float_binary(
        fixture_dir / "audio_conv_ar1_896x69.f32"
    );
    const std::vector<float> expected = read_float_binary(
        fixture_dir / "audio_embedding_ar1_1024x69.f32"
    );
    REQUIRE(source.size() == 896 * 69);
    REQUIRE(expected.size() == 1024 * 69);

    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor * input = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        896,
        69
    );
    ggml_set_input(input);

    const auto cast_f32 = [context](ggml_tensor * value) {
        return value->type == GGML_TYPE_F32 ?
            value : ggml_cast(context, value, GGML_TYPE_F32);
    };
    const auto add_bias = [context, &cast_f32](
        ggml_tensor * value,
        ggml_tensor * bias
    ) {
        return ggml_add(
            context,
            value,
            ggml_repeat(context, cast_f32(bias), value)
        );
    };
    const auto linear = [context, &add_bias](
        ggml_tensor * value,
        ggml_tensor * weight,
        ggml_tensor * bias
    ) {
        value = ggml_mul_mat(context, weight, value);
        return bias != nullptr ? add_bias(value, bias) : value;
    };
    const auto layer_norm = [context, &cast_f32](
        ggml_tensor * value,
        ggml_tensor * weight,
        ggml_tensor * bias
    ) {
        value = ggml_norm(context, value, 1.0e-5F);
        ggml_tensor * gamma = ggml_reshape_2d(
            context,
            cast_f32(weight),
            896,
            1
        );
        ggml_tensor * beta = ggml_reshape_2d(
            context,
            cast_f32(bias),
            896,
            1
        );
        value = ggml_mul(
            context,
            value,
            ggml_repeat(context, gamma, value)
        );
        return ggml_add(
            context,
            value,
            ggml_repeat(context, beta, value)
        );
    };

    const auto build_layer = [
        context,
        &linear,
        &layer_norm
    ](
        ggml_tensor * value,
        const qwen3_asr::AutEncoderLayerWeights & layer
    ) {
        ggml_tensor * residual = value;
        ggml_tensor * normalized = layer_norm(
            value,
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
        ggml_tensor * layer_value = linear(
            normalized,
            layer.self_attn_v_proj_weight,
            layer.self_attn_v_proj_bias
        );
        query = ggml_permute(
            context,
            ggml_reshape_3d(context, query, 64, 14, 69),
            0,
            2,
            1,
            3
        );
        key = ggml_permute(
            context,
            ggml_reshape_3d(context, key, 64, 14, 69),
            0,
            2,
            1,
            3
        );
        layer_value = ggml_permute(
            context,
            ggml_reshape_3d(context, layer_value, 64, 14, 69),
            0,
            2,
            1,
            3
        );
        ggml_tensor * attention = ggml_mul_mat(
            context,
            key,
            query
        );
        attention = ggml_soft_max(
            context,
            ggml_scale(context, attention, 1.0F / 8.0F)
        );
        ggml_tensor * value_transposed = ggml_cont_3d(
            context,
            ggml_permute(context, layer_value, 1, 0, 2, 3),
            69,
            64,
            14
        );
        ggml_tensor * attended = ggml_mul_mat(
            context,
            value_transposed,
            attention
        );
        attended = ggml_cont_2d(
            context,
            ggml_permute(context, attended, 0, 2, 1, 3),
            896,
            69
        );
        attended = linear(
            attended,
            layer.self_attn_out_proj_weight,
            layer.self_attn_out_proj_bias
        );
        value = ggml_add(context, residual, attended);
        residual = value;
        normalized = layer_norm(
            value,
            layer.final_layer_norm_weight,
            layer.final_layer_norm_bias
        );
        value = linear(normalized, layer.fc1_weight, layer.fc1_bias);
        value = ggml_gelu_erf(context, value);
        value = linear(value, layer.fc2_weight, layer.fc2_bias);
        return ggml_add(context, residual, value);
    };

    ggml_tensor * hidden = input;
    for (const qwen3_asr::AutEncoderLayerWeights & layer :
         encoder.layer_weights()) {
        hidden = build_layer(hidden, layer);
    }
    hidden = layer_norm(
        hidden,
        encoder.output_weights().ln_post_weight,
        encoder.output_weights().ln_post_bias
    );
    hidden = linear(
        hidden,
        encoder.output_weights().proj1_weight,
        encoder.output_weights().proj1_bias
    );
    hidden = ggml_gelu_erf(context, hidden);
    hidden = linear(
        hidden,
        encoder.output_weights().proj2_weight,
        encoder.output_weights().proj2_bias
    );

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, hidden);
    ggml_backend_sched_reset(scheduler);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));
    ggml_backend_tensor_set(
        input,
        source.data(),
        0,
        source.size() * sizeof(float)
    );
    REQUIRE(
        ggml_backend_sched_graph_compute(
            scheduler,
            graph
        ) == GGML_STATUS_SUCCESS
    );
    std::vector<float> actual(ggml_nelements(hidden));
    ggml_backend_tensor_get(
        hidden,
        actual.data(),
        0,
        actual.size() * sizeof(float)
    );
    double mean_error = 0.0;
    double max_error = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = std::fabs(
            static_cast<double>(actual[index]) -
            static_cast<double>(expected[index])
        );
        mean_error += error;
        max_error = std::max(max_error, error);
        dot += static_cast<double>(actual[index]) *
            static_cast<double>(expected[index]);
        actual_norm += static_cast<double>(actual[index]) *
            static_cast<double>(actual[index]);
        expected_norm += static_cast<double>(expected[index]) *
            static_cast<double>(expected[index]);
    }
    mean_error /= static_cast<double>(expected.size());
    const double cosine = dot /
        std::sqrt(actual_norm * expected_norm);
    std::cerr << "Audio Encoder transformer error: mean=" << mean_error
              << " max=" << max_error
              << " cosine=" << cosine << '\n';
    CHECK(mean_error <= 1.0e-3);
    CHECK(max_error <= 5.0e-3);
    CHECK(cosine >= 0.9999);
    ggml_backend_sched_reset(scheduler);
    ggml_backend_sched_free(scheduler);
    ggml_free(context);
}

QWEN3ASR_TEST_CASE(aut_encoder_loads_weights_into_backend_buffer) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.encoder;
    qwen3_asr::GgmlSession session;
    qwen3_asr::AutEncoder encoder;
    encoder.load(model.audio, model, session);

    CHECK(encoder.loaded());
    CHECK(encoder.weight_count() == 301);
    CHECK(encoder.layer_weights().size() == 18);
    CHECK(encoder.scheduler() != nullptr);
    CHECK(encoder.native_context() != nullptr);
    CHECK(encoder.native_weight_buffer() != nullptr);
    const qwen3_asr::AutEncoderShape shape =
        encoder.output_shape(525);
    CHECK(shape.input_padded_frames == 600);
    CHECK(shape.padded_frames == 78);
    CHECK(shape.valid_frames == 69);
    CHECK(shape.attention_window_size == 104);
    CHECK(shape.attention_windows == 1);

    const qwen3_asr::AutEncoderShape twenty_minutes =
        encoder.output_shape(120000);
    CHECK(twenty_minutes.input_padded_frames == 120000);
    CHECK(twenty_minutes.padded_frames == 15600);
    CHECK(twenty_minutes.valid_frames == 15600);
    CHECK(twenty_minutes.attention_window_size == 104);
    CHECK(twenty_minutes.attention_windows == 150);

    ggml_init_params graph_params{};
    graph_params.mem_size = 16 * 1024 * 1024;
    graph_params.mem_buffer = nullptr;
    graph_params.no_alloc = true;
    ggml_context * graph_context = ggml_init(graph_params);
    REQUIRE(graph_context != nullptr);
    const qwen3_asr::AutEncoderGraph long_graph =
        encoder.build_graph(graph_context, 120000);
    REQUIRE(long_graph.graph != nullptr);
    CHECK(long_graph.attention_mask->ne[0] == 104);
    CHECK(long_graph.attention_mask->ne[1] == 104);
    CHECK(long_graph.attention_mask->ne[2] == 1);
    CHECK(long_graph.attention_mask->ne[3] == 150);
    CHECK(long_graph.window_frame_indices->ne[0] == 15600);
    CHECK(long_graph.output->ne[0] == 1024);
    CHECK(long_graph.output->ne[1] == 15600);
    int flash_attention_nodes = 0;
    for (int index = 0;
         index < ggml_graph_n_nodes(long_graph.graph);
         ++index) {
        if (ggml_graph_node(long_graph.graph, index)->op ==
            GGML_OP_FLASH_ATTN_EXT) {
            ++flash_attention_nodes;
        }
    }
    CHECK(flash_attention_nodes == 18);
    ggml_free(graph_context);
    CHECK(encoder.conv_weights().conv2d1_weight != nullptr);
    CHECK(encoder.conv_weights().conv2d3_weight != nullptr);
    CHECK(encoder.output_weights().proj2_weight != nullptr);
    CHECK(encoder.layer_weights().front().fc1_weight != nullptr);
    CHECK(encoder.layer_weights().back().fc2_weight != nullptr);

    const auto descriptor = std::find_if(
        model.tensors.begin(),
        model.tensors.end(),
        [](const qwen3_asr::GgufTensorDescriptor & value) {
            return value.name ==
                "thinker.audio_tower.conv2d1.weight";
        }
    );
    REQUIRE(descriptor != model.tensors.end());
    const ggml_tensor * weight = encoder.require_weight(
        "thinker.audio_tower.conv2d1.weight"
    );
    CHECK(weight->type == static_cast<ggml_type>(descriptor->type));
    CHECK(ggml_nbytes(weight) == descriptor->byte_count);
    CHECK(std::strcmp(
        ggml_get_name(weight),
        "thinker.audio_tower.conv2d1.weight"
    ) == 0);

    std::array<std::uint16_t, 8> weight_prefix{};
    ggml_backend_tensor_get(
        weight,
        weight_prefix.data(),
        0,
        sizeof(weight_prefix)
    );
    constexpr std::array<std::uint16_t, 8> expected_prefix{
        47969, 48408, 48625, 48416, 48278, 16060, 15272, 48411,
    };
    CHECK(weight_prefix == expected_prefix);
}

QWEN3ASR_TEST_CASE(aut_encoder_unload_releases_backend_buffer) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.encoder;
    qwen3_asr::GgmlSession session;
    qwen3_asr::AutEncoder encoder;
    encoder.load(model.audio, model, session);
    encoder.unload();

    CHECK(!encoder.loaded());
    CHECK(encoder.weight_count() == 0);
    CHECK(encoder.layer_weights().empty());
    CHECK(encoder.scheduler() == nullptr);
    CHECK(encoder.native_context() == nullptr);
    CHECK(encoder.native_weight_buffer() == nullptr);
}

QWEN3ASR_TEST_CASE(aut_encoder_rejects_uninitialized_session) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.encoder;
    qwen3_asr::GgmlSession session;
    session.reset();
    qwen3_asr::AutEncoder encoder;
    bool threw = false;
    try {
        encoder.load(model.audio, model, session);
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() ==
            qwen3_asr::ErrorCode::BackendUnavailable;
    }
    CHECK(threw);
    CHECK(!encoder.loaded());
}

QWEN3ASR_TEST_CASE(aut_encoder_forward_matches_pytorch_golden) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    qwen3_asr::AutEncoder encoder;
    encoder.load(pair.encoder.audio, pair.encoder, session);
    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    qwen3_asr::LogMelFeatures features;
    features.n_mel_bins = 128;
    features.n_frames = 525;
    features.data = read_float_binary(
        fixture_dir / "logmel_ar1_128x525.f32"
    );
    ggml_tensor * actual = encoder.forward(features);
    REQUIRE(actual != nullptr);
    REQUIRE(actual->buffer != nullptr);
    REQUIRE(actual->type == GGML_TYPE_F32);
    REQUIRE(actual->ne[0] == 1024);
    REQUIRE(actual->ne[1] == 69);

    ggml_backend_t lm_backends[1] = {session.native_backend()};
    ggml_backend_sched_t lm_scheduler = ggml_backend_sched_new(
        lm_backends,
        nullptr,
        1,
        session.graph_size(),
        false,
        true
    );
    REQUIRE(lm_scheduler != nullptr);
    ggml_init_params dummy_params{};
    dummy_params.mem_size = 1024 * 1024;
    dummy_params.mem_buffer = nullptr;
    dummy_params.no_alloc = true;
    ggml_context * dummy_context = ggml_init(dummy_params);
    REQUIRE(dummy_context != nullptr);
    ggml_tensor * dummy_input = ggml_new_tensor_1d(
        dummy_context,
        GGML_TYPE_F32,
        1
    );
    ggml_tensor * dummy_output = ggml_scale(
        dummy_context,
        dummy_input,
        2.0F
    );
    ggml_cgraph * dummy_graph = ggml_new_graph(dummy_context);
    ggml_build_forward_expand(dummy_graph, dummy_output);
    REQUIRE(
        ggml_backend_sched_alloc_graph(lm_scheduler, dummy_graph)
    );
    const float dummy_value = 1.0F;
    ggml_backend_tensor_set(
        dummy_input,
        &dummy_value,
        0,
        sizeof(dummy_value)
    );
    REQUIRE(
        ggml_backend_sched_graph_compute(
            lm_scheduler,
            dummy_graph
        ) == GGML_STATUS_SUCCESS
    );
    ggml_backend_sched_reset(lm_scheduler);
    ggml_backend_sched_free(lm_scheduler);
    ggml_free(dummy_context);

    check_float_golden(
        actual,
        fixture_dir / "audio_embedding_ar1_1024x69.f32",
        "Audio Encoder ar1"
    );
}

QWEN3ASR_TEST_CASE(aut_encoder_forward_matches_multi_window_golden) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    qwen3_asr::AutEncoder encoder;
    encoder.load(pair.encoder.audio, pair.encoder, session);

    const qwen3_asr::AutEncoderShape shape =
        encoder.output_shape(2076);
    CHECK(shape.valid_frames == 270);
    CHECK(shape.attention_window_size == 104);
    CHECK(shape.attention_windows == 3);

    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    qwen3_asr::LogMelFeatures features;
    features.n_mel_bins = 128;
    features.n_frames = 2076;
    features.data = read_float_binary(
        fixture_dir / "logmel_raokouling_128x2076.f32"
    );
    ggml_tensor * actual = encoder.forward(features);
    REQUIRE(actual != nullptr);
    REQUIRE(actual->ne[0] == 1024);
    REQUIRE(actual->ne[1] == 270);
    check_float_golden(
        actual,
        fixture_dir / "audio_embedding_raokouling_1024x270.f32",
        "Audio Encoder raokouling",
        2.0e-2
    );
}

}  // namespace
