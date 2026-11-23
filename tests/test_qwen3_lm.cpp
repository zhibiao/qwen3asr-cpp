#include "test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "aut_encoder.h"
#include "errors.h"
#include "ggml.h"
#include "ggml_session.h"
#include "gguf_model.h"
#include "qwen3_lm.h"

namespace {

std::vector<std::int32_t> read_i32_binary(
    const std::filesystem::path & path
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open LM golden");
    }
    const std::streamsize byte_count = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<std::int32_t> values(
        static_cast<std::size_t>(byte_count) /
        sizeof(std::int32_t)
    );
    input.read(
        reinterpret_cast<char *>(values.data()),
        byte_count
    );
    return values;
}

std::vector<float> read_f32_binary(
    const std::filesystem::path & path
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open LM golden");
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
    return values;
}

void check_logits(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const std::string & label,
    double mean_limit = 3.0e-2,
    double cosine_limit = 0.9999
) {
    REQUIRE(actual.size() == 151936);
    REQUIRE(actual.size() == expected.size());

    double mean_error = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        mean_error += std::fabs(
            static_cast<double>(actual[index]) -
            static_cast<double>(expected[index])
        );
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
    const auto actual_top = std::max_element(actual.begin(), actual.end());
    const auto expected_top =
        std::max_element(expected.begin(), expected.end());
    std::cerr << label << " mean=" << mean_error
              << " cosine=" << cosine
              << " top=" << (actual_top - actual.begin())
              << "/" << (expected_top - expected.begin())
              << '\n';
    CHECK(mean_error <= mean_limit);
    CHECK(cosine >= cosine_limit);
    CHECK(
        std::distance(actual.begin(), actual_top) ==
        std::distance(expected.begin(), expected_top)
    );
}

QWEN3ASR_TEST_CASE(qwen3_lm_config_defaults_context_to_4096) {
    const qwen3_asr::Qwen3LmConfig config;
    CHECK(config.num_ctx == 4096);
}

QWEN3ASR_TEST_CASE(qwen3_lm_prefill_and_decode_match_goldens) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    qwen3_asr::Qwen3LmConfig config = pair.lm.text;
    config.num_ctx = 128;
    qwen3_asr::Qwen3Lm lm;
    lm.load(config, pair.lm, session);

    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const std::vector<std::int32_t> text_ids =
        read_i32_binary(fixture_dir / "text_lm_input_ids.i32");
    const std::vector<std::int32_t> decode_token =
        read_i32_binary(fixture_dir / "text_lm_decode_token.i32");
    REQUIRE(decode_token.size() == 1);
    const std::vector<float> expected_prefill =
        read_f32_binary(fixture_dir / "text_lm_prefill_logits.f32");
    const std::vector<float> expected_decode =
        read_f32_binary(fixture_dir / "text_lm_decode_logits.f32");

    check_logits(
        lm.prefill(text_ids, nullptr, 0, 0),
        expected_prefill,
        "text prefill",
        2.0e-2,
        0.99999
    );
    REQUIRE(lm.cache_length() == 16);
    check_logits(
        lm.decode(decode_token[0], 16),
        expected_decode,
        "text decode",
        2.0e-2,
        0.99999
    );
    REQUIRE(lm.cache_length() == 17);
    bool rejected_position = false;
    try {
        static_cast<void>(lm.decode(decode_token[0], 15));
    } catch (const qwen3_asr::Error & error) {
        rejected_position =
            error.code() == qwen3_asr::ErrorCode::InvalidArgument;
    }
    CHECK(rejected_position);
    lm.reset_cache();
    CHECK(lm.cache_length() == 0);

    qwen3_asr::Qwen3LmGenerationConfig generation;
    generation.max_new_tokens = 2;
    generation.eos_token_ids = {-1};
    generation.temperature = 1.0F;
    generation.top_k = 1;
    generation.seed = 7;
    const std::vector<std::int32_t> generated = lm.generate(
        text_ids,
        nullptr,
        0,
        0,
        generation
    );
    REQUIRE(generated.size() == 2);
    const auto expected_first = std::distance(
        expected_prefill.begin(),
        std::max_element(expected_prefill.begin(), expected_prefill.end())
    );
    CHECK(generated[0] == expected_first);
    CHECK(lm.cache_length() == static_cast<int64_t>(text_ids.size() + 1));

    generation.max_new_tokens = 200;
    bool rejected_context = false;
    try {
        static_cast<void>(lm.generate(
            text_ids,
            nullptr,
            0,
            0,
            generation
        ));
    } catch (const qwen3_asr::Error & error) {
        rejected_context =
            error.code() == qwen3_asr::ErrorCode::ContextLengthExceeded;
    }
    CHECK(rejected_context);
}

QWEN3ASR_TEST_CASE(qwen3_lm_prefill_uses_encoder_tensor_without_copy) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    qwen3_asr::AutEncoder encoder;
    encoder.load(pair.encoder.audio, pair.encoder, session);
    qwen3_asr::Qwen3LmConfig config = pair.lm.text;
    config.num_ctx = 128;
    qwen3_asr::Qwen3Lm lm;
    lm.load(config, pair.lm, session);

    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    qwen3_asr::LogMelFeatures features;
    features.n_mel_bins = 128;
    features.n_frames = 525;
    features.data = read_f32_binary(
        fixture_dir / "logmel_ar1_128x525.f32"
    );
    ggml_tensor * audio = encoder.forward(features);
    REQUIRE(audio != nullptr);
    REQUIRE(audio->ne[0] == 1024);
    REQUIRE(audio->ne[1] == 69);

    const std::vector<std::int32_t> audio_ids =
        read_i32_binary(fixture_dir / "audio_lm_input_ids.i32");
    check_logits(
        lm.prefill(audio_ids, audio, 9, 69),
        read_f32_binary(fixture_dir / "audio_lm_prefill_logits.f32"),
        "audio prefill",
        1.0e-2,
        0.99999
    );
    REQUIRE(lm.cache_length() ==
            static_cast<int64_t>(audio_ids.size()));
    const std::vector<std::int32_t> decode_token =
        read_i32_binary(fixture_dir / "text_lm_decode_token.i32");
    check_logits(
        lm.decode(
            decode_token[0],
            static_cast<std::int32_t>(audio_ids.size())
        ),
        read_f32_binary(fixture_dir / "audio_lm_decode_logits.f32"),
        "audio decode",
        1.0e-2,
        0.99999
    );
}

QWEN3ASR_TEST_CASE(qwen3_lm_long_prefill_and_decode_match_goldens) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::GgmlSession session;
    qwen3_asr::AutEncoder encoder;
    encoder.load(pair.encoder.audio, pair.encoder, session);
    qwen3_asr::Qwen3LmConfig config = pair.lm.text;
    config.num_ctx = 384;
    qwen3_asr::Qwen3Lm lm;
    lm.load(config, pair.lm, session);

    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";
    const std::int32_t decode_token =
        read_i32_binary(fixture_dir / "text_lm_decode_token.i32")[0];

    const std::vector<std::int32_t> long_text_ids =
        read_i32_binary(fixture_dir / "long_text_lm_input_ids.i32");
    REQUIRE(long_text_ids.size() == 192);
    check_logits(
        lm.prefill(long_text_ids, nullptr, 0, 0),
        read_f32_binary(fixture_dir / "long_text_lm_prefill_logits.f32"),
        "long text prefill",
        1.0e-2,
        0.99999
    );
    REQUIRE(lm.cache_length() == 192);
    check_logits(
        lm.decode(decode_token, 192),
        read_f32_binary(fixture_dir / "long_text_lm_decode_logits.f32"),
        "long text decode",
        1.0e-2,
        0.99999
    );

    qwen3_asr::LogMelFeatures features;
    features.n_mel_bins = 128;
    features.n_frames = 2076;
    features.data = read_f32_binary(
        fixture_dir / "logmel_raokouling_128x2076.f32"
    );
    ggml_tensor * audio = encoder.forward(features);
    REQUIRE(audio != nullptr);
    REQUIRE(audio->ne[0] == 1024);
    REQUIRE(audio->ne[1] == 270);

    const std::vector<std::int32_t> long_audio_ids =
        read_i32_binary(fixture_dir / "long_audio_lm_input_ids.i32");
    REQUIRE(long_audio_ids.size() == 285);
    check_logits(
        lm.prefill(long_audio_ids, audio, 9, 270),
        read_f32_binary(fixture_dir / "long_audio_lm_prefill_logits.f32"),
        "long audio prefill",
        1.0e-2,
        0.99999
    );
    REQUIRE(lm.cache_length() == 285);
    check_logits(
        lm.decode(decode_token, 285),
        read_f32_binary(fixture_dir / "long_audio_lm_decode_logits.f32"),
        "long audio decode",
        1.0e-1,
        0.9995
    );
}

QWEN3ASR_TEST_CASE(qwen3_lm_loads_weights_into_backend_buffer) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.lm;
    qwen3_asr::GgmlSession session;
    qwen3_asr::Qwen3Lm lm;
    lm.load(model.text, model, session);

    CHECK(lm.loaded());
    CHECK(lm.weight_count() == 311);
    CHECK(lm.layer_weights().size() == 28);
    CHECK(lm.native_context() != nullptr);
    CHECK(lm.native_weight_buffer() != nullptr);
    CHECK(lm.scheduler() != nullptr);
    CHECK(lm.output_weights().embed_tokens_weight != nullptr);
    CHECK(lm.output_weights().final_norm_weight != nullptr);
    CHECK(lm.output_weights().lm_head_weight != nullptr);
    CHECK(lm.layer_weights().front().input_layernorm_weight != nullptr);
    CHECK(lm.layer_weights().back().mlp_down_proj_weight != nullptr);

    const auto descriptor = std::find_if(
        model.tensors.begin(),
        model.tensors.end(),
        [](const qwen3_asr::GgufTensorDescriptor & value) {
            return value.name == "thinker.lm_head.weight";
        }
    );
    REQUIRE(descriptor != model.tensors.end());
    const ggml_tensor * weight = lm.require_weight(
        "thinker.lm_head.weight"
    );
    CHECK(weight->type == static_cast<ggml_type>(descriptor->type));
    CHECK(ggml_nbytes(weight) == descriptor->byte_count);
    CHECK(std::strcmp(
        ggml_get_name(weight),
        "thinker.lm_head.weight"
    ) == 0);
}

QWEN3ASR_TEST_CASE(qwen3_lm_unload_releases_backend_buffer) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.lm;
    qwen3_asr::GgmlSession session;
    qwen3_asr::Qwen3Lm lm;
    lm.load(model.text, model, session);
    lm.unload();

    CHECK(!lm.loaded());
    CHECK(lm.weight_count() == 0);
    CHECK(lm.layer_weights().empty());
    CHECK(lm.native_context() == nullptr);
    CHECK(lm.native_weight_buffer() == nullptr);
    CHECK(lm.scheduler() == nullptr);
}

QWEN3ASR_TEST_CASE(qwen3_lm_rejects_uninitialized_session) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.lm;
    qwen3_asr::GgmlSession session;
    session.reset();
    qwen3_asr::Qwen3Lm lm;
    bool threw = false;
    try {
        lm.load(model.text, model, session);
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() ==
            qwen3_asr::ErrorCode::BackendUnavailable;
    }
    CHECK(threw);
    CHECK(!lm.loaded());
}

}  // namespace
