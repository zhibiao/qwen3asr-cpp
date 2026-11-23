#include "test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "audio_frontend.h"
#include "errors.h"
#include "gguf_model.h"
#include "prompt_builder.h"
#include "text_tokenizer.h"

namespace {

std::vector<float> read_float_binary(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "failed to open fixture: " + path.string()
        );
    }
    const std::streamsize size = input.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float))) {
        throw std::runtime_error("invalid float fixture");
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(
        static_cast<std::size_t>(size) / sizeof(float)
    );
    input.read(reinterpret_cast<char *>(values.data()), size);
    if (!input) {
        throw std::runtime_error("failed to read float fixture");
    }
    return values;
}

nlohmann::json read_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "failed to open JSON fixture: " + path.string()
        );
    }
    nlohmann::json value;
    input >> value;
    return value;
}

double maximum_error(
    const std::vector<float> & actual,
    const std::vector<float> & expected
) {
    if (actual.size() != expected.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result = std::max(
            result,
            std::fabs(
                static_cast<double>(actual[index]) -
                static_cast<double>(expected[index])
            )
        );
    }
    return result;
}

QWEN3ASR_TEST_CASE(gguf_model_loads_basic_and_audio_metadata) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & encoder = pair.encoder;
    const qwen3_asr::GgufModel & lm = pair.lm;

    CHECK(encoder.format_version == 2);
    CHECK(encoder.part == "encoder");
    CHECK(!encoder.path.empty());
    CHECK(!encoder.bundle_id.empty());
    CHECK(encoder.gf_name == "Qwen3-ASR-0.6B");
    CHECK(encoder.model_family == "Qwen3-ASR");
    CHECK(encoder.architecture == "Qwen3ASRForConditionalGeneration");
    CHECK(encoder.supported_languages.size() == 30);
    CHECK(encoder.weight_layout == "huggingface_safetensors_v1");
    CHECK(encoder.weight_dtype == "BF16");
    CHECK(encoder.weight_tensor_count == 301);
    CHECK(encoder.audio_weight_tensor_count == 301);
    CHECK(encoder.tensors.size() == 302);

    CHECK(encoder.audio.d_model == 896);
    CHECK(encoder.audio.encoder_layers == 18);
    CHECK(encoder.audio.encoder_attention_heads == 14);
    CHECK(encoder.audio.num_mel_bins == 128);
    CHECK(encoder.audio.n_window == 50);
    CHECK(encoder.audio.n_window_infer == 800);
    CHECK(encoder.audio.output_dim == 1024);
    CHECK(encoder.audio.downsample_hidden_size == 480);
    CHECK(encoder.audio.conv_chunksize == 500);
    CHECK(encoder.audio.activation_function == "gelu");
    CHECK(encoder.preprocessor.sample_rate == 16000);
    CHECK(encoder.preprocessor.feature_size == 128);
    CHECK(encoder.preprocessor.n_fft == 400);
    CHECK(encoder.preprocessor.hop_length == 160);
    CHECK(encoder.preprocessor.mel_scale == "slaney");
    CHECK(encoder.preprocessor.mel_norm == "slaney");

    CHECK(lm.format_version == 2);
    CHECK(lm.part == "lm");
    CHECK(!lm.path.empty());
    CHECK(lm.bundle_id == encoder.bundle_id);
    CHECK(lm.weight_dtype == "BF16");
    CHECK(lm.weight_tensor_count == 311);
    CHECK(lm.text_weight_tensor_count == 310);
    CHECK(lm.lm_head_weight_tensor_count == 1);
    CHECK(lm.tensors.size() == 311);
    CHECK(lm.text.hidden_size == 1024);
    CHECK(lm.text.layers == 28);
    CHECK(lm.text.attention_heads == 16);
    CHECK(lm.text.kv_heads == 8);
    CHECK(lm.text.head_dim == 128);
    CHECK(lm.text.max_position_embeddings == 65536);
    CHECK(lm.text.num_ctx == 4096);
    CHECK(lm.text.intermediate_size == 3072);
    CHECK(lm.text.tie_word_embeddings);
    CHECK(lm.text.mrope_section.size() == 3);
    CHECK(lm.tokenizer_data.vocab.size() == 151705);
    CHECK(lm.tokenizer_data.added_tokens.size() == 62);
    CHECK(lm.tokenizer_model == "gpt2");
    CHECK(lm.tokenizer_pre == "qwen2");
    CHECK(lm.prompt_version == "qwen3_asr_transcription_v1");
}

QWEN3ASR_TEST_CASE(gguf_model_restores_mel_filters_and_log_mel) {
    const qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    const qwen3_asr::GgufModel & model = pair.encoder;
    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";

    const std::vector<float> expected_filters =
        read_float_binary(fixture_dir / "mel_filters_201x128.f32");
    CHECK(model.mel_filters.n_frequency_bins == 201);
    CHECK(model.mel_filters.n_mel_bins == 128);
    CHECK(maximum_error(model.mel_filters.data, expected_filters) <= 1.0e-7);

    qwen3_asr::MonoAudio audio;
    audio.sample_rate = model.frontend_config.sampling_rate;
    audio.samples = read_float_binary(
        fixture_dir / "pcm16k_ar1.f32"
    );
    const qwen3_asr::AudioFrontend frontend(
        model.frontend_config,
        model.mel_filters
    );
    const qwen3_asr::LogMelFeatures actual = frontend.extract(audio);
    const std::vector<float> expected_log_mel =
        read_float_binary(fixture_dir / "logmel_ar1_128x525.f32");
    CHECK(actual.n_mel_bins == 128);
    CHECK(actual.n_frames == 525);
    CHECK(maximum_error(actual.data, expected_log_mel) <= 5.0e-5);

    for (const std::string name : {"ar1.wav", "ja1.wav"}) {
        const qwen3_asr::AudioBuffer decoded =
            qwen3_asr::AudioLoader::load(
                std::filesystem::path(QWEN3ASR_TEST_WAV_DIR) / name
            );
        qwen3_asr::MonoAudio wav_audio =
            qwen3_asr::AudioProcessor::prepare_mono(decoded);
        wav_audio = qwen3_asr::AudioResampler::resample(
            wav_audio,
            model.frontend_config.sampling_rate
        );
        const qwen3_asr::LogMelFeatures wav_features =
            frontend.extract(wav_audio);

        const std::string stem =
            std::filesystem::path(name).stem().string();
        const std::string expected_name =
            "logmel_" + stem + "_128x" +
            std::to_string(wav_features.n_frames) + ".f32";
        const std::vector<float> expected =
            read_float_binary(fixture_dir / expected_name);

        double mean_error = 0.0;
        double max_error = 0.0;
        REQUIRE(wav_features.data.size() == expected.size());
        for (std::size_t index = 0;
             index < expected.size();
             ++index) {
            const double error = std::fabs(
                static_cast<double>(wav_features.data[index]) -
                static_cast<double>(expected[index])
            );
            mean_error += error;
            max_error = std::max(max_error, error);
        }
        mean_error /= static_cast<double>(expected.size());

        if (decoded.sample_rate == 16000) {
            CHECK(max_error <= 5.0e-5);
        } else {
            CHECK(mean_error <= 2.0e-3);
            CHECK(max_error <= 0.5);
        }
    }
}

QWEN3ASR_TEST_CASE(gguf_tokenizer_and_prompt_match_pytorch_goldens) {
    qwen3_asr::GgufModelPair pair =
        qwen3_asr::load_qwen3_asr_gguf_pair(
            QWEN3ASR_ENCODER_GGUF,
            QWEN3ASR_LM_GGUF
        );
    qwen3_asr::TextTokenizer tokenizer(
        std::move(pair.lm.tokenizer_data)
    );
    const qwen3_asr::PromptBuilder prompt_builder(tokenizer);
    const std::filesystem::path fixture_dir =
        std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
        "qwen3_asr_0_6b";

    const nlohmann::json tokenizer_golden =
        read_json(fixture_dir / "tokenizer_golden.json");
    for (const auto & item : tokenizer_golden.at("cases")) {
        const std::vector<std::int32_t> expected =
            item.at("ids").get<std::vector<std::int32_t>>();
        CHECK(
            tokenizer.encode(item.at("text").get<std::string>()) ==
            expected
        );
    }

    const nlohmann::json prompt_golden =
        read_json(fixture_dir / "prompt_golden.json");
    for (const auto & item : prompt_golden.at("cases")) {
        std::optional<std::string> language;
        if (!item.at("language").is_null()) {
            language = item.at("language").get<std::string>();
        }
        const qwen3_asr::TranscriptionPrompt prompt =
            prompt_builder.build(
                item.at("context").get<std::string>(),
                language
            );
        CHECK(
            prompt.input_ids ==
            item.at("ids").get<std::vector<std::int32_t>>()
        );
    }
}

QWEN3ASR_TEST_CASE(gguf_model_rejects_missing_file) {
    bool threw = false;
    try {
        static_cast<void>(
            qwen3_asr::load_qwen3_asr_gguf_pair(
                std::filesystem::temp_directory_path() /
                    "qwen3_asr_missing_encoder.gguf",
                QWEN3ASR_LM_GGUF
            )
        );
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() == qwen3_asr::ErrorCode::ModelLoadFailed;
    }
    CHECK(threw);
}

QWEN3ASR_TEST_CASE(gguf_pair_rejects_wrong_component) {
    bool threw = false;
    try {
        static_cast<void>(
            qwen3_asr::load_qwen3_asr_gguf_pair(
                QWEN3ASR_LM_GGUF,
                QWEN3ASR_ENCODER_GGUF
            )
        );
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() == qwen3_asr::ErrorCode::ModelLoadFailed;
    }
    CHECK(threw);
}

}  // namespace
