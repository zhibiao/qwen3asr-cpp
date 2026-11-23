#include "test_framework.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#include "errors.h"
#include "qwen3_asr.h"

namespace {

qwen3_asr::Qwen3AsrConfig make_config() {
    qwen3_asr::Qwen3AsrConfig config;
    config.encoder_gguf = QWEN3ASR_ENCODER_GGUF;
    config.lm_gguf = QWEN3ASR_LM_GGUF;
    return config;
}

QWEN3ASR_TEST_CASE(qwen3_asr_loads_split_model_and_backend) {
    qwen3_asr::Qwen3Asr asr(make_config());

    CHECK(asr.loaded());
    CHECK(asr.ggml_session().initialized());
    CHECK(!asr.ggml_session().backend_name().empty());
    CHECK(
        asr.encoder_gguf().path.filename() ==
        "qwen3-asr-0.6b-encoder-bf16.gguf"
    );
    CHECK(
        asr.lm_gguf().path.filename() ==
        "qwen3-asr-0.6b-lm-bf16.gguf"
    );
    CHECK(asr.encoder_gguf().weight_tensor_count == 301);
    CHECK(asr.lm_gguf().weight_tensor_count == 311);
    CHECK(asr.encoder_gguf().audio.output_dim == 1024);
    CHECK(asr.lm_gguf().text.hidden_size == 1024);
}

QWEN3ASR_TEST_CASE(qwen3_asr_unload_and_reset) {
    qwen3_asr::Qwen3Asr asr;
    CHECK(!asr.loaded());
    CHECK(asr.ggml_session().initialized());

    asr.load(make_config());
    CHECK(asr.loaded());

    asr.unload();
    CHECK(!asr.loaded());
    CHECK(asr.encoder_gguf().path.empty());
    CHECK(asr.lm_gguf().path.empty());
    CHECK(asr.ggml_session().initialized());

    asr.load(make_config());
    asr.reset();
    CHECK(!asr.loaded());
    CHECK(!asr.ggml_session().initialized());
}

QWEN3ASR_TEST_CASE(qwen3_asr_generate_ar1) {
    qwen3_asr::Qwen3Asr asr(make_config());
    qwen3_asr::Qwen3AsrGenerateOptions options;
    options.generation.max_new_tokens = 64;
    const qwen3_asr::Qwen3AsrResult result = asr.generate(
        std::filesystem::path(QWEN3ASR_TEST_WAV_DIR) / "ar1.wav",
        options
    );

    std::cerr << "ASR language=" << result.language
              << " tokens=" << result.token_ids.size()
              << " audio_tokens=" << result.audio_token_count
              << " text=" << result.text << '\n';
    CHECK(result.language == "Arabic");
    CHECK(!result.text.empty());
    CHECK(!result.token_ids.empty());
    CHECK(result.audio_token_count == 69);
    CHECK(result.performance.model_load_ms > 0.0);
    CHECK(result.performance.frontend_ms > 0.0);
    CHECK(result.performance.encoder_ms > 0.0);
    CHECK(result.performance.generation_ms > 0.0);
    CHECK(result.performance.total_ms > 0.0);
    CHECK(result.lm_generation.prefill_ms > 0.0);
    CHECK(result.lm_generation.decode_ms > 0.0);
    CHECK(result.performance.audio_seconds > 5.0);
}

QWEN3ASR_TEST_CASE(qwen3_asr_rejects_context_before_encoder) {
    qwen3_asr::Qwen3Asr asr(make_config());

    qwen3_asr::MonoAudio short_audio;
    short_audio.sample_rate = qwen3_asr::kQwen3AsrSampleRate;
    short_audio.samples.assign(16000, 0.0F);
    qwen3_asr::Qwen3AsrGenerateOptions options;
    options.generation.max_new_tokens = 5000;
    bool rejected_context = false;
    try {
        static_cast<void>(asr.generate(short_audio, options));
    } catch (const qwen3_asr::Error & error) {
        rejected_context =
            error.code() == qwen3_asr::ErrorCode::ContextLengthExceeded;
    }
    CHECK(rejected_context);
}

QWEN3ASR_TEST_CASE(qwen3_asr_rejects_audio_over_twenty_minutes) {
    qwen3_asr::Qwen3Asr asr(make_config());
    qwen3_asr::MonoAudio audio;
    audio.sample_rate = 1000;
    audio.samples.assign(1200001, 0.0F);

    bool rejected = false;
    try {
        static_cast<void>(asr.generate(audio));
    } catch (const qwen3_asr::Error & error) {
        rejected = error.code() == qwen3_asr::ErrorCode::AudioTooLong;
    }
    CHECK(rejected);
}

QWEN3ASR_TEST_CASE(qwen3_asr_honors_text_num_ctx_override) {
    qwen3_asr::Qwen3AsrConfig config = make_config();
    config.text_num_ctx = 128;
    qwen3_asr::Qwen3Asr asr(config);

    qwen3_asr::MonoAudio audio;
    audio.sample_rate = qwen3_asr::kQwen3AsrSampleRate;
    audio.samples.assign(16000, 0.0F);
    qwen3_asr::Qwen3AsrGenerateOptions options;
    options.generation.max_new_tokens = 200;

    bool rejected = false;
    try {
        static_cast<void>(asr.generate(audio, options));
    } catch (const qwen3_asr::Error & error) {
        rejected =
            error.code() == qwen3_asr::ErrorCode::ContextLengthExceeded;
    }
    CHECK(rejected);
}

}  // namespace
