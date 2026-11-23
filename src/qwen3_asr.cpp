#include "qwen3_asr.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <utility>

#include "errors.h"
#include "prompt_builder.h"

namespace qwen3_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::int64_t kMaxAudioSeconds = 1200;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start
    ).count();
}

std::string trim_copy(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

std::string lower_ascii(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        }
    );
    return value;
}

std::string normalize_language(std::string value) {
    value = trim_copy(value);
    if (value.empty()) {
        return {};
    }
    value[0] = static_cast<char>(
        std::toupper(static_cast<unsigned char>(value[0]))
    );
    for (std::size_t index = 1; index < value.size(); ++index) {
        value[index] = static_cast<char>(
            std::tolower(static_cast<unsigned char>(value[index]))
        );
    }
    return value;
}

}  // namespace

Qwen3AsrResult Qwen3Asr::parse_output(
    std::string_view raw,
    const std::optional<std::string> & forced_language
) {
    Qwen3AsrResult result;
    const std::string trimmed = trim_copy(raw);
    if (trimmed.empty()) {
        return result;
    }
    if (forced_language.has_value() && !forced_language->empty()) {
        result.language = normalize_language(*forced_language);
        result.text = trimmed;
        return result;
    }

    constexpr std::string_view kAsrTextTag = "<asr_text>";
    const std::size_t tag = trimmed.find(kAsrTextTag);
    if (tag == std::string::npos) {
        result.text = trimmed;
        return result;
    }

    const std::string metadata = trimmed.substr(0, tag);
    result.text = trim_copy(
        std::string_view(trimmed).substr(tag + kAsrTextTag.size())
    );
    if (lower_ascii(metadata).find("language none") != std::string::npos) {
        return result;
    }

    constexpr std::string_view kLanguagePrefix = "language ";
    std::size_t line_start = 0;
    while (line_start < metadata.size()) {
        std::size_t line_end = metadata.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = metadata.size();
        }
        const std::string line = trim_copy(
            std::string_view(metadata).substr(
                line_start,
                line_end - line_start
            )
        );
        const std::string lower_line = lower_ascii(line);
        if (lower_line.compare(
                0,
                kLanguagePrefix.size(),
                kLanguagePrefix
            ) == 0) {
            result.language = normalize_language(
                line.substr(kLanguagePrefix.size())
            );
            break;
        }
        line_start = line_end + 1;
    }
    return result;
}

Qwen3Asr::Qwen3Asr() = default;

Qwen3Asr::Qwen3Asr(const Qwen3AsrConfig & config)
    : ggml_session_(config.session) {
    const Clock::time_point start = Clock::now();
    load_model(config);
    model_load_ms_ = elapsed_ms(start);
}

void Qwen3Asr::load(const Qwen3AsrConfig & config) {
    unload();
    const Clock::time_point start = Clock::now();
    ggml_session_.initialize(config.session);
    load_model(config);
    model_load_ms_ = elapsed_ms(start);
}

void Qwen3Asr::load_model(const Qwen3AsrConfig & config) {
    GgufModelPair loaded_models = load_qwen3_asr_gguf_pair(
        config.encoder_gguf,
        config.lm_gguf
    );
    if (config.text_num_ctx.has_value()) {
        const int num_ctx = *config.text_num_ctx;
        if (num_ctx <= 0 ||
            num_ctx > loaded_models.lm.text.max_position_embeddings) {
            throw Error(
                ErrorCode::InvalidArgument,
                "Qwen3-ASR text num_ctx is out of range"
            );
        }
        loaded_models.lm.text.num_ctx = num_ctx;
    }
    aut_encoder_.load(
        loaded_models.encoder.audio,
        loaded_models.encoder,
        ggml_session_
    );
    qwen3_lm_.load(
        loaded_models.lm.text,
        loaded_models.lm,
        ggml_session_
    );
    _encoder_gguf = std::move(loaded_models.encoder);
    lm_gguf_ = std::move(loaded_models.lm);
    text_tokenizer_.emplace(lm_gguf_.tokenizer_data);
    audio_frontend_.emplace(
        _encoder_gguf.frontend_config,
        _encoder_gguf.mel_filters
    );
    loaded_ = true;
}

void Qwen3Asr::unload() {
    aut_encoder_.unload();
    qwen3_lm_.unload();
    _encoder_gguf = GgufModel{};
    lm_gguf_ = GgufModel{};
    text_tokenizer_.reset();
    audio_frontend_.reset();
    model_load_ms_ = 0.0;
    loaded_ = false;
}

void Qwen3Asr::reset() {
    unload();
    ggml_session_.reset();
}

Qwen3AsrResult Qwen3Asr::generate(
    const std::filesystem::path & audio_path,
    const Qwen3AsrGenerateOptions & options
) {
    if (!loaded_) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3-ASR model is not loaded"
        );
    }
    const Clock::time_point load_start = Clock::now();
    const AudioBuffer decoded = AudioLoader::load(audio_path);
    MonoAudio mono = AudioProcessor::prepare_mono(decoded);
    const double audio_load_ms = elapsed_ms(load_start);
    Qwen3AsrResult result = generate(std::move(mono), options);
    result.performance.audio_load_ms = audio_load_ms;
    result.performance.total_ms += audio_load_ms;
    result.performance.audio_seconds =
        decoded.sample_rate == 0
            ? 0.0
            : static_cast<double>(decoded.frame_count()) /
                  static_cast<double>(decoded.sample_rate);
    return result;
}

Qwen3AsrResult Qwen3Asr::generate(
    MonoAudio audio,
    const Qwen3AsrGenerateOptions & options
) {
    if (!loaded_ ||
        !text_tokenizer_.has_value() ||
        !audio_frontend_.has_value()) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3-ASR model is not loaded"
        );
    }
    const Clock::time_point total_start = Clock::now();
    if (audio.empty() || audio.sample_rate == 0) {
        throw Error(
            ErrorCode::InvalidAudioData,
            "Qwen3-ASR input audio is empty or has no sample rate"
        );
    }
    const std::int64_t input_samples =
        static_cast<std::int64_t>(audio.samples.size());
    const std::uint32_t input_sample_rate = audio.sample_rate;
    if (input_samples >
        static_cast<std::int64_t>(input_sample_rate) * kMaxAudioSeconds) {
        throw Error(
            ErrorCode::AudioTooLong,
            "Qwen3-ASR input exceeds the 20 minute limit"
        );
    }
    const double audio_seconds =
        static_cast<double>(input_samples) /
        static_cast<double>(input_sample_rate);

    double resample_ms = 0.0;
    if (audio.sample_rate != kQwen3AsrSampleRate) {
        const Clock::time_point resample_start = Clock::now();
        audio = AudioResampler::resample(audio, kQwen3AsrSampleRate);
        resample_ms = elapsed_ms(resample_start);
    }

    const int hop_length = audio_frontend_->config().hop_length;
    const std::int64_t feature_length =
        static_cast<std::int64_t>(audio.samples.size()) / hop_length;
    const AutEncoderShape encoder_shape =
        aut_encoder_.output_shape(feature_length);

    PromptBuilder prompt_builder(*text_tokenizer_);
    const TranscriptionPrompt prompt = prompt_builder.build(
        options.context,
        options.language
    );
    const ExpandedTranscriptionPrompt expanded =
        prompt_builder.expand_audio_tokens(
            prompt,
            static_cast<std::size_t>(encoder_shape.valid_frames)
        );
    if (options.generation.max_new_tokens < 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "Qwen3-ASR max_new_tokens must be non-negative"
        );
    }
    const std::uint64_t required_tokens =
        static_cast<std::uint64_t>(expanded.input_ids.size()) +
        static_cast<std::uint64_t>(options.generation.max_new_tokens);
    const std::uint64_t context_capacity = static_cast<std::uint64_t>(
        std::min(
            qwen3_lm_.config().num_ctx,
            qwen3_lm_.config().max_position_embeddings
        )
    );
    if (required_tokens > context_capacity) {
        throw Error(
            ErrorCode::ContextLengthExceeded,
            "Qwen3-ASR prompt requires " +
                std::to_string(required_tokens) +
                " tokens but context capacity is " +
                std::to_string(context_capacity)
        );
    }

    const Clock::time_point frontend_start = Clock::now();
    const LogMelFeatures features = audio_frontend_->extract(audio);
    const double frontend_ms = elapsed_ms(frontend_start);

    const Clock::time_point encoder_start = Clock::now();
    ggml_tensor * audio_embeddings = aut_encoder_.forward(features);
    const double encoder_ms = elapsed_ms(encoder_start);
    if (audio_embeddings == nullptr ||
        audio_embeddings->ne[0] != qwen3_lm_.config().hidden_size ||
        audio_embeddings->ne[1] != encoder_shape.valid_frames) {
        throw Error(
            ErrorCode::InferenceFailed,
            "Qwen3-ASR audio encoder returned invalid embeddings"
        );
    }
    const std::size_t audio_token_count = static_cast<std::size_t>(
        audio_embeddings->ne[1]
    );

    Qwen3LmGenerationStats lm_stats;
    const Clock::time_point generation_start = Clock::now();
    const std::vector<std::int32_t> generated = qwen3_lm_.generate(
        expanded.input_ids,
        audio_embeddings,
        expanded.first_audio_token_index,
        expanded.audio_token_count,
        options.generation,
        &lm_stats
    );
    const double generation_ms = elapsed_ms(generation_start);
    const std::string raw_text = text_tokenizer_->decode(generated, true);

    Qwen3AsrResult result = parse_output(
        raw_text,
        options.language
    );
    result.token_ids = generated;
    result.audio_token_count = static_cast<std::int64_t>(audio_token_count);
    result.lm_generation = lm_stats;
    result.performance.model_load_ms = model_load_ms_;
    result.performance.resample_ms = resample_ms;
    result.performance.frontend_ms = frontend_ms;
    result.performance.encoder_ms = encoder_ms;
    result.performance.generation_ms = generation_ms;
    result.performance.total_ms = elapsed_ms(total_start);
    result.performance.audio_seconds = audio_seconds;
    return result;
}

}  // namespace qwen3_asr
