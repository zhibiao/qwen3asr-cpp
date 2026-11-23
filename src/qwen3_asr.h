#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "audio_frontend.h"
#include "aut_encoder.h"
#include "ggml_session.h"
#include "gguf_model.h"
#include "qwen3_lm.h"
#include "text_tokenizer.h"

namespace qwen3_asr {

struct Qwen3AsrGenerateOptions {
    // Text placed in the system prompt to provide names, terminology, or
    // background information for recognition. Empty disables this context.
    std::string context;
    std::optional<std::string> language;
    Qwen3LmGenerationConfig generation;
};

struct Qwen3AsrPerformanceStats {
    double model_load_ms = 0.0;
    double audio_load_ms = 0.0;
    double resample_ms = 0.0;
    double frontend_ms = 0.0;
    double encoder_ms = 0.0;
    double generation_ms = 0.0;
    double total_ms = 0.0;
    double audio_seconds = 0.0;
};

struct Qwen3AsrResult {
    std::string language;
    std::string text;
    std::vector<std::int32_t> token_ids;
    std::int64_t audio_token_count = 0;
    Qwen3AsrPerformanceStats performance;
    Qwen3LmGenerationStats lm_generation;
};

class Qwen3Asr {
public:
    Qwen3Asr();
    explicit Qwen3Asr(const Qwen3AsrConfig & config);

    void load(const Qwen3AsrConfig & config);

    void unload();
    void reset();

    Qwen3AsrResult generate(
        const std::filesystem::path & audio_path,
        const Qwen3AsrGenerateOptions & options = {}
    );

    Qwen3AsrResult generate(
        MonoAudio audio,
        const Qwen3AsrGenerateOptions & options = {}
    );

    bool loaded() const noexcept {
        return loaded_;
    }

    double model_load_ms() const noexcept {
        return model_load_ms_;
    }

    const GgufModel & encoder_gguf() const noexcept {
        return _encoder_gguf;
    }

    const GgufModel & lm_gguf() const noexcept {
        return lm_gguf_;
    }

    const GgmlSession & ggml_session() const noexcept {
        return ggml_session_;
    }

private:
    static Qwen3AsrResult parse_output(
        std::string_view raw,
        const std::optional<std::string> & forced_language
    );

    void load_model(const Qwen3AsrConfig & config);

    GgmlSession ggml_session_;
    AutEncoder aut_encoder_;
    Qwen3Lm qwen3_lm_;
    GgufModel _encoder_gguf;
    GgufModel lm_gguf_;
    std::optional<TextTokenizer> text_tokenizer_;
    std::optional<AudioFrontend> audio_frontend_;
    double model_load_ms_ = 0.0;
    bool loaded_ = false;
};

}  // namespace qwen3_asr
