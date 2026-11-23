#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "text_tokenizer.h"

namespace qwen3_asr {

struct TranscriptionPrompt {
    std::vector<std::int32_t> input_ids;
    std::size_t audio_start_index = 0;
    std::size_t audio_pad_index = 0;
    std::size_t audio_end_index = 0;
    std::size_t generation_start_index = 0;
    std::string rendered_text;
};

struct ExpandedTranscriptionPrompt {
    std::vector<std::int32_t> input_ids;
    std::size_t audio_start_index = 0;
    std::size_t first_audio_token_index = 0;
    std::size_t audio_token_count = 0;
    std::size_t audio_end_index = 0;
    std::size_t generation_start_index = 0;
};

class PromptBuilder {
public:
    explicit PromptBuilder(const TextTokenizer & tokenizer);

    TranscriptionPrompt build(
        std::string_view context,
        const std::optional<std::string> & language = std::nullopt
    ) const;

    ExpandedTranscriptionPrompt expand_audio_tokens(
        const TranscriptionPrompt & prompt,
        std::size_t audio_token_count
    ) const;

private:
    const TextTokenizer & tokenizer_;
};

}  // namespace qwen3_asr
