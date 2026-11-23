#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qwen3_asr {

struct TextTokenizerAddedToken {
    std::string content;
    std::int32_t id = -1;
    bool special = false;
};

struct TextTokenizerData {
    std::unordered_map<std::string, std::int32_t> vocab;
    std::vector<std::pair<std::string, std::string>> merges;
    std::vector<TextTokenizerAddedToken> added_tokens;

    std::int32_t eos_token_id = -1;
    std::int32_t pad_token_id = -1;
    std::int32_t im_start_token_id = -1;
    std::int32_t im_end_token_id = -1;
    std::int32_t audio_start_token_id = -1;
    std::int32_t audio_pad_token_id = -1;
    std::int32_t audio_end_token_id = -1;
    std::int32_t asr_text_token_id = -1;
};

class TextTokenizer {
public:
    explicit TextTokenizer(TextTokenizerData data);

    std::vector<std::int32_t> encode(std::string_view text) const;
    std::vector<std::string> tokenize(std::string_view text) const;
    std::string decode(
        const std::vector<std::int32_t> & token_ids,
        bool skip_special_tokens = false
    ) const;

    std::optional<std::int32_t> token_to_id(
        std::string_view token
    ) const;
    const TextTokenizerData & data() const noexcept {
        return data_;
    }

private:
    TextTokenizerData data_;
    std::vector<TextTokenizerAddedToken> added_by_length_;
    std::unordered_map<std::string, std::int32_t> merge_ranks_;
    std::unordered_map<std::int32_t, TextTokenizerAddedToken> added_by_id_;
    std::vector<std::string> id_to_token_;
};

}  // namespace qwen3_asr
