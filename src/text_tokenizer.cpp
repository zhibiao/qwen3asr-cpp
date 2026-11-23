#include "text_tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <uni-algo-1.2.0/norm.h>
#include <uni-algo-1.2.0/prop.h>

#include "errors.h"

namespace qwen3_asr {
namespace {

struct Codepoint {
    char32_t value = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
};

std::vector<Codepoint> decode_utf8(std::string_view text) {
    std::vector<Codepoint> result;
    result.reserve(text.size());

    std::size_t index = 0;
    while (index < text.size()) {
        const std::uint8_t first =
            static_cast<std::uint8_t>(text[index]);
        std::uint32_t value = 0;
        int continuation_count = 0;

        if (first < 0x80) {
            value = first;
        } else if ((first & 0xE0U) == 0xC0U) {
            value = first & 0x1FU;
            continuation_count = 1;
        } else if ((first & 0xF0U) == 0xE0U) {
            value = first & 0x0FU;
            continuation_count = 2;
        } else if ((first & 0xF8U) == 0xF0U) {
            value = first & 0x07U;
            continuation_count = 3;
        } else {
            throw Error(
                ErrorCode::InvalidArgument,
                "tokenizer input contains invalid UTF-8"
            );
        }

        if (index + static_cast<std::size_t>(continuation_count) >=
            text.size()) {
            throw Error(
                ErrorCode::InvalidArgument,
                "tokenizer input contains truncated UTF-8"
            );
        }

        for (int offset = 1;
             offset <= continuation_count;
             ++offset) {
            const std::uint8_t continuation =
                static_cast<std::uint8_t>(text[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                throw Error(
                    ErrorCode::InvalidArgument,
                    "tokenizer input contains invalid UTF-8 continuation"
                );
            }
            value = (value << 6U) | (continuation & 0x3FU);
        }

        const std::size_t end =
            index + 1 + static_cast<std::size_t>(continuation_count);
        result.push_back(Codepoint{
            static_cast<char32_t>(value),
            index,
            end,
        });
        index = end;
    }

    return result;
}

void append_utf8(char32_t codepoint, std::string & output) {
    const std::uint32_t value = static_cast<std::uint32_t>(codepoint);
    if (value <= 0x7FU) {
        output.push_back(static_cast<char>(value));
    } else if (value <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (value >> 6U)));
        output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else if (value <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (value >> 12U)));
        output.push_back(static_cast<char>(
            0x80U | ((value >> 6U) & 0x3FU)
        ));
        output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xF0U | (value >> 18U)));
        output.push_back(static_cast<char>(
            0x80U | ((value >> 12U) & 0x3FU)
        ));
        output.push_back(static_cast<char>(
            0x80U | ((value >> 6U) & 0x3FU)
        ));
        output.push_back(static_cast<char>(0x80U | (value & 0x3FU)));
    }
}

struct ByteUnicodeMaps {
    std::array<std::string, 256> byte_to_unicode;
    std::unordered_map<std::string, std::uint8_t> unicode_to_byte;
};

const ByteUnicodeMaps & byte_unicode_maps() {
    static const ByteUnicodeMaps maps = []() {
        ByteUnicodeMaps value;
        std::array<bool, 256> direct{};
        for (int byte = static_cast<int>('!');
             byte <= static_cast<int>('~');
             ++byte) {
            direct[static_cast<std::size_t>(byte)] = true;
        }
        for (int byte = 161; byte <= 172; ++byte) {
            direct[static_cast<std::size_t>(byte)] = true;
        }
        for (int byte = 174; byte <= 255; ++byte) {
            direct[static_cast<std::size_t>(byte)] = true;
        }

        int next_codepoint = 256;
        for (int byte = 0; byte < 256; ++byte) {
            const int codepoint =
                direct[static_cast<std::size_t>(byte)]
                    ? byte
                    : next_codepoint++;
            std::string token;
            append_utf8(static_cast<char32_t>(codepoint), token);
            value.byte_to_unicode[static_cast<std::size_t>(byte)] = token;
            value.unicode_to_byte.emplace(
                token,
                static_cast<std::uint8_t>(byte)
            );
        }
        return value;
    }();
    return maps;
}

bool is_letter(char32_t codepoint) {
    return una::codepoint::prop{codepoint}.General_Category_L();
}

bool is_number(char32_t codepoint) {
    return una::codepoint::prop{codepoint}.General_Category_N();
}

bool is_whitespace(char32_t codepoint) {
    return una::codepoint::prop{codepoint}.White_Space();
}

bool is_crlf(char32_t codepoint) {
    return codepoint == U'\r' || codepoint == U'\n';
}

bool is_punctuation_prefix(char32_t codepoint) {
    return !is_whitespace(codepoint) &&
           !is_letter(codepoint) &&
           !is_number(codepoint);
}

char ascii_lower(char value) {
    return static_cast<char>(
        std::tolower(static_cast<unsigned char>(value))
    );
}

int contraction_length(
    const std::vector<Codepoint> & codepoints,
    std::size_t index
) {
    if (index >= codepoints.size() ||
        codepoints[index].value != U'\'') {
        return 0;
    }

    static constexpr std::array<std::string_view, 7> suffixes{
        "s", "t", "re", "ve", "m", "ll", "d",
    };
    for (std::string_view suffix : suffixes) {
        if (index + suffix.size() >= codepoints.size()) {
            continue;
        }
        bool matched = true;
        for (std::size_t offset = 0; offset < suffix.size(); ++offset) {
            const char32_t codepoint =
                codepoints[index + 1 + offset].value;
            if (codepoint > 0x7F ||
                ascii_lower(static_cast<char>(codepoint)) !=
                    ascii_lower(suffix[offset])) {
                matched = false;
                break;
            }
        }
        if (matched) {
            return static_cast<int>(suffix.size() + 1);
        }
    }
    return 0;
}

std::vector<std::string> pretokenize(std::string_view text) {
    const std::vector<Codepoint> codepoints = decode_utf8(text);
    std::vector<std::string> pieces;
    std::size_t index = 0;

    while (index < codepoints.size()) {
        const std::size_t start = index;
        bool matched = false;

        const int contraction = contraction_length(codepoints, index);
        if (contraction > 0) {
            index += static_cast<std::size_t>(contraction);
            matched = true;
        }

        if (!matched && is_letter(codepoints[index].value)) {
            ++index;
            while (index < codepoints.size() &&
                   is_letter(codepoints[index].value)) {
                ++index;
            }
            matched = true;
        }

        if (!matched &&
            !is_crlf(codepoints[index].value) &&
            !is_letter(codepoints[index].value) &&
            !is_number(codepoints[index].value) &&
            index + 1 < codepoints.size() &&
            is_letter(codepoints[index + 1].value)) {
            index += 2;
            while (index < codepoints.size() &&
                   is_letter(codepoints[index].value)) {
                ++index;
            }
            matched = true;
        }

        if (!matched && is_number(codepoints[index].value)) {
            ++index;
            matched = true;
        }

        if (!matched) {
            std::size_t punctuation_start = index;
            if (codepoints[index].value == U' ') {
                ++punctuation_start;
            }
            if (punctuation_start < codepoints.size() &&
                is_punctuation_prefix(
                    codepoints[punctuation_start].value
                )) {
                index = punctuation_start;
                while (index < codepoints.size() &&
                       is_punctuation_prefix(
                           codepoints[index].value
                       )) {
                    ++index;
                }
                while (index < codepoints.size() &&
                       is_crlf(codepoints[index].value)) {
                    ++index;
                }
                matched = true;
            }
        }

        if (!matched && is_whitespace(codepoints[index].value)) {
            const std::size_t whitespace_start = index;
            bool contains_newline = false;
            ++index;
            while (index < codepoints.size() &&
                   is_whitespace(codepoints[index].value)) {
                contains_newline =
                    contains_newline ||
                    is_crlf(codepoints[index].value);
                ++index;
            }
            if (!contains_newline &&
                index < codepoints.size() &&
                index - whitespace_start > 1) {
                --index;
            }
            matched = true;
        }

        if (!matched) {
            ++index;
        }

        pieces.emplace_back(
            text.substr(
                codepoints[start].begin,
                codepoints[index - 1].end -
                    codepoints[start].begin
            )
        );
    }

    return pieces;
}

std::vector<std::string> split_utf8_codepoints(std::string_view text) {
    const std::vector<Codepoint> codepoints = decode_utf8(text);
    std::vector<std::string> result;
    result.reserve(codepoints.size());
    for (const Codepoint & codepoint : codepoints) {
        result.emplace_back(
            text.substr(
                codepoint.begin,
                codepoint.end - codepoint.begin
            )
        );
    }
    return result;
}

std::string merge_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\x1F');
    key.append(right);
    return key;
}

std::string byte_encode(std::string_view text) {
    const ByteUnicodeMaps & maps = byte_unicode_maps();
    std::string encoded;
    encoded.reserve(text.size() * 2);
    for (const unsigned char byte : text) {
        encoded += maps.byte_to_unicode[byte];
    }
    return encoded;
}

}  // namespace

TextTokenizer::TextTokenizer(TextTokenizerData data)
    : data_(std::move(data)),
      added_by_length_(data_.added_tokens) {
    std::sort(
        added_by_length_.begin(),
        added_by_length_.end(),
        [](const TextTokenizerAddedToken & lhs,
           const TextTokenizerAddedToken & rhs) {
            return lhs.content.size() > rhs.content.size();
        }
    );

    std::int32_t maximum_id = -1;
    for (const auto & entry : data_.vocab) {
        maximum_id = std::max(maximum_id, entry.second);
    }
    for (const TextTokenizerAddedToken & token : data_.added_tokens) {
        maximum_id = std::max(maximum_id, token.id);
    }
    if (maximum_id < 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "tokenizer vocab is empty"
        );
    }
    id_to_token_.resize(static_cast<std::size_t>(maximum_id) + 1);

    for (const auto & entry : data_.vocab) {
        if (entry.second >= 0) {
            id_to_token_[static_cast<std::size_t>(entry.second)] =
                entry.first;
        }
    }
    for (const TextTokenizerAddedToken & token : data_.added_tokens) {
        if (token.id < 0) {
            throw Error(
                ErrorCode::InvalidArgument,
                "added token has a negative ID"
            );
        }
        data_.vocab.emplace(token.content, token.id);
        id_to_token_[static_cast<std::size_t>(token.id)] = token.content;
        added_by_id_.emplace(token.id, token);
    }

    std::int32_t rank = 0;
    for (const auto & merge : data_.merges) {
        merge_ranks_.emplace(
            merge_key(merge.first, merge.second),
            rank++
        );
    }
}

std::vector<std::string> TextTokenizer::tokenize(
    std::string_view text
) const {
    std::vector<std::string> output;
    std::size_t ordinary_start = 0;
    std::size_t position = 0;

    auto encode_ordinary = [&](std::string_view ordinary) {
        if (ordinary.empty()) {
            return;
        }
        const std::string normalized =
            una::norm::to_nfc_utf8(ordinary);
        for (const std::string & piece : pretokenize(normalized)) {
            const std::string encoded = byte_encode(piece);
            std::vector<std::string> symbols =
                split_utf8_codepoints(encoded);

            while (symbols.size() > 1) {
                std::int32_t best_rank =
                    std::numeric_limits<std::int32_t>::max();
                std::size_t best_index = symbols.size();
                for (std::size_t index = 0;
                     index + 1 < symbols.size();
                     ++index) {
                    const auto found = merge_ranks_.find(
                        merge_key(symbols[index], symbols[index + 1])
                    );
                    if (found != merge_ranks_.end() &&
                        found->second < best_rank) {
                        best_rank = found->second;
                        best_index = index;
                    }
                }
                if (best_index == symbols.size()) {
                    break;
                }

                const std::string left = symbols[best_index];
                const std::string right = symbols[best_index + 1];
                std::vector<std::string> merged;
                merged.reserve(symbols.size());
                for (std::size_t index = 0;
                     index < symbols.size();) {
                    if (index + 1 < symbols.size() &&
                        symbols[index] == left &&
                        symbols[index + 1] == right) {
                        merged.push_back(left + right);
                        index += 2;
                    } else {
                        merged.push_back(std::move(symbols[index]));
                        ++index;
                    }
                }
                symbols = std::move(merged);
            }

            for (std::string & symbol : symbols) {
                output.push_back(std::move(symbol));
            }
        }
    };

    while (position < text.size()) {
        const TextTokenizerAddedToken * matched_token = nullptr;
        for (const TextTokenizerAddedToken & token : added_by_length_) {
            if (position + token.content.size() <= text.size() &&
                text.substr(position, token.content.size()) ==
                    token.content) {
                matched_token = &token;
                break;
            }
        }

        if (matched_token == nullptr) {
            ++position;
            continue;
        }

        encode_ordinary(text.substr(
            ordinary_start,
            position - ordinary_start
        ));
        output.push_back(matched_token->content);
        position += matched_token->content.size();
        ordinary_start = position;
    }
    encode_ordinary(text.substr(ordinary_start));
    return output;
}

std::vector<std::int32_t> TextTokenizer::encode(
    std::string_view text
) const {
    const std::vector<std::string> tokens = tokenize(text);
    std::vector<std::int32_t> ids;
    ids.reserve(tokens.size());
    for (const std::string & token : tokens) {
        const auto found = data_.vocab.find(token);
        if (found == data_.vocab.end()) {
            throw Error(
                ErrorCode::InvalidArgument,
                "BPE token is missing from vocabulary: " + token
            );
        }
        ids.push_back(found->second);
    }
    return ids;
}

std::string TextTokenizer::decode(
    const std::vector<std::int32_t> & token_ids,
    bool skip_special_tokens
) const {
    const ByteUnicodeMaps & maps = byte_unicode_maps();
    std::string output;
    std::string pending_bytes;

    auto flush_pending = [&]() {
        output += pending_bytes;
        pending_bytes.clear();
    };

    for (std::int32_t id : token_ids) {
        const auto added = added_by_id_.find(id);
        if (added != added_by_id_.end()) {
            flush_pending();
            if (!(skip_special_tokens && added->second.special)) {
                output += added->second.content;
            }
            continue;
        }

        if (id < 0 ||
            static_cast<std::size_t>(id) >= id_to_token_.size() ||
            id_to_token_[static_cast<std::size_t>(id)].empty()) {
            throw Error(
                ErrorCode::InvalidArgument,
                "token ID is missing from decoder: " +
                    std::to_string(id)
            );
        }

        for (const std::string & codepoint :
             split_utf8_codepoints(
                 id_to_token_[static_cast<std::size_t>(id)]
             )) {
            const auto byte = maps.unicode_to_byte.find(codepoint);
            if (byte == maps.unicode_to_byte.end()) {
                throw Error(
                    ErrorCode::InvalidArgument,
                    "token contains a non-byte-level codepoint"
                );
            }
            pending_bytes.push_back(static_cast<char>(byte->second));
        }
    }
    flush_pending();
    return output;
}

std::optional<std::int32_t> TextTokenizer::token_to_id(
    std::string_view token
) const {
    const auto found = data_.vocab.find(std::string(token));
    if (found == data_.vocab.end()) {
        return std::nullopt;
    }
    return found->second;
}

}  // namespace qwen3_asr
