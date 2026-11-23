#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "text_tokenizer.h"

namespace qwen3_asr_test {

inline nlohmann::json read_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "failed to open JSON file: " + path.string()
        );
    }
    nlohmann::json value;
    input >> value;
    return value;
}

inline qwen3_asr::TextTokenizer load_model_tokenizer() {
    const std::filesystem::path model_dir = QWEN3ASR_MODEL_DIR;
    const nlohmann::json vocab =
        read_json(model_dir / "vocab.json");
    const nlohmann::json config =
        read_json(model_dir / "tokenizer_config.json");

    qwen3_asr::TextTokenizerData data;
    data.vocab.reserve(vocab.size());
    for (const auto & item : vocab.items()) {
        data.vocab.emplace(item.key(), item.value().get<std::int32_t>());
    }

    std::ifstream merges_file(model_dir / "merges.txt");
    if (!merges_file) {
        throw std::runtime_error("failed to open merges.txt");
    }
    std::string line;
    while (std::getline(merges_file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const std::size_t separator = line.find(' ');
        if (separator == std::string::npos) {
            throw std::runtime_error("invalid merges.txt line");
        }
        data.merges.emplace_back(
            line.substr(0, separator),
            line.substr(separator + 1)
        );
    }

    std::unordered_map<std::string, std::int32_t> added_ids;
    for (const auto & item :
         config.at("added_tokens_decoder").items()) {
        qwen3_asr::TextTokenizerAddedToken token;
        token.id = std::stoi(item.key());
        token.content = item.value().at("content").get<std::string>();
        token.special = item.value().at("special").get<bool>();
        added_ids.emplace(token.content, token.id);
        data.added_tokens.push_back(std::move(token));
    }

    data.eos_token_id =
        config.at("eos_token").is_null()
            ? -1
            : added_ids.at(
                  config.at("eos_token").get<std::string>()
              );
    data.pad_token_id =
        config.at("pad_token").is_null()
            ? -1
            : added_ids.at(
                  config.at("pad_token").get<std::string>()
              );
    data.im_start_token_id = added_ids.at("<|im_start|>");
    data.im_end_token_id = added_ids.at("<|im_end|>");
    data.audio_start_token_id = added_ids.at("<|audio_start|>");
    data.audio_pad_token_id = added_ids.at("<|audio_pad|>");
    data.audio_end_token_id = added_ids.at("<|audio_end|>");
    data.asr_text_token_id = added_ids.at("<asr_text>");

    return qwen3_asr::TextTokenizer(std::move(data));
}

}  // namespace qwen3_asr_test
