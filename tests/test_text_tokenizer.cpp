#include "test_framework.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model_tokenizer_loader.h"
#include "text_tokenizer.h"

namespace {

using qwen3_asr_test::load_model_tokenizer;
using qwen3_asr_test::read_json;

std::int32_t token_id(
    const qwen3_asr::TextTokenizer & tokenizer,
    const char * token
) {
    const auto id = tokenizer.token_to_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(std::string("missing token: ") + token);
    }
    return *id;
}

template <typename Value>
void check_sequence(
    const std::vector<Value> & actual,
    const std::vector<Value> & expected,
    const std::string & context
) {
    if (actual == expected) {
        CHECK(true);
        return;
    }

    CHECK(false);
    const std::size_t count =
        std::min(actual.size(), expected.size());
    std::size_t mismatch = count;
    for (std::size_t index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            mismatch = index;
            break;
        }
    }
    std::cerr << context << ": sequence mismatch at " << mismatch
              << ", actual_size=" << actual.size()
              << ", expected_size=" << expected.size() << '\n';
}

QWEN3ASR_TEST_CASE(model_special_token_ids_are_loaded) {
    const qwen3_asr::TextTokenizer tokenizer = load_model_tokenizer();
    CHECK(token_id(tokenizer, "<|im_start|>") == 151644);
    CHECK(token_id(tokenizer, "<|im_end|>") == 151645);
    CHECK(token_id(tokenizer, "<|audio_start|>") == 151669);
    CHECK(token_id(tokenizer, "<|audio_pad|>") == 151676);
    CHECK(token_id(tokenizer, "<|audio_end|>") == 151670);
    CHECK(token_id(tokenizer, "<asr_text>") == 151704);
}

QWEN3ASR_TEST_CASE(tokenizer_matches_all_pytorch_goldens) {
    const qwen3_asr::TextTokenizer tokenizer = load_model_tokenizer();
    const nlohmann::json golden =
        read_json(
            std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
            "qwen3_asr_0_6b" /
            "tokenizer_golden.json"
        );

    int checked_cases = 0;
    for (const auto & item : golden.at("cases")) {
        const std::string name = item.at("name").get<std::string>();
        const std::string text = item.at("text").get<std::string>();

        const std::vector<std::string> expected_tokens =
            item.at("tokens").get<std::vector<std::string>>();
        const std::vector<std::int32_t> expected_ids =
            item.at("ids").get<std::vector<std::int32_t>>();
        const std::string expected_decoded =
            item.at("decoded").get<std::string>();
        const std::string expected_without_special =
            item.at("decoded_without_special").get<std::string>();

        const std::vector<std::string> actual_tokens =
            tokenizer.tokenize(text);
        const std::vector<std::int32_t> actual_ids =
            tokenizer.encode(text);
        const std::string actual_decoded =
            tokenizer.decode(actual_ids, false);
        const std::string actual_without_special =
            tokenizer.decode(actual_ids, true);

        check_sequence(actual_tokens, expected_tokens, name + " tokens");
        check_sequence(actual_ids, expected_ids, name + " ids");
        CHECK(actual_decoded == expected_decoded);
        CHECK(actual_without_special == expected_without_special);
        std::cout << "tokenizer " << name
                  << " tokens=" << actual_tokens.size() << '\n';
        ++checked_cases;
    }
    CHECK(
        checked_cases ==
        golden.at("case_count").get<int>()
    );
}

QWEN3ASR_TEST_CASE(tokenizer_special_tokens_round_trip) {
    const qwen3_asr::TextTokenizer tokenizer = load_model_tokenizer();
    const std::string text =
        "<|im_start|>user\n<|audio_start|><|audio_pad|>"
        "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    const std::vector<std::int32_t> ids = tokenizer.encode(text);
    CHECK(tokenizer.decode(ids, false) == text);
    CHECK(tokenizer.decode(ids, true) == "user\n\nassistant\n");
}

}  // namespace
