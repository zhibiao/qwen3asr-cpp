#include "test_framework.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "errors.h"
#include "model_tokenizer_loader.h"
#include "prompt_builder.h"

namespace {

using qwen3_asr_test::load_model_tokenizer;
using qwen3_asr_test::read_json;

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
    std::cerr << context << ": sequence mismatch, actual_size="
              << actual.size() << ", expected_size="
              << expected.size() << '\n';
}

QWEN3ASR_TEST_CASE(prompt_matches_all_pytorch_goldens) {
    const qwen3_asr::TextTokenizer tokenizer = load_model_tokenizer();
    const qwen3_asr::PromptBuilder builder(tokenizer);
    const nlohmann::json golden =
        read_json(
            std::filesystem::path(QWEN3ASR_TEST_DATA_DIR) /
            "qwen3_asr_0_6b" /
            "prompt_golden.json"
        );

    int checked_cases = 0;
    for (const auto & item : golden.at("cases")) {
        const std::string name = item.at("name").get<std::string>();
        const std::string context =
            item.at("context").get<std::string>();
        std::optional<std::string> language;
        if (!item.at("language").is_null()) {
            language = item.at("language").get<std::string>();
        }

        const std::vector<std::int32_t> expected_ids =
            item.at("ids").get<std::vector<std::int32_t>>();
        const qwen3_asr::TranscriptionPrompt prompt =
            builder.build(context, language);

        CHECK(prompt.rendered_text == item.at("rendered").get<std::string>());
        check_sequence(prompt.input_ids, expected_ids, name + " ids");
        CHECK(
            prompt.audio_start_index ==
            item.at("audio_start_index").get<std::size_t>()
        );
        CHECK(
            prompt.audio_pad_index ==
            item.at("audio_pad_index").get<std::size_t>()
        );
        CHECK(
            prompt.audio_end_index ==
            item.at("audio_end_index").get<std::size_t>()
        );
        CHECK(
            prompt.generation_start_index ==
            item.at("generation_start_index").get<std::size_t>()
        );
        CHECK(tokenizer.decode(prompt.input_ids, false) == prompt.rendered_text);
        std::cout << "prompt " << name
                  << " tokens=" << prompt.input_ids.size()
                  << " pad_index=" << prompt.audio_pad_index << '\n';
        ++checked_cases;
    }
    CHECK(checked_cases == golden.at("case_count").get<int>());
}

QWEN3ASR_TEST_CASE(prompt_expands_audio_placeholder_span) {
    const qwen3_asr::TextTokenizer tokenizer = load_model_tokenizer();
    const qwen3_asr::PromptBuilder builder(tokenizer);
    const qwen3_asr::TranscriptionPrompt prompt =
        builder.build("测试 context", std::string("Chinese"));

    constexpr std::size_t audio_token_count = 390;
    const qwen3_asr::ExpandedTranscriptionPrompt expanded =
        builder.expand_audio_tokens(prompt, audio_token_count);

    CHECK(
        expanded.input_ids.size() ==
        prompt.input_ids.size() - 1 + audio_token_count
    );
    CHECK(expanded.first_audio_token_index == prompt.audio_pad_index);
    CHECK(expanded.audio_token_count == audio_token_count);
    CHECK(expanded.audio_end_index ==
          expanded.first_audio_token_index + audio_token_count);
    CHECK(expanded.input_ids[expanded.audio_start_index] ==
          tokenizer.data().audio_start_token_id);
    CHECK(expanded.input_ids[expanded.audio_end_index] ==
          tokenizer.data().audio_end_token_id);
    CHECK(expanded.generation_start_index == expanded.input_ids.size());

    std::size_t pad_count = 0;
    for (std::int32_t id : expanded.input_ids) {
        if (id == tokenizer.data().audio_pad_token_id) {
            ++pad_count;
        }
    }
    CHECK(pad_count == audio_token_count);

    std::string expected_text = prompt.rendered_text;
    const std::string placeholder = "<|audio_pad|>";
    const std::size_t position =
        expected_text.find(placeholder);
    REQUIRE(position != std::string::npos);
    std::string repeated;
    for (std::size_t index = 0; index < audio_token_count; ++index) {
        repeated += placeholder;
    }
    expected_text.replace(position, placeholder.size(), repeated);
    CHECK(
        tokenizer.decode(expanded.input_ids, false) ==
        expected_text
    );
}

QWEN3ASR_TEST_CASE(prompt_rejects_zero_audio_tokens) {
    const qwen3_asr::TextTokenizer tokenizer = load_model_tokenizer();
    const qwen3_asr::PromptBuilder builder(tokenizer);
    const qwen3_asr::TranscriptionPrompt prompt = builder.build("");

    bool threw = false;
    try {
        static_cast<void>(
            builder.expand_audio_tokens(prompt, 0)
        );
    } catch (const qwen3_asr::Error & error) {
        threw = error.code() == qwen3_asr::ErrorCode::InvalidArgument;
    }
    CHECK(threw);
}

}  // namespace
