#include "prompt_builder.h"

#include <utility>

#include "errors.h"

namespace qwen3_asr {

PromptBuilder::PromptBuilder(const TextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

TranscriptionPrompt PromptBuilder::build(
    std::string_view context,
    const std::optional<std::string> & language
) const {
    const TextTokenizerData & tokenizer_data = tokenizer_.data();
    if (tokenizer_data.audio_start_token_id < 0 ||
        tokenizer_data.audio_pad_token_id < 0 ||
        tokenizer_data.audio_end_token_id < 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "audio prompt token IDs are missing from tokenizer data"
        );
    }

    std::string before_audio = "<|im_start|>system\n";
    before_audio.append(context);
    before_audio +=
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|>";

    std::string rendered = before_audio;
    rendered +=
        "<|audio_pad|>"
        "<|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n";

    if (language.has_value() && !language->empty()) {
        rendered += "language ";
        rendered += *language;
        rendered += "<asr_text>";
    }

    TranscriptionPrompt prompt;
    prompt.rendered_text = rendered;
    prompt.input_ids = tokenizer_.encode(prompt.rendered_text);

    const std::vector<std::int32_t> before_ids =
        tokenizer_.encode(before_audio);
    if (before_ids.empty() ||
        before_ids.back() != tokenizer_data.audio_start_token_id) {
        throw Error(
            ErrorCode::InvalidArgument,
            "failed to locate audio start token in prompt"
        );
    }

    prompt.audio_start_index = before_ids.size() - 1;
    prompt.audio_pad_index = before_ids.size();
    prompt.audio_end_index = before_ids.size() + 1;
    prompt.generation_start_index = prompt.input_ids.size();

    if (prompt.audio_end_index >= prompt.input_ids.size() ||
        prompt.input_ids[prompt.audio_start_index] !=
            tokenizer_data.audio_start_token_id ||
        prompt.input_ids[prompt.audio_pad_index] !=
            tokenizer_data.audio_pad_token_id ||
        prompt.input_ids[prompt.audio_end_index] !=
            tokenizer_data.audio_end_token_id) {
        throw Error(
            ErrorCode::InvalidArgument,
            "audio token span does not match tokenized prompt"
        );
    }

    return prompt;
}

ExpandedTranscriptionPrompt PromptBuilder::expand_audio_tokens(
    const TranscriptionPrompt & prompt,
    std::size_t audio_token_count
) const {
    if (audio_token_count == 0) {
        throw Error(
            ErrorCode::InvalidArgument,
            "audio token count must be greater than zero"
        );
    }
    if (prompt.audio_pad_index >= prompt.input_ids.size() ||
        prompt.input_ids[prompt.audio_pad_index] !=
            tokenizer_.data().audio_pad_token_id) {
        throw Error(
            ErrorCode::InvalidArgument,
            "prompt does not contain the expected audio pad token"
        );
    }

    ExpandedTranscriptionPrompt result;
    result.audio_start_index = prompt.audio_start_index;
    result.first_audio_token_index = prompt.audio_pad_index;
    result.audio_token_count = audio_token_count;
    result.audio_end_index =
        prompt.audio_pad_index + audio_token_count;

    const std::size_t suffix_count =
        prompt.input_ids.size() - prompt.audio_pad_index - 1;
    result.input_ids.reserve(
        prompt.input_ids.size() - 1 + audio_token_count
    );
    result.input_ids.insert(
        result.input_ids.end(),
        prompt.input_ids.begin(),
        prompt.input_ids.begin() +
            static_cast<std::ptrdiff_t>(prompt.audio_pad_index)
    );
    result.input_ids.insert(
        result.input_ids.end(),
        audio_token_count,
        tokenizer_.data().audio_pad_token_id
    );
    result.input_ids.insert(
        result.input_ids.end(),
        prompt.input_ids.begin() +
            static_cast<std::ptrdiff_t>(prompt.audio_pad_index + 1),
        prompt.input_ids.end()
    );
    result.generation_start_index = result.input_ids.size();

    if (suffix_count + prompt.audio_pad_index + 1 !=
            prompt.input_ids.size() ||
        result.audio_end_index >= result.input_ids.size() ||
        result.input_ids[result.audio_end_index] !=
            tokenizer_.data().audio_end_token_id) {
        throw Error(
            ErrorCode::InvalidArgument,
            "expanded audio token span is invalid"
        );
    }

    return result;
}

}  // namespace qwen3_asr
