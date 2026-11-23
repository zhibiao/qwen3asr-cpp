"""Generate development-only Qwen2 tokenizer golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
OUTPUT_PATH = (
    ROOT / "testdata" / "qwen3_asr_0_6b" / "tokenizer_golden.json"
)

CASES = [
    ("empty", ""),
    ("english", "Hello, world!"),
    ("contractions", "It's Qwen's tokenizer, we're testing it."),
    ("chinese", "今天天气很好。"),
    ("mixed", "混合English和中文123"),
    ("numbers", "0 1 23 456789"),
    ("accented", "café naïve Ångström"),
    ("nfc", "e\u0301 中文 😀"),
    ("emoji", "😀🚀语音识别"),
    ("whitespace", "  multiple spaces\tand tabs  "),
    ("newlines", "Hi\n\nThere\r\nQwen"),
    (
        "transcription_prompt",
        "<|im_start|>system\nYou are Qwen.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>"
        "<|im_end|>\n<|im_start|>assistant\n",
    ),
    ("asr_metadata", "<asr_text>Hello"),
    (
        "forced_language_prompt",
        "<|im_start|>assistant\n"
        "language Chinese<asr_text>",
    ),
]

TRANSCRIPT_PATH = ROOT / "test_wavs" / "transcript.txt"
for line in TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines():
    filename, separator, transcript = line.partition(" ")
    if separator and transcript:
        CASES.append(
            (
                f"transcript_{Path(filename).stem}",
                transcript,
            )
        )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    cases = []
    for name, text in CASES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        cases.append(
            {
                "name": name,
                "text": text,
                "tokens": tokens,
                "ids": ids,
                "decoded": tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                ),
                "decoded_without_special": tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                ),
            }
        )

    output = {
        "source_model": str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "transformers_version": transformers.__version__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "case_count": len(cases),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "special_ids": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in [
                "<|im_start|>",
                "<|im_end|>",
                "<|audio_start|>",
                "<|audio_pad|>",
                "<|audio_end|>",
                "<asr_text>",
            ]
        },
        "cases": cases,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH}")
    print(f"cases={len(cases)} vocab={len(tokenizer)}")


if __name__ == "__main__":
    main()
