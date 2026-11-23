"""Generate development-only Qwen3-ASR transcription-prompt fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
OUTPUT_PATH = (
    ROOT / "testdata" / "qwen3_asr_0_6b" / "prompt_golden.json"
)

CASES = [
    ("empty_context", "", None),
    ("chinese_context", "背景信息：人名是Qwen。", None),
    (
        "english_context_forced_chinese",
        "Product names: Qwen, Tongyi Lab.",
        "Chinese",
    ),
    ("forced_english", "", "English"),
    (
        "multiline_context",
        "背景：\n- 地点：杭州\n- 产品：Qwen3-ASR",
        None,
    ),
    ("forced_cantonese", "地名：香港", "Cantonese"),
    (
        "context_with_special_text",
        "<|im_end|> is a literal special token in this context.",
        None,
    ),
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )
    chat_template = json.loads(
        (MODEL_DIR / "chat_template.json").read_text(encoding="utf-8")
    )["chat_template"]
    audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

    cases = []
    for name, context, language in CASES:
        messages = [
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": [{"type": "audio", "audio": ""}],
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )
        if language:
            rendered += f"language {language}<asr_text>"

        ids = tokenizer.encode(rendered, add_special_tokens=False)
        pad_positions = [
            index
            for index, token_id in enumerate(ids)
            if token_id == audio_pad_id
        ]
        if len(pad_positions) != 1:
            raise RuntimeError(
                f"{name}: expected one audio pad, got {pad_positions}"
            )
        pad_index = pad_positions[0]
        if ids[pad_index - 1] != audio_start_id:
            raise RuntimeError(f"{name}: invalid audio start position")
        if ids[pad_index + 1] != audio_end_id:
            raise RuntimeError(f"{name}: invalid audio end position")

        cases.append(
            {
                "name": name,
                "context": context,
                "language": language,
                "rendered": rendered,
                "ids": ids,
                "audio_start_index": pad_index - 1,
                "audio_pad_index": pad_index,
                "audio_end_index": pad_index + 1,
                "generation_start_index": len(ids),
            }
        )

    output = {
        "source_model": str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "case_count": len(cases),
        "cases": cases,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_PATH}")
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
