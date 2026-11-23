"""Generate development-only Qwen3 ASR text LM golden fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
OUTPUT_DIR = ROOT / "testdata" / "qwen3_asr_0_6b"
PREFIX = "thinker.model."

TEXT_IDS = [
    151644,
    8948,
    198,
    151645,
    198,
    151644,
    872,
    198,
    151669,
    151676,
    151670,
    151645,
    198,
    151644,
    77091,
    198,
]
AUDIO_IDS = (
    TEXT_IDS[:9] + [151676] * 69 + TEXT_IDS[10:]
)
AUDIO_START = 9
AUDIO_COUNT = 69
DECODE_TOKEN = 9707
LONG_TEXT_IDS = (TEXT_IDS * 12)[:192]
LONG_AUDIO_COUNT = 270
LONG_AUDIO_IDS = (
    TEXT_IDS[:AUDIO_START] +
    [151676] * LONG_AUDIO_COUNT +
    TEXT_IDS[AUDIO_START + 1:]
)


def load_weights() -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with safe_open(
        str(MODEL_DIR / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as source:
        for name in source.keys():
            if name.startswith(PREFIX) or name == "thinker.lm_head.weight":
                result[name] = source.get_tensor(name).float()
    return result


def rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    variance = value.float().pow(2).mean(-1, keepdim=True)
    return weight * (value * torch.rsqrt(variance + eps))


def rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta **
        (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]
    embedding = torch.cat((freqs, freqs), dim=-1)
    cos = embedding.cos()[:, None, :]
    sin = embedding.sin()[:, None, :]

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        first = value[..., : head_dim // 2]
        second = value[..., head_dim // 2 :]
        return torch.cat((-second, first), dim=-1)

    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    return value[:, None, :, :].expand(
        value.shape[0],
        groups,
        value.shape[1],
        value.shape[2],
    ).reshape(
        value.shape[0] * groups,
        value.shape[1],
        value.shape[2],
    )


def forward(
    input_ids: list[int],
    weights: dict[str, torch.Tensor],
    audio_embeddings: torch.Tensor | None = None,
    audio_start: int = AUDIO_START,
    audio_count: int = AUDIO_COUNT,
) -> torch.Tensor:
    ids = torch.tensor(input_ids, dtype=torch.long)
    hidden = weights[PREFIX + "embed_tokens.weight"][ids]
    if audio_embeddings is not None:
        audio_end = audio_start + audio_count
        hidden[audio_start:audio_end] = audio_embeddings

    sequence_length = hidden.shape[0]
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    groups = num_heads // num_kv_heads
    positions = torch.arange(sequence_length, dtype=torch.long)
    causal = torch.triu(
        torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            dtype=torch.float32,
        ),
        diagonal=1,
    )

    for layer_index in range(28):
        prefix = PREFIX + f"layers.{layer_index}."
        residual = hidden
        normalized = rms_norm(
            hidden,
            weights[prefix + "input_layernorm.weight"],
        )
        query = F.linear(
            normalized,
            weights[prefix + "self_attn.q_proj.weight"],
        ).view(sequence_length, num_heads, head_dim)
        key = F.linear(
            normalized,
            weights[prefix + "self_attn.k_proj.weight"],
        ).view(sequence_length, num_kv_heads, head_dim)
        value = F.linear(
            normalized,
            weights[prefix + "self_attn.v_proj.weight"],
        ).view(sequence_length, num_kv_heads, head_dim)
        query = rms_norm(
            query,
            weights[prefix + "self_attn.q_norm.weight"],
        )
        key = rms_norm(
            key,
            weights[prefix + "self_attn.k_norm.weight"],
        )
        query, key = rope(
            query,
            key,
            positions,
            head_dim,
            1000000.0,
        )
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
        attention = torch.matmul(
            query,
            key.transpose(-1, -2),
        ) * (head_dim ** -0.5)
        attention = torch.softmax(attention + causal, dim=-1)
        attended = torch.matmul(
            attention,
            value,
        ).transpose(0, 1).reshape(sequence_length, -1)
        attended = F.linear(
            attended,
            weights[prefix + "self_attn.o_proj.weight"],
        )
        hidden = residual + attended

        residual = hidden
        normalized = rms_norm(
            hidden,
            weights[prefix + "post_attention_layernorm.weight"],
        )
        gate = F.silu(F.linear(
            normalized,
            weights[prefix + "mlp.gate_proj.weight"],
        ))
        up = F.linear(
            normalized,
            weights[prefix + "mlp.up_proj.weight"],
        )
        hidden = F.linear(
            gate * up,
            weights[prefix + "mlp.down_proj.weight"],
        )
        hidden = residual + hidden

    hidden = rms_norm(
        hidden,
        weights[PREFIX + "norm.weight"],
    )
    return F.linear(
        hidden[-1:],
        weights["thinker.lm_head.weight"],
    ).reshape(-1)


def main() -> None:
    weights = load_weights()
    audio = np.fromfile(
        OUTPUT_DIR / "audio_embedding_ar1_1024x69.f32",
        dtype=np.float32,
    ).reshape(69, 1024)
    audio_tensor = torch.from_numpy(audio)
    long_audio = np.fromfile(
        OUTPUT_DIR / "audio_embedding_raokouling_1024x270.f32",
        dtype=np.float32,
    ).reshape(270, 1024)
    long_audio_tensor = torch.from_numpy(long_audio)

    text_prefill = forward(TEXT_IDS, weights)
    text_decode = forward(TEXT_IDS + [DECODE_TOKEN], weights)
    audio_prefill = forward(
        AUDIO_IDS,
        weights,
        audio_tensor,
        AUDIO_START,
        AUDIO_COUNT,
    )
    audio_decode = forward(
        AUDIO_IDS + [DECODE_TOKEN],
        weights,
        audio_tensor,
        AUDIO_START,
        AUDIO_COUNT,
    )
    long_text_prefill = forward(LONG_TEXT_IDS, weights)
    long_text_decode = forward(
        LONG_TEXT_IDS + [DECODE_TOKEN],
        weights,
    )
    long_audio_prefill = forward(
        LONG_AUDIO_IDS,
        weights,
        long_audio_tensor,
        AUDIO_START,
        LONG_AUDIO_COUNT,
    )
    long_audio_decode = forward(
        LONG_AUDIO_IDS + [DECODE_TOKEN],
        weights,
        long_audio_tensor,
        AUDIO_START,
        LONG_AUDIO_COUNT,
    )

    np.asarray(TEXT_IDS, dtype=np.int32).tofile(
        OUTPUT_DIR / "text_lm_input_ids.i32"
    )
    np.asarray(AUDIO_IDS, dtype=np.int32).tofile(
        OUTPUT_DIR / "audio_lm_input_ids.i32"
    )
    np.asarray([DECODE_TOKEN], dtype=np.int32).tofile(
        OUTPUT_DIR / "text_lm_decode_token.i32"
    )
    np.asarray(LONG_TEXT_IDS, dtype=np.int32).tofile(
        OUTPUT_DIR / "long_text_lm_input_ids.i32"
    )
    np.asarray(LONG_AUDIO_IDS, dtype=np.int32).tofile(
        OUTPUT_DIR / "long_audio_lm_input_ids.i32"
    )
    text_prefill.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "text_lm_prefill_logits.f32"
    )
    text_decode.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "text_lm_decode_logits.f32"
    )
    audio_prefill.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "audio_lm_prefill_logits.f32"
    )
    audio_decode.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "audio_lm_decode_logits.f32"
    )
    long_text_prefill.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "long_text_lm_prefill_logits.f32"
    )
    long_text_decode.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "long_text_lm_decode_logits.f32"
    )
    long_audio_prefill.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "long_audio_lm_prefill_logits.f32"
    )
    long_audio_decode.numpy().astype(np.float32).tofile(
        OUTPUT_DIR / "long_audio_lm_decode_logits.f32"
    )
    manifest = {
        "text_input_ids": "text_lm_input_ids.i32",
        "audio_input_ids": "audio_lm_input_ids.i32",
        "decode_token": DECODE_TOKEN,
        "audio_start": AUDIO_START,
        "audio_token_count": AUDIO_COUNT,
        "text_prefill_logits": "text_lm_prefill_logits.f32",
        "text_decode_logits": "text_lm_decode_logits.f32",
        "audio_prefill_logits": "audio_lm_prefill_logits.f32",
        "audio_decode_logits": "audio_lm_decode_logits.f32",
        "long_text_input_ids": "long_text_lm_input_ids.i32",
        "long_audio_input_ids": "long_audio_lm_input_ids.i32",
        "long_text_prefill_logits": "long_text_lm_prefill_logits.f32",
        "long_text_decode_logits": "long_text_lm_decode_logits.f32",
        "long_audio_prefill_logits": "long_audio_lm_prefill_logits.f32",
        "long_audio_decode_logits": "long_audio_lm_decode_logits.f32",
        "long_audio_start": AUDIO_START,
        "long_audio_token_count": LONG_AUDIO_COUNT,
        "vocab_size": 151936,
    }
    (OUTPUT_DIR / "text_lm.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
