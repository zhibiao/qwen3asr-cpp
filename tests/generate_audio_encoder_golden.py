"""Generate development-only Audio Encoder golden fixtures."""

from __future__ import annotations

import argparse
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
PREFIX = "thinker.audio_tower."


def conv_output_length(length: int) -> int:
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return length


def load_audio_weights() -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with safe_open(
        str(MODEL_DIR / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as source:
        for name in source.keys():
            if name.startswith(PREFIX):
                result[name] = source.get_tensor(name).float()
    return result


def positional_embedding(
    positions: int,
    channels: int,
    count: int,
) -> torch.Tensor:
    increment = math.log(10000.0) / (channels // 2 - 1)
    inverse = torch.exp(
        -increment * torch.arange(channels // 2, dtype=torch.float32)
    )
    scaled = (
        torch.arange(positions, dtype=torch.float32)[:, None] *
        inverse[None, :]
    )
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)[:count]


def linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.linear(value, weight, bias)


def audio_encoder(
    features: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mel_bins, frames = features.shape
    chunk_frames = 100
    chunk_lengths = [
        min(chunk_frames, frames - start)
        for start in range(0, frames, chunk_frames)
    ]
    padded = torch.zeros(
        len(chunk_lengths),
        1,
        mel_bins,
        chunk_frames,
        dtype=torch.float32,
    )
    offset = 0
    for chunk, length in enumerate(chunk_lengths):
        padded[chunk, 0, :, :length] = features[:, offset:offset + length]
        offset += length

    hidden = padded
    for index in (1, 2, 3):
        hidden = F.conv2d(
            hidden,
            weights[f"{PREFIX}conv2d{index}.weight"],
            weights[f"{PREFIX}conv2d{index}.bias"],
            stride=2,
            padding=1,
        )
        hidden = F.gelu(hidden)

    batch, channels, frequency, time = hidden.shape
    hidden = hidden.permute(0, 3, 1, 2).reshape(
        batch,
        time,
        channels * frequency,
    )
    hidden = linear(hidden, weights[f"{PREFIX}conv_out.weight"])
    positions = positional_embedding(
        time,
        hidden.shape[-1],
        time,
    )
    hidden = hidden + positions[None]

    valid_chunks = [conv_output_length(length) for length in chunk_lengths]
    hidden = torch.cat(
        [
            hidden[index, :length]
            for index, length in enumerate(valid_chunks)
        ],
        dim=0,
    )
    conv_output = hidden.clone()

    sequence_length = hidden.shape[0]
    attention_window = time * 8
    segments = []
    start = 0
    while start < sequence_length:
        end = min(start + attention_window, sequence_length)
        segments.append((start, end))
        start = end

    num_heads = 14
    head_dim = hidden.shape[-1] // num_heads
    hidden = hidden
    for layer_index in range(18):
        layer_prefix = f"{PREFIX}layers.{layer_index}."
        residual = hidden
        normalized = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            weights[layer_prefix + "self_attn_layer_norm.weight"],
            weights[layer_prefix + "self_attn_layer_norm.bias"],
            1.0e-5,
        )
        query = linear(
            normalized,
            weights[layer_prefix + "self_attn.q_proj.weight"],
            weights[layer_prefix + "self_attn.q_proj.bias"],
        )
        key = linear(
            normalized,
            weights[layer_prefix + "self_attn.k_proj.weight"],
            weights[layer_prefix + "self_attn.k_proj.bias"],
        )
        value = linear(
            normalized,
            weights[layer_prefix + "self_attn.v_proj.weight"],
            weights[layer_prefix + "self_attn.v_proj.bias"],
        )
        query = query.view(sequence_length, num_heads, head_dim).transpose(0, 1)
        key = key.view(sequence_length, num_heads, head_dim).transpose(0, 1)
        value = value.view(sequence_length, num_heads, head_dim).transpose(0, 1)

        attended = torch.empty_like(normalized).view(
            sequence_length,
            num_heads,
            head_dim,
        )
        for start, end in segments:
            part_attention = torch.matmul(
                query[:, start:end],
                key[:, start:end].transpose(-1, -2),
            ) * (head_dim ** -0.5)
            part_attention = torch.softmax(part_attention, dim=-1)
            attended[start:end] = torch.matmul(
                part_attention,
                value[:, start:end],
            ).transpose(0, 1)
        attended = attended.reshape(sequence_length, -1)
        attended = linear(
            attended,
            weights[layer_prefix + "self_attn.out_proj.weight"],
            weights[layer_prefix + "self_attn.out_proj.bias"],
        )
        hidden = residual + attended

        residual = hidden
        normalized = F.layer_norm(
            hidden,
            (hidden.shape[-1],),
            weights[layer_prefix + "final_layer_norm.weight"],
            weights[layer_prefix + "final_layer_norm.bias"],
            1.0e-5,
        )
        hidden = linear(
            normalized,
            weights[layer_prefix + "fc1.weight"],
            weights[layer_prefix + "fc1.bias"],
        )
        hidden = F.gelu(hidden)
        hidden = linear(
            hidden,
            weights[layer_prefix + "fc2.weight"],
            weights[layer_prefix + "fc2.bias"],
        )
        hidden = residual + hidden

    hidden = F.layer_norm(
        hidden,
        (hidden.shape[-1],),
        weights[f"{PREFIX}ln_post.weight"],
        weights[f"{PREFIX}ln_post.bias"],
        1.0e-5,
    )
    hidden = linear(
        hidden,
        weights[f"{PREFIX}proj1.weight"],
        weights[f"{PREFIX}proj1.bias"],
    )
    hidden = F.gelu(hidden)
    hidden = linear(
        hidden,
        weights[f"{PREFIX}proj2.weight"],
        weights[f"{PREFIX}proj2.bias"],
    )
    return conv_output, hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ar1")
    args = parser.parse_args()

    features_path = OUTPUT_DIR / f"logmel_{args.name}_128x"
    candidates = sorted(OUTPUT_DIR.glob(features_path.name + "*.f32"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one log-mel file for {args.name}")
    features = np.fromfile(candidates[0], dtype=np.float32).reshape(
        128,
        -1,
    )

    weights = load_audio_weights()
    conv_output, embeddings = audio_encoder(
        torch.from_numpy(features),
        weights,
    )
    conv_path = OUTPUT_DIR / (
        f"audio_conv_{args.name}_896x{conv_output.shape[0]}.f32"
    )
    embedding_path = OUTPUT_DIR / (
        f"audio_embedding_{args.name}_1024x{embeddings.shape[0]}.f32"
    )
    conv_output.contiguous().numpy().astype(np.float32).tofile(conv_path)
    embeddings.contiguous().numpy().astype(np.float32).tofile(embedding_path)
    manifest = {
        "name": args.name,
        "source_log_mel": candidates[0].name,
        "conv_shape": list(conv_output.shape),
        "conv_file": conv_path.name,
        "embedding_shape": list(embeddings.shape),
        "embedding_file": embedding_path.name,
        "dtype": "float32",
    }
    (OUTPUT_DIR / f"audio_encoder_{args.name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
