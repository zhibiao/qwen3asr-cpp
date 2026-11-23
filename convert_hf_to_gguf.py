"""Export a Qwen3-ASR checkpoint as split encoder and LM GGUF files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open
from transformers import WhisperFeatureExtractor


FORMAT_VERSION = 2
DEFAULT_DTYPE = "bf16"
DEFAULT_NUM_CTX = 4096
SUPPORTED_DTYPES = ("f32", "f16", "bf16")
WEIGHTS_LAYOUT = "huggingface_safetensors_v1"

AUDIO_LAYER_TENSORS = {
    "fc1.bias",
    "fc1.weight",
    "fc2.bias",
    "fc2.weight",
    "final_layer_norm.bias",
    "final_layer_norm.weight",
    "self_attn.k_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.out_proj.bias",
    "self_attn.out_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn_layer_norm.bias",
    "self_attn_layer_norm.weight",
}

TEXT_LAYER_TENSORS = {
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_merges(path: Path) -> list[str]:
    merges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            merges.append(line)
    return merges


def discover_safetensors(
    model_dir: Path,
) -> tuple[list[Path], set[str] | None]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        shards = sorted(
            model_dir / filename
            for filename in set(weight_map.values())
        )
        missing = [path for path in shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing safetensors shard(s): " +
                ", ".join(str(path) for path in missing)
            )
        return shards, set(weight_map)

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model.safetensors found in {model_dir}")
    return shards, None


def scan_model_tensors(
    shards: list[Path],
) -> tuple[dict[str, dict[str, object]], dict[Path, list[str]]]:
    tensors: dict[str, dict[str, object]] = {}
    names_by_shard: dict[Path, list[str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            names_by_shard[shard] = names
            for name in names:
                if name in tensors:
                    raise ValueError(f"duplicate safetensors tensor: {name}")
                tensor_slice = handle.get_slice(name)
                tensors[name] = {
                    "shape": list(tensor_slice.get_shape()),
                    "dtype": tensor_slice.get_dtype(),
                }
    return tensors, names_by_shard


def validate_model_tensors(
    config: dict,
    tensors: dict[str, dict[str, object]],
    indexed_names: set[str] | None,
) -> None:
    names = set(tensors)
    if indexed_names is not None and names != indexed_names:
        missing = sorted(indexed_names - names)
        extra = sorted(names - indexed_names)
        raise ValueError(
            "safetensors shards disagree with index; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    thinker = config["thinker_config"]
    audio_layers = int(thinker["audio_config"]["encoder_layers"])
    text_layers = int(thinker["text_config"]["num_hidden_layers"])
    expected = {
        "thinker.audio_tower.conv2d1.bias",
        "thinker.audio_tower.conv2d1.weight",
        "thinker.audio_tower.conv2d2.bias",
        "thinker.audio_tower.conv2d2.weight",
        "thinker.audio_tower.conv2d3.bias",
        "thinker.audio_tower.conv2d3.weight",
        "thinker.audio_tower.conv_out.weight",
        "thinker.audio_tower.ln_post.bias",
        "thinker.audio_tower.ln_post.weight",
        "thinker.audio_tower.proj1.bias",
        "thinker.audio_tower.proj1.weight",
        "thinker.audio_tower.proj2.bias",
        "thinker.audio_tower.proj2.weight",
        "thinker.model.embed_tokens.weight",
        "thinker.model.norm.weight",
        "thinker.lm_head.weight",
    }
    for layer in range(audio_layers):
        prefix = f"thinker.audio_tower.layers.{layer}."
        expected.update(prefix + suffix for suffix in AUDIO_LAYER_TENSORS)
    for layer in range(text_layers):
        prefix = f"thinker.model.layers.{layer}."
        expected.update(prefix + suffix for suffix in TEXT_LAYER_TENSORS)

    missing = sorted(expected - names)
    unexpected = sorted(names - expected)
    if missing or unexpected:
        raise ValueError(
            "checkpoint tensor set does not match Qwen3-ASR config; "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )


def tensor_belongs_to_part(name: str, part: str) -> bool:
    if part == "encoder":
        return name.startswith("thinker.audio_tower.")
    if part == "lm":
        return (
            name.startswith("thinker.model.") or
            name.startswith("thinker.lm_head.")
        )
    raise ValueError(f"unknown GGUF part: {part}")


def target_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "f32":
        return torch.float32
    if dtype == "f16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported output dtype: {dtype}")


def gguf_tensor_payload(
    tensor: torch.Tensor,
) -> tuple[np.ndarray, gguf.GGMLQuantizationType, str]:
    tensor = tensor.detach().contiguous()
    if tensor.dtype == torch.bfloat16:
        return (
            tensor.view(torch.uint16).numpy(),
            gguf.GGMLQuantizationType.BF16,
            "BF16",
        )
    if tensor.dtype == torch.float16:
        return (
            tensor.view(torch.uint16).numpy(),
            gguf.GGMLQuantizationType.F16,
            "F16",
        )
    if tensor.dtype == torch.float32:
        return tensor.numpy(), gguf.GGMLQuantizationType.F32, "F32"
    raise TypeError(f"unsupported tensor dtype: {tensor.dtype}")


def write_model_tensors(
    writer: gguf.GGUFWriter,
    tensors: dict[str, dict[str, object]],
    names_by_shard: dict[Path, list[str]],
    part: str,
    dtype: str,
) -> dict[str, object]:
    byte_count = 0
    written = 0
    actual_dtype = ""
    target_dtype = target_torch_dtype(dtype)
    for shard, names in names_by_shard.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in names:
                if not tensor_belongs_to_part(name, part):
                    continue
                tensor = handle.get_tensor(name)
                if list(tensor.shape) != tensors[name]["shape"]:
                    raise ValueError(
                        f"tensor shape changed while reading: {name}"
                    )
                tensor = tensor.to(target_dtype)
                payload, gguf_dtype, actual_dtype = gguf_tensor_payload(
                    tensor
                )
                writer.add_tensor(
                    name,
                    payload,
                    raw_shape=list(tensor.shape),
                    raw_dtype=gguf_dtype,
                )
                byte_count += payload.nbytes
                written += 1
                del payload
                del tensor

    expected_count = sum(
        tensor_belongs_to_part(name, part) for name in tensors
    )
    if written != expected_count:
        raise ValueError(
            f"wrote {written} {part} tensors, expected {expected_count}"
        )
    return {
        "count": written,
        "bytes": byte_count,
        "dtype": actual_dtype,
    }


def compute_bundle_id(
    model_dir: Path,
    tensors: dict[str, dict[str, object]],
) -> str:
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "configuration.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
        "vocab.json",
        "merges.txt",
    ):
        path = model_dir / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in sorted(tensors):
        info = tensors[name]
        digest.update(
            f"{name}\t{info['shape']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def special_id(
    token: str,
    vocab: dict[str, int],
    added: dict[str, int],
) -> int:
    if token in added:
        return added[token]
    if token in vocab:
        return vocab[token]
    raise KeyError(f"missing tokenizer special token: {token}")


def write_common_metadata(
    writer: gguf.GGUFWriter,
    model_dir: Path,
    config: dict,
    bundle_id: str,
    part: str,
) -> None:
    writer.add_name(model_dir.name)
    writer.add_string("qwen3_asr.model.family", "Qwen3-ASR")
    writer.add_string("qwen3_asr.model.model_type", config["model_type"])
    writer.add_string(
        "qwen3_asr.model.architecture",
        config["architectures"][0],
    )
    writer.add_int32("qwen3_asr.format_version", FORMAT_VERSION)
    writer.add_string("qwen3_asr.bundle.id", bundle_id)
    writer.add_string("qwen3_asr.part", part)
    writer.add_array(
        "qwen3_asr.model.supported_languages",
        config.get("support_languages", []),
    )


def write_audio_metadata(
    writer: gguf.GGUFWriter,
    audio: dict,
    preprocessor: dict,
    sample_rate: int,
) -> None:
    writer.add_int32("qwen3_asr.audio.d_model", int(audio["d_model"]))
    writer.add_int32(
        "qwen3_asr.audio.encoder_layers",
        int(audio["encoder_layers"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.encoder_attention_heads",
        int(audio["encoder_attention_heads"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.encoder_ffn_dim",
        int(audio["encoder_ffn_dim"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.num_mel_bins",
        int(audio["num_mel_bins"]),
    )
    writer.add_int32("qwen3_asr.audio.n_window", int(audio["n_window"]))
    writer.add_int32(
        "qwen3_asr.audio.n_window_infer",
        int(audio["n_window_infer"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.max_source_positions",
        int(audio["max_source_positions"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.output_dim",
        int(audio["output_dim"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.downsample_hidden_size",
        int(audio["downsample_hidden_size"]),
    )
    writer.add_int32(
        "qwen3_asr.audio.conv_chunksize",
        int(audio["conv_chunksize"]),
    )
    writer.add_string(
        "qwen3_asr.audio.activation_function",
        str(audio["activation_function"]),
    )
    writer.add_float32(
        "qwen3_asr.audio.layer_norm_eps",
        1.0e-5,
    )

    writer.add_int32("qwen3_asr.preprocessor.sample_rate", sample_rate)
    writer.add_int32(
        "qwen3_asr.preprocessor.feature_size",
        int(preprocessor["feature_size"]),
    )
    writer.add_int32(
        "qwen3_asr.preprocessor.n_fft",
        int(preprocessor["n_fft"]),
    )
    writer.add_int32(
        "qwen3_asr.preprocessor.hop_length",
        int(preprocessor["hop_length"]),
    )
    writer.add_int32(
        "qwen3_asr.preprocessor.chunk_length",
        int(preprocessor["chunk_length"]),
    )
    writer.add_int32(
        "qwen3_asr.preprocessor.n_samples",
        int(preprocessor["n_samples"]),
    )
    writer.add_int32(
        "qwen3_asr.preprocessor.nb_max_frames",
        int(preprocessor["nb_max_frames"]),
    )
    writer.add_float32(
        "qwen3_asr.preprocessor.dither",
        float(preprocessor["dither"]),
    )
    writer.add_float32(
        "qwen3_asr.preprocessor.padding_value",
        float(preprocessor["padding_value"]),
    )
    writer.add_float32(
        "qwen3_asr.preprocessor.mel_min_frequency",
        0.0,
    )
    writer.add_float32(
        "qwen3_asr.preprocessor.mel_max_frequency",
        sample_rate / 2.0,
    )
    writer.add_string("qwen3_asr.preprocessor.mel_scale", "slaney")
    writer.add_string("qwen3_asr.preprocessor.mel_norm", "slaney")
    writer.add_bool(
        "qwen3_asr.preprocessor.return_attention_mask",
        bool(preprocessor.get("return_attention_mask", True)),
    )


def write_text_metadata(
    writer: gguf.GGUFWriter,
    text: dict,
    num_ctx: int,
) -> None:
    writer.add_int32(
        "qwen3_asr.text.hidden_size",
        int(text["hidden_size"]),
    )
    writer.add_int32(
        "qwen3_asr.text.layers",
        int(text["num_hidden_layers"]),
    )
    writer.add_int32(
        "qwen3_asr.text.attention_heads",
        int(text["num_attention_heads"]),
    )
    writer.add_int32(
        "qwen3_asr.text.kv_heads",
        int(text["num_key_value_heads"]),
    )
    writer.add_int32(
        "qwen3_asr.text.head_dim",
        int(text["head_dim"]),
    )
    writer.add_int32(
        "qwen3_asr.text.vocab_size",
        int(text["vocab_size"]),
    )
    writer.add_int32(
        "qwen3_asr.text.max_position_embeddings",
        int(text["max_position_embeddings"]),
    )
    writer.add_int32("qwen3_asr.text.num_ctx", num_ctx)
    writer.add_float32(
        "qwen3_asr.text.rope_theta",
        float(text["rope_theta"]),
    )
    writer.add_float32(
        "qwen3_asr.text.rms_norm_eps",
        float(text["rms_norm_eps"]),
    )
    writer.add_int32(
        "qwen3_asr.text.intermediate_size",
        int(text["intermediate_size"]),
    )
    writer.add_bool(
        "qwen3_asr.text.tie_word_embeddings",
        bool(text.get("tie_word_embeddings", False)),
    )
    writer.add_array(
        "qwen3_asr.text.mrope_section",
        [
            int(value)
            for value in text.get("rope_scaling", {}).get(
                "mrope_section",
                [],
            )
        ],
    )


def write_tokenizer_metadata(
    writer: gguf.GGUFWriter,
    tokenizer_config: dict,
    generation_config: dict,
    vocab: dict[str, int],
    merges: list[str],
) -> None:
    added_tokens: list[tuple[int, str, bool]] = []
    added_by_content: dict[str, int] = {}
    for id_text, value in tokenizer_config["added_tokens_decoder"].items():
        token_id = int(id_text)
        content = value["content"]
        added_tokens.append((token_id, content, bool(value["special"])))
        added_by_content[content] = token_id
    added_tokens.sort()

    maximum_id = max(
        list(vocab.values()) +
        [token_id for token_id, _, _ in added_tokens]
    )
    tokens = [""] * (maximum_id + 1)
    token_types = [int(gguf.TokenType.NORMAL)] * len(tokens)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    for token_id, content, special in added_tokens:
        tokens[token_id] = content
        token_types[token_id] = int(
            gguf.TokenType.CONTROL
            if special
            else gguf.TokenType.USER_DEFINED
        )
    if any(not token for token in tokens):
        missing = [index for index, token in enumerate(tokens) if not token]
        raise ValueError(f"token IDs contain holes: {missing[:20]}")

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_int32(
        "qwen3_asr.tokenizer.model_max_length",
        int(tokenizer_config["model_max_length"]),
    )
    writer.add_int32(
        "qwen3_asr.tokenizer.eos_token_id",
        special_id(
            tokenizer_config["eos_token"],
            vocab,
            added_by_content,
        ),
    )
    writer.add_int32(
        "qwen3_asr.tokenizer.pad_token_id",
        special_id(
            tokenizer_config["pad_token"],
            vocab,
            added_by_content,
        ),
    )
    special_tokens = {
        "im_start_token_id": "<|im_start|>",
        "im_end_token_id": "<|im_end|>",
        "audio_start_token_id": "<|audio_start|>",
        "audio_pad_token_id": "<|audio_pad|>",
        "audio_end_token_id": "<|audio_end|>",
        "asr_text_token_id": "<asr_text>",
    }
    for key, token in special_tokens.items():
        writer.add_int32(
            f"qwen3_asr.tokenizer.{key}",
            special_id(token, vocab, added_by_content),
        )
    writer.add_array(
        "qwen3_asr.tokenizer.added_token_ids",
        [token_id for token_id, _, _ in added_tokens],
    )
    writer.add_array(
        "qwen3_asr.tokenizer.added_token_special",
        [1 if special else 0 for _, _, special in added_tokens],
    )

    generation_eos = generation_config.get("eos_token_id", [])
    if isinstance(generation_eos, int):
        generation_eos = [generation_eos]
    writer.add_array(
        "qwen3_asr.generation.eos_token_ids",
        [int(token_id) for token_id in generation_eos],
    )
    writer.add_bool(
        "qwen3_asr.generation.do_sample",
        bool(generation_config.get("do_sample", False)),
    )
    writer.add_string(
        "qwen3_asr.prompt.version",
        "qwen3_asr_transcription_v1",
    )
    writer.add_string(
        "qwen3_asr.prompt.template",
        (
            "<|im_start|>system\n{context}<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    )


def write_weight_metadata(
    writer: gguf.GGUFWriter,
    part: str,
    stats: dict[str, object],
) -> None:
    writer.add_string("qwen3_asr.weights.layout", WEIGHTS_LAYOUT)
    writer.add_string(
        "qwen3_asr.weights.dtype",
        str(stats["dtype"]),
    )
    writer.add_array(
        "qwen3_asr.weights.type_names",
        [str(stats["dtype"])],
    )
    writer.add_array(
        "qwen3_asr.weights.type_counts",
        [int(stats["count"])],
    )
    writer.add_int32(
        "qwen3_asr.weights.tensor_count",
        int(stats["count"]),
    )
    writer.add_int32(
        "qwen3_asr.weights.audio_tensor_count",
        int(stats["count"]) if part == "encoder" else 0,
    )
    writer.add_int32(
        "qwen3_asr.weights.text_tensor_count",
        310 if part == "lm" else 0,
    )
    writer.add_int32(
        "qwen3_asr.weights.lm_head_tensor_count",
        1 if part == "lm" else 0,
    )
    writer.add_int64(
        "qwen3_asr.weights.byte_count",
        int(stats["bytes"]),
    )


def export_part(
    output_path: Path,
    part: str,
    model_dir: Path,
    config: dict,
    preprocessor: dict,
    tokenizer_config: dict,
    generation_config: dict,
    vocab: dict[str, int],
    merges: list[str],
    mel_filters: np.ndarray,
    bundle_id: str,
    model_tensors: dict[str, dict[str, object]],
    names_by_shard: dict[Path, list[str]],
    dtype: str,
    num_ctx: int,
) -> dict[str, object]:
    writer = gguf.GGUFWriter(
        output_path,
        "qwen3_asr",
        use_temp_file=True,
    )
    write_common_metadata(writer, model_dir, config, bundle_id, part)
    thinker = config["thinker_config"]
    if part == "encoder":
        sample_rate = int(
            preprocessor["n_samples"] // preprocessor["chunk_length"]
        )
        write_audio_metadata(
            writer,
            thinker["audio_config"],
            preprocessor,
            sample_rate,
        )
    else:
        write_text_metadata(writer, thinker["text_config"], num_ctx)
        write_tokenizer_metadata(
            writer,
            tokenizer_config,
            generation_config,
            vocab,
            merges,
        )

    stats = write_model_tensors(
        writer,
        model_tensors,
        names_by_shard,
        part,
        dtype,
    )
    write_weight_metadata(writer, part, stats)
    if part == "encoder":
        writer.add_tensor("preprocessor.mel_filters", mel_filters)

    writer.write_header_to_file(output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return stats


def convert(
    model_dir: Path,
    output_path: Path,
    dtype: str = DEFAULT_DTYPE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> tuple[Path, Path]:
    config = read_json(model_dir / "config.json")
    preprocessor = read_json(model_dir / "preprocessor_config.json")
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    generation_config = read_json(model_dir / "generation_config.json")
    vocab = read_json(model_dir / "vocab.json")
    merges = load_merges(model_dir / "merges.txt")
    shards, indexed_names = discover_safetensors(model_dir)
    model_tensors, names_by_shard = scan_model_tensors(shards)
    validate_model_tensors(config, model_tensors, indexed_names)
    bundle_id = compute_bundle_id(model_dir, model_tensors)
    max_position_embeddings = int(
        config["thinker_config"]["text_config"]["max_position_embeddings"]
    )
    if num_ctx <= 0 or num_ctx > max_position_embeddings:
        raise ValueError(
            f"num_ctx must be in [1, {max_position_embeddings}], "
            f"got {num_ctx}"
        )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    mel_filters = np.asarray(
        feature_extractor.mel_filters,
        dtype=np.float32,
    )
    expected_shape = (
        1 + int(preprocessor["n_fft"]) // 2,
        int(preprocessor["feature_size"]),
    )
    if mel_filters.shape != expected_shape:
        raise ValueError(
            f"unexpected Mel filter shape {mel_filters.shape}, "
            f"expected {expected_shape}"
        )

    prefix = (
        output_path.with_suffix("")
        if output_path.suffix.lower() == ".gguf"
        else output_path
    )
    if output_path.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError(
            "GGUF output must be outside the PyTorch model directory"
        )
    encoder_path = prefix.parent / f"{prefix.name}-encoder-{dtype}.gguf"
    lm_path = prefix.parent / f"{prefix.name}-lm-{dtype}.gguf"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    encoder_stats = export_part(
        encoder_path,
        "encoder",
        model_dir,
        config,
        preprocessor,
        tokenizer_config,
        generation_config,
        vocab,
        merges,
        mel_filters,
        bundle_id,
        model_tensors,
        names_by_shard,
        dtype,
        num_ctx,
    )
    lm_stats = export_part(
        lm_path,
        "lm",
        model_dir,
        config,
        preprocessor,
        tokenizer_config,
        generation_config,
        vocab,
        merges,
        mel_filters,
        bundle_id,
        model_tensors,
        names_by_shard,
        dtype,
        num_ctx,
    )
    print(
        f"wrote {encoder_path} "
        f"({encoder_stats['count']} weights + Mel)"
    )
    print(f"wrote {lm_path} ({lm_stats['count']} weights)")
    return encoder_path, lm_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Qwen3-ASR HF checkpoint to split GGUF files"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output prefix or .gguf path; component suffix is added",
    )
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_DTYPES,
        type=str.lower,
        default=DEFAULT_DTYPE,
        help="Model tensor dtype (default: bf16)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="KV cache context length (default: 4096)",
    )
    args = parser.parse_args()
    convert(args.model, args.output, args.dtype, args.num_ctx)


if __name__ == "__main__":
    main()
