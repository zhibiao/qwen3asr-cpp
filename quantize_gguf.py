"""Quantize a Qwen3-ASR split GGUF file to Q8_0."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import gguf
import numpy as np


QUANT_TYPE = gguf.GGMLQuantizationType.Q8_0
QUANT_NAME = "Q8_0"
SKIP_METADATA = {
    "general.architecture",
    "general.alignment",
    "qwen3_asr.weights.dtype",
    "qwen3_asr.weights.type_names",
    "qwen3_asr.weights.type_counts",
    "qwen3_asr.weights.byte_count",
    "qwen3_asr.weights.tensor_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a Qwen3-ASR split GGUF file to Q8_0."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=("encoder", "lm"),
        help="GGUF component to quantize",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input F16/F32 split GGUF file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Q8_0 GGUF file (default: input-q8_0.gguf)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("-f16") or stem.endswith("-f32"):
        stem = stem[:-4]
    return input_path.with_name(f"{stem}-q8_0.gguf")


def metadata_value(field: gguf.ReaderField) -> tuple[object, object, object]:
    value = field.contents()
    value_type = field.types[0]
    sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
    return value, value_type, sub_type


def copy_metadata(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
) -> None:
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in SKIP_METADATA:
            continue
        value, value_type, sub_type = metadata_value(field)
        writer.add_key_value(name, value, value_type, sub_type)


def tensor_data(tensor: gguf.ReaderTensor) -> np.ndarray:
    if tensor.tensor_type not in (
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.F32,
    ):
        raise ValueError(
            f"unsupported source dtype for {tensor.name}: "
            f"{tensor.tensor_type.name}"
        )
    return np.ascontiguousarray(tensor.data)


def should_quantize(tensor: gguf.ReaderTensor, data: np.ndarray) -> bool:
    if tensor.name == "preprocessor.mel_filters":
        return False
    if data.ndim < 2 or not tensor.name.endswith(".weight"):
        return False
    return gguf.Q8_0.can_quantize(data)


def quantize(
    model_part: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input GGUF does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} (use --overwrite)"
        )

    reader = gguf.GGUFReader(input_path)
    part_field = reader.fields.get("qwen3_asr.part")
    if part_field is None:
        raise ValueError("input GGUF has no qwen3_asr.part metadata")
    actual_part = str(part_field.contents())
    if actual_part != model_part:
        raise ValueError(
            f"--model={model_part} does not match input part={actual_part}"
        )

    architecture_field = reader.fields.get("general.architecture")
    if architecture_field is None:
        raise ValueError("input GGUF has no general.architecture metadata")
    architecture = str(architecture_field.contents())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(
        output_path,
        arch=architecture,
        use_temp_file=True,
    )
    copy_metadata(reader, writer)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_quantized_by("qwen3-asr-cpp")

    type_counts: Counter[str] = Counter()
    weight_count = 0
    weight_bytes = 0
    quantized_count = 0
    try:
        for tensor in reader.tensors:
            if tensor.name == "preprocessor.mel_filters":
                writer.add_tensor(
                    tensor.name,
                    tensor_data(tensor),
                    raw_shape=tensor.data.shape,
                    raw_dtype=tensor.tensor_type,
                )
                continue

            data = tensor_data(tensor)
            if should_quantize(tensor, data):
                payload = gguf.Q8_0.quantize(data)
                payload_type = QUANT_TYPE
                payload_shape = payload.shape
                quantized_count += 1
            else:
                payload = data
                payload_type = tensor.tensor_type
                payload_shape = data.shape

            writer.add_tensor(
                tensor.name,
                payload,
                raw_shape=payload_shape,
                raw_dtype=payload_type,
            )
            type_counts[payload_type.name] += 1
            weight_count += 1
            weight_bytes += payload.nbytes

        if quantized_count == 0:
            raise ValueError("no tensors were eligible for Q8_0 quantization")

        type_names = sorted(
            type_counts,
            key=lambda name: (name != QUANT_NAME, name),
        )
        writer.add_string("qwen3_asr.weights.dtype", type_names[0])
        writer.add_array("qwen3_asr.weights.type_names", type_names)
        writer.add_array(
            "qwen3_asr.weights.type_counts",
            [type_counts[name] for name in type_names],
        )
        writer.add_int32(
            "qwen3_asr.weights.tensor_count",
            weight_count,
        )
        writer.add_int64(
            "qwen3_asr.weights.byte_count",
            weight_bytes,
        )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    print(
        f"wrote {output_path} "
        f"({model_part}, Q8_0={quantized_count}, "
        f"weights={weight_count}, bytes={weight_bytes})"
    )


def main() -> None:
    args = parse_args()
    output = args.output or default_output_path(args.input)
    quantize(args.model, args.input, output, args.overwrite)


if __name__ == "__main__":
    main()
