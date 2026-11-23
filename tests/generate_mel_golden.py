"""Generate development-only Qwen3-ASR audio-frontend golden fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torchaudio
import transformers
from transformers import WhisperFeatureExtractor


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3-ASR-0.6B"
WAV_DIR = ROOT / "test_wavs"
OUTPUT_DIR = ROOT / "testdata" / "qwen3_asr_0_6b"
TARGET_SAMPLE_RATE = 16000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)

    mel_filters = np.asarray(
        feature_extractor.mel_filters,
        dtype=np.float32,
    )
    mel_path = OUTPUT_DIR / "mel_filters_201x128.f32"
    mel_filters.tofile(mel_path)

    files: list[dict[str, object]] = []
    for wav_path in sorted(WAV_DIR.glob("*.wav")):
        source_sample_rate = int(soundfile.info(wav_path).samplerate)
        samples, _ = librosa.load(
            wav_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
        samples = np.ascontiguousarray(samples, dtype=np.float32)

        output = feature_extractor(
            samples,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        log_mel = (
            output["input_features"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        attention_mask = (
            output["attention_mask"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8, copy=False)
        )

        stem = wav_path.stem
        pcm_path = OUTPUT_DIR / f"pcm16k_{stem}.f32"
        log_mel_path = (
            OUTPUT_DIR
            / f"logmel_{stem}_{log_mel.shape[0]}x{log_mel.shape[1]}.f32"
        )
        mask_path = OUTPUT_DIR / f"attention_mask_{stem}.u8"

        samples.tofile(pcm_path)
        log_mel.tofile(log_mel_path)
        attention_mask.tofile(mask_path)

        files.append(
            {
                "name": wav_path.name,
                "source_sample_rate": source_sample_rate,
                "pcm_samples": int(samples.size),
                "pcm_file": pcm_path.name,
                "pcm_sha256": sha256(pcm_path),
                "log_mel_shape": list(log_mel.shape),
                "log_mel_file": log_mel_path.name,
                "log_mel_sha256": sha256(log_mel_path),
                "attention_mask_shape": list(attention_mask.shape),
                "attention_mask_sum": int(attention_mask.sum()),
            }
        )

    metadata = {
        "source_model": str(MODEL_DIR.relative_to(ROOT)).replace("\\", "/"),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "transformers_version": transformers.__version__,
        "sample_rate": TARGET_SAMPLE_RATE,
        "n_fft": feature_extractor.n_fft,
        "hop_length": feature_extractor.hop_length,
        "feature_size": feature_extractor.feature_size,
        "mel_filter_shape": list(mel_filters.shape),
        "mel_filter_file": mel_path.name,
        "mel_filter_sha256": sha256(mel_path),
        "files": files,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
