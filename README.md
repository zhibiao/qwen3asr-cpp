# Qwen3-ASR C++ / GGML

从 Hugging Face 模型目录导出 Encoder 和 LM 两个 GGUF 文件，然后使用
`qwen3-asr` 进行离线转录。

## 模型转换

### Windows

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

.\.venv\Scripts\python.exe .\convert_hf_to_gguf.py `
  --model .\models\Qwen3-ASR-0.6B `
  --output .\models\qwen3-asr-0.6b.gguf
```

默认生成 BF16 文件：

```text
models/qwen3-asr-0.6b-encoder-bf16.gguf
models/qwen3-asr-0.6b-lm-bf16.gguf
```

### Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python convert_hf_to_gguf.py \
  --model ./models/Qwen3-ASR-0.6B \
  --output ./models/qwen3-asr-0.6b.gguf
```

指定其他权重类型：

```text
--dtype f32
--dtype f16
--dtype bf16
```

指定 KV cache 上下文长度：

```text
--num-ctx 4096
```

### Q8_0 量化

分别量化 Encoder 或 LM：

```powershell
.\.venv\Scripts\python.exe .\quantize_gguf.py `
  --model encoder `
  --input .\models\qwen3-asr-0.6b-encoder-f16.gguf

.\.venv\Scripts\python.exe .\quantize_gguf.py `
  --model lm `
  --input .\models\qwen3-asr-0.6b-lm-f16.gguf
```

Linux：

```bash
python quantize_gguf.py \
  --model encoder \
  --input ./models/qwen3-asr-0.6b-encoder-f16.gguf

python quantize_gguf.py \
  --model lm \
  --input ./models/qwen3-asr-0.6b-lm-f16.gguf
```

默认输出为：

```text
models/qwen3-asr-0.6b-encoder-q8_0.gguf
models/qwen3-asr-0.6b-lm-q8_0.gguf
```

只支持 Q8_0。矩阵权重量化为 Q8_0，norm、bias、卷积核和 Mel filter 保持原类型。

## 转录

### Windows

```powershell
cmake --preset windows-cpu
cmake --build --preset windows-cpu

.\build\windows-cpu\bin\Release\qwen3-asr.exe `
  --encoder .\models\qwen3-asr-0.6b-encoder-bf16.gguf `
  --lm .\models\qwen3-asr-0.6b-lm-bf16.gguf `
  --audio .\test_wavs\raokouling.wav
```

Windows CUDA（需要设置 `CUDA_PATH`）：

```powershell
cmake --preset windows-cuda
cmake --build --preset windows-cuda

.\build\windows-cuda\bin\Release\qwen3-asr.exe `
  --encoder .\models\qwen3-asr-0.6b-encoder-bf16.gguf `
  --lm .\models\qwen3-asr-0.6b-lm-bf16.gguf `
  --audio .\test_wavs\raokouling.wav `
  --device cuda
```

### Linux

```bash
cmake --preset linux-cpu
cmake --build --preset linux-cpu

./build/linux-cpu/bin/qwen3-asr \
  --encoder ./models/qwen3-asr-0.6b-encoder-bf16.gguf \
  --lm ./models/qwen3-asr-0.6b-lm-bf16.gguf \
  --audio ./test_wavs/raokouling.wav
```

Linux CUDA：

```bash
cmake --preset linux-cuda
cmake --build --preset linux-cuda

./build/linux-cuda/bin/qwen3-asr \
  --encoder ./models/qwen3-asr-0.6b-encoder-bf16.gguf \
  --lm ./models/qwen3-asr-0.6b-lm-bf16.gguf \
  --audio ./test_wavs/raokouling.wav \
  --device cuda
```

可选参数：

```text
--device cpu|cuda|CUDA0
--context <text>
--language <name>
--num-ctx <count>
--max-new-tokens <count>
--temperature <value>
--top-k <count>
--top-p <value>
--seed <value>
```

默认使用 greedy 采样和 1024 个最大生成 token。不传 `--audio` 时只加载并输出
模型信息。

单块 20 分钟音频需要显式提高 LM 上下文，例如：

```text
--num-ctx 20480
```
