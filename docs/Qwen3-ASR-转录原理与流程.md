# Qwen3-ASR 转录原理与流程

参考实现位于 `Qwen3-ASR.zip`。本文只描述离线转录，不包含时间戳、强制对齐和流式推理。

## 整体架构

Qwen3-ASR 由 Audio Encoder 和 Qwen3 因果语言模型组成：

```text
音频
  -> 16 kHz 单声道 float32
  -> 128 维 log-mel
  -> Audio Encoder
  -> 连续音频 embedding
  -> 替换 prompt 中的 <|audio_pad|>
  -> Qwen3 自回归生成
  -> 解析 language 和 text
```

模型原始输出通常为：

```text
language Arabic<asr_text>转录文本
```

语言和正文由同一个语言模型生成，没有独立语言分类头。

## 音频预处理

输入统一为：

```text
采样率：16000 Hz
声道：单声道
类型：float32
范围：[-1, 1]
```

多声道音频先下混；高于 1 的峰值会缩放并裁剪。输出按 400 点 FFT、160 hop、128 Mel
生成 `log-mel[128, frames]`。

Python 参考实现会对长音频按低能量边界切块。本 C++ 实现不做应用层自动切分；超过
Encoder 或 LM 上下文容量时直接报错。

## Prompt

固定转录模板：

```text
<|im_start|>system
{prompt}<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
<|im_start|>assistant
```

自动语言模式在这里结束 prompt，由模型生成：

```text
language X<asr_text>文本
```

强制语言模式会在 assistant 段后追加：

```text
language X<asr_text>
```

此时模型通常只生成正文。

## 音频 Token 展开

Prompt 初始只有一个 `<|audio_pad|>`。Audio Encoder 输出 N 个 embedding 后，
placeholder 必须展开为 N 个：

```text
<|audio_start|>
<|audio_pad|> × N
<|audio_end|>
```

N 必须与 Encoder 输出帧数完全一致。

## Audio Encoder

主要步骤：

1. log-mel 沿时间维按 100 帧分块，不足部分补零；
2. 三层 `Conv2d(3×3, stride=2, padding=1)`，每层后接 GELU；
3. channel 与 frequency 合并并投影到 `d_model`；
4. 加入位置编码；
5. 执行 18 层窗口化 Transformer attention；
6. 输出投影到语言模型 hidden size。

Qwen3-ASR-0.6B 参数：

```text
mel bins       = 128
n_window       = 50
conv chunk     = 100 frames
n_window_infer = 800
d_model        = 896
heads          = 14
layers         = 18
output_dim     = 1024
```

Attention 按 104 帧固定窗口计算，不同窗口之间不直接互相注意。C++ 实现使用共享的
`104 × 104` mask 批量处理窗口，所有 attention 均通过 `ggml_flash_attn_ext()` 计算，
避免构造完整音频长度或完整文本上下文的 dense attention score。

## 图文 Embedding 合并

语言模型先查完整 prompt 的 token embedding，再在 `<|audio_pad|>` 位置替换为
Audio Encoder 输出：

```text
text embeddings
  + audio embeddings
  + assistant prompt
  -> Qwen3 decoder
```

替换后，后续层不再区分音频 token 和文本 token。

## Qwen3 文本解码

Qwen3-ASR-0.6B 文本模型参数：

```text
hidden_size       = 1024
layers            = 28
query heads       = 16
KV heads          = 8
head_dim          = 128
intermediate_size = 3072
rope_theta        = 1000000
num_ctx           = 4096
```

每层包括：

```text
RMSNorm
Q/K/V projection
Q/K RMSNorm
RoPE
causal GQA attention
O projection
residual
RMSNorm
SwiGLU
residual
```

Prefill 一次处理完整 prompt 并写入 KV cache；decode 每次追加一个 token。

## 生成与采样

默认使用 greedy 解码。采样器支持：

```text
temperature
top-k
top-p
seed
```

生成在以下任一条件停止：

- 采样到 EOS；
- 达到 `max_new_tokens`；
- 无法满足上下文容量。

## 输出解析

有 `<asr_text>`：

```text
language Chinese<asr_text>你好
  -> language = Chinese
  -> text     = 你好
```

无标签时整段视为正文。`language None<asr_text>` 表示无有效语音。

## 当前 C++ 实现

```text
AudioLoader
  -> AudioProcessor
  -> AudioResampler
  -> AudioFrontend
  -> AutEncoder
  -> PromptBuilder
  -> Qwen3Lm
  -> TextTokenizer
  -> Qwen3AsrResult
```

当前 C++ 支持单块 20 分钟音频，Encoder 会产生约 15600 个音频 token。LM 的
`num_ctx` 必须覆盖 prompt 和生成 token，可通过运行时 `--num-ctx` 调节；20 分钟
推荐至少 20480。

运行期只读取 split GGUF，不依赖 PyTorch、Transformers 或 Python。
