# Whisper-Large-v3 Indic Code-Switching (Streaming Optimized)

![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models_Available-orange)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

A high-performance **Multilingual Code-Switching ASR system** derived from `openai/whisper-large-v3`.

This model is fine-tuned on a diverse dataset of **English mixed with 7+ Indic languages** (Hindi, Bengali, Marathi, Tamil, Telugu, Kannada, Malayalam, etc.). It utilizes novel data engineering and training strategies to achieve robust mixed-language transcription with **real-time streaming capabilities (3-5x faster than base)**.

**Model:** [Lokesh2482/whisper-large-v3-cs-lora-ct2](https://huggingface.co/Lokesh2482/whisper-large-v3-cs-lora-ct2)

---

## Strategies : 

We addressed three critical challenges in Multilingual Code-Switching ASR:

### 1. Space-Aware Data Mixing (Data Engineering)
Concatenating audio clips from different languages blindly confuses the tokenizer. We implemented a **Space-Aware Mixing Strategy**:
- **Technique:** When synthetically mixing audio from different languages (e.g., English + Indic), we explicitly inject spaces in the ground truth text: `f"{text1} {text2}"`.
- **Effect:** This artificial boundary acts as a delimiter, triggering Whisper's internal language identification probabilities at the exact acoustic boundary. This prevents the model from "gluing" words from different scripts together (e.g., ensuring Latin and Indic scripts remain distinct).

### 2. Encoder-Only LoRA Fine-Tuning (Training)
We specifically targeted the **Encoder** layers for Low-Rank Adaptation (LoRA), leaving the Decoder completely frozen.
- **Hypothesis:** The pre-trained Decoder already possesses excellent knowledge of grammar and text generation for all target languages. The bottleneck in code-switching is the Encoder's ability to "hear" the rapid acoustic shift between languages.
- **Result:** The model learns to detect language transitions acoustically without overwriting or degrading the grammatical stability of the pre-trained Decoder.

### 3. CTranslate2 + Script Inertia Fix (Inference)
- **Speed:** The model is quantized and converted to **CTranslate2 (Float16)** format, reducing inference latency by **300-500%** compared to standard PyTorch implementation.
- **Inertia Fix:** Standard models often suffer from "Script Inertia," where they get stuck in English/Latin script even when the audio switches to an Indic language. We implemented a **Context Priming Strategy** during inference, forcing the model to keep the probability distribution active for multiple scripts, ensuring accurate switching.

---

## Quick Start : 

### 1. Installation
```bash
# Clone repository
git clone [https://github.com/Lokesh2482/Whisper-Indic-CodeSwitching-Streaming.git](https://github.com/Lokesh2482/Whisper-Indic-CodeSwitching-Streaming.git)
cd Whisper-Indic-CodeSwitching-Streaming

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (Required for audio processing)
sudo apt-get install ffmpeg