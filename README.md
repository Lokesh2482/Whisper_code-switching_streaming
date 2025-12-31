# Whisper-Large-v3 Indic Code-Switching (Streaming Optimized)

![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models_Available-orange)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

A high-performance **Multilingual Code-Switching ASR system** derived from `openai/whisper-large-v3`.

This model is fine-tuned on a diverse dataset of **English mixed with 7+ Indic languages** (Hindi, Bengali, Marathi, Tamil, Telugu, Kannada, Malayalam, etc.). It utilizes novel data engineering and training strategies to achieve robust mixed-language transcription with **real-time streaming capabilities (3-5x faster than base)**.

**Model:** [Lokesh2482/whisper-large-v3-cs-lora-ct2](https://huggingface.co/Lokesh2482/whisper-large-v3-cs-lora-ct2)

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