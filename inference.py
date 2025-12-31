from faster_whisper import WhisperModel
import os
import subprocess
import tempfile
import Whisper_code-switching_streaming.config

# --- LOAD MODEL (From HuggingFace) ---
MODEL_ID = f"{config.HF_USERNAME}/{config.REPO_NAME_CT2}"

def preprocess_audio(audio_path):
    """Ensures audio is 16kHz Mono WAV using FFmpeg"""
    if not os.path.exists(audio_path): return None

    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    output_path = temp_wav.name
    temp_wav.close()

    command = [
        'ffmpeg', '-i', audio_path, '-ar', '16000', '-ac', '1',
        '-c:a', 'pcm_s16le', '-y', output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return output_path
    except:
        return None

def transcribe(audio_path):
    print(f"Loading Model: {MODEL_ID}...")
    # Load optimized model from HF cache
    model = WhisperModel(MODEL_ID, device="cuda", compute_type="float16")

    processed_path = preprocess_audio(audio_path)
    if not processed_path:
        print("Error: Could not process audio file.")
        return

    print(f"Transcribing {audio_path}...")

    segments, info = model.transcribe(
        processed_path,

        # --- OPTIMIZED SETTINGS (Novel Strategies) ---
        beam_size=1,            # Speed: Greedy decoding (5x faster)
        best_of=1,
        condition_on_previous_text=False, # Stability: Prevents "Translation Loops"
        temperature=0.0,        # Stability: Prevents hallucinations

        task="transcribe",
        language="hi",
        # English Inertia Fix: Prime with Devanagari
        initial_prompt="नमस्ते, conversation is in Hindi and English mixed.",

        # --- SEGMENTATION ---
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=True
    )

    print(f"Language: {info.language} ({info.language_probability:.0%})")

    full_text = []
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        full_text.append(segment.text.strip())

    os.remove(processed_path)
    return " ".join(full_text)

if __name__ == "__main__":
    # Replace with your file
    audio_file = "test_audio.mp3"
    transcribe(audio_file)