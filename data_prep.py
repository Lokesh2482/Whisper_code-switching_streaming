import os
import random
import pandas as pd
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from tqdm.auto import tqdm
import Whisper_code-switching_streaming.config

def save_monolingual_clips(lang_code, config_name, num_samples):
    """Downloads clips and saves to disk to prevent RAM overload."""
    lang_dir = os.path.join(config.STAGING_DIR, lang_code)
    os.makedirs(lang_dir, exist_ok=True)

    print(f"Staging {lang_code} ({config_name})...")
    ds = load_dataset("google/fleurs", config_name, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    saved_meta = []
    count = 0

    for sample in ds:
        if count >= num_samples: break
        try:
            if sample["audio"]["array"] is None: continue

            # Save audio
            filename = f"{lang_code}_{count:04d}.wav"
            filepath = os.path.join(lang_dir, filename)
            sf.write(filepath, sample["audio"]["array"].flatten(), 16000)

            saved_meta.append({
                "path": filepath,
                "text": sample["transcription"],
                "lang": lang_code
            })
            count += 1
        except Exception: continue

    return saved_meta

def generate_synthetic_data():
    os.makedirs(config.STAGING_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Stage Data
    staged_data = {}
    staged_data["en"] = save_monolingual_clips("en", "en_us", config.SAMPLES_PER_LANG)
    for iso, conf in config.LANG_CONFIG_MAP.items():
        if iso == "en": continue
        staged_data[iso] = save_monolingual_clips(iso, conf, config.SAMPLES_PER_LANG)

    # 2. Space-Aware Mixing Strategy
    print("\nRunning Space-Aware Mixing...")
    manifest = []
    all_langs = list(staged_data.keys())

    for i in tqdm(range(config.TOTAL_CS_SAMPLES), desc="Mixing"):
        try:
            l1, l2 = random.sample(all_langs, 2)
            m1 = random.choice(staged_data[l1])
            m2 = random.choice(staged_data[l2])

            a1, _ = sf.read(m1["path"])
            a2, _ = sf.read(m2["path"])

            # Generate Silence
            silence_dur = random.uniform(0.1, 0.8)
            silence = np.zeros(int(silence_dur * 16000), dtype=np.float32)

            mixed_audio = np.concatenate([a1, silence, a2])

            # Duration Check
            if len(mixed_audio)/16000 > config.MAX_DURATION_SEC: continue

            # --- NOVEL STRATEGY: SPACE-AWARE TEXT ---
            # Explicitly inject space to trigger tokenizer language ID
            mixed_text = f"{m1['text'].strip()} {m2['text'].strip()}"

            out_path = os.path.join(config.OUTPUT_DIR, f"cs_{i:05d}.wav")
            sf.write(out_path, mixed_audio, 16000)

            manifest.append({"audio_filepath": out_path, "text": mixed_text})
        except Exception: continue

    df = pd.DataFrame(manifest)
    os.makedirs(os.path.dirname(config.TRAIN_MANIFEST), exist_ok=True)
    df.to_csv(config.TRAIN_MANIFEST, index=False)
    print(f"Generated {len(df)} training samples.")

if __name__ == "__main__":
    generate_synthetic_data()