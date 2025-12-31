import os

# --- USER CONFIGURATION ---
HF_USERNAME = "Lokesh2482"
REPO_NAME_CT2 = "whisper-large-v3-cs-lora-ct2"  

# --- PATHS ---
BASE_DIR = "data"
STAGING_DIR = os.path.join(BASE_DIR, "staging_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed/cs_synthetic")
TRAIN_MANIFEST = os.path.join(BASE_DIR, "processed/train_cs.csv")

# Model Paths (Local)
BASE_MODEL_ID = "openai/whisper-large-v3"
# Location of the LoRA adapter after training
ADAPTER_LOCAL_PATH = "whisper-cs-lora-encoder-only/final"
# Intermediate path for merging (needed for conversion)
MERGED_LOCAL_PATH = "models/whisper-merged-temp"
# Final CT2 output path
CT2_LOCAL_PATH = "models/whisper-ct2-final"

# --- DATA SETTINGS ---
SAMPLES_PER_LANG = 1500
TOTAL_CS_SAMPLES = 10000
MAX_DURATION_SEC = 29.5

# ISO to FLEURS mapping
LANG_CONFIG_MAP = {
    "en": "en_us", "hi": "hi_in", "bn": "bn_in", "mr": "mr_in",
    "ta": "ta_in", "te": "te_in", "kn": "kn_in", "ml": "ml_in"
}