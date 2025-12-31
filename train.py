import torch
import librosa
from datasets import load_dataset
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import Whisper_code-switching_streaming.config

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 448: labels = labels[:, :448]

        batch["labels"] = labels
        return batch

def run_training():
    print("Loading Dataset...")
    dataset = load_dataset("csv", data_files=config.TRAIN_MANIFEST, split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    processor = WhisperProcessor.from_pretrained(config.BASE_MODEL_ID, task="transcribe")

    def prepare(batch):
        audio = [librosa.load(p, sr=16000)[0] for p in batch["audio_filepath"]]
        batch["input_features"] = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare, batched=True, batch_size=8, remove_columns=["audio_filepath", "text"])

    print("Loading Model in 8-bit...")
    model = WhisperForConditionalGeneration.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # --- NOVEL STRATEGY: ENCODER-ONLY LORA ---
    # Freeze decoder, train encoder to "hear" language switches
    encoder_targets = [n for n,m in model.named_modules()
                       if "encoder" in n and any(x in n for x in ["q_proj", "v_proj", "fc1", "fc2"])]

    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=encoder_targets, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = Seq2SeqTrainingArguments(
        output_dir=config.ADAPTER_LOCAL_PATH,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_steps=100,
        fp16=True,
        save_strategy="steps", save_steps=50,
        logging_steps=10,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=dataset["train"],
        tokenizer=processor.tokenizer,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor)
    )

    print("Starting Training...")
    trainer.train()

    trainer.save_model(f"{config.ADAPTER_LOCAL_PATH}/final")
    processor.save_pretrained(f"{config.ADAPTER_LOCAL_PATH}/final")
    print("Training Complete.")

if __name__ == "__main__":
    run_training()