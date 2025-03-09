import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForMaskedLM,
    Trainer, TrainingArguments, AdamW
)
from datetime import datetime
import paths
import pandas as pd

class PeptideDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.data = [seq for seq in sequences if len(seq) <= 50]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        encoding = self.tokenizer(seq, return_tensors="pt", padding="max_length", max_length=50, truncation=True)
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def load_peptide_sequences(full_dataset_path):
    df = pd.read_csv(full_dataset_path)
    return df['sequence'].tolist()


def train_esm2_hf(esm2_model_name, sequences,from_scratch, epochs=20, batch_size=256, lr=5e-4):
    pretrained_model_name = f"peptide_{esm2_model_name.split('/')[-1]}"
    date = datetime.now().strftime("%Y-%m-%d")
    from_scratch_addition = "_from_scratch" if from_scratch else ""
    pretrained_model_name = f"{pretrained_model_name}{from_scratch_addition}_{date}"
    model_folder = os.path.join(paths.esm2_peptide_pretrained_models_path, pretrained_model_name)
    os.makedirs(model_folder, exist_ok=True)
    logs_folder = os.path.join(model_folder, "logs")
    checkpoints_folder = os.path.join(model_folder, "checkpoints")
    os.makedirs(logs_folder, exist_ok=True)
    config = AutoConfig.from_pretrained(esm2_model_name)
    config.max_position_embeddings = 50  # Set max sequence length to 50
    config.hidden_dropout_prob = 0.1  # Regularization
    if from_scratch:
        model = AutoModelForMaskedLM.from_config(config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(esm2_model_name)
    tokenizer = AutoTokenizer.from_pretrained(esm2_model_name, do_lower_case=False)
    dataset = PeptideDataset(sequences, tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_folder,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Warmup for 10% of total steps
        bf16=True,  # Use bf16 for better stability on H100
        gradient_accumulation_steps=1 if batch_size >= 256 else 2,  # Adjust if needed
        save_strategy="epoch",
        save_total_limit=2,  # Keep only the last 2 checkpoints
        logging_dir=logs_folder,
        logging_steps=100,
        dataloader_num_workers=4,  # Speed up data loading
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01), None)
    )

    trainer.train()
    model.save_pretrained(model_folder)
    tokenizer.save_pretrained(model_folder)

if __name__ == "__main__":
    esm2_model_name = "facebook/esm2_t6_8M_UR50D"
    date = "13_02"
    from_scratch = True
    full_dataset_path = os.path.join(paths.full_datasets_path, f"full_peptide_dataset_{date}.csv")
    peptide_sequences = load_peptide_sequences(full_dataset_path)
    train_esm2_hf(esm2_model_name,peptide_sequences,from_scratch, epochs=20,batch_size=256, lr=5e-4)
