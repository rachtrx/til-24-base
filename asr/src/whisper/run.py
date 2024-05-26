import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import Optional, Dict, Union, List
from dataclasses import dataclass

import numpy as np
import random
import re
import json
import jsonlines
# from tqdm import tqdm

import multiprocessing as mp

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from jiwer import wer
import whisper

import torch
from torch.utils.data import IterableDataset, DataLoader
import torchaudio
from torchaudio import transforms
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from tqdm import tqdm

cur_dir = os.getcwd()
src_dir = os.path.dirname(cur_dir)
til_dir = os.path.dirname(os.path.dirname(src_dir))
home_dir = os.path.dirname(til_dir)
novice_dir = os.path.join(home_dir, 'novice')
audio_dir = os.path.join(novice_dir, 'audio')
data_dir = os.path.join(cur_dir, 'data')
model_path = os.path.join(src_dir, "models", "whisper")
metadata_path = os.path.join(novice_dir, "asr.jsonl")

# paths for converting datasets to manifest files
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
val_dir = os.path.join(data_dir, "val")

# Define your ASR model
class ASRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_name = "distil-whisper/distil-medium.en"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name).half()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(self, input_ids, decoder_input_ids=None):
        return self.model(input_features=input_ids, decoder_input_ids=decoder_input_ids)
    
    def compute_wer(self, logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
        print(f'Predicted: {pred_str}')
        print(f'Actual: {label_str}')
        return wer(label_str, pred_str), pred_str, label_str

    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch
        outputs = self(input_ids, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits
        
        # Reshape logits to (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss)
        
        wer_value = self.compute_wer(logits, labels)[0]
        self.log('train_wer', wer_value, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch
        outputs = self(input_ids, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits
        
        # Reshape logits to (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        loss = self.loss_function(logits, labels)
        self.log('val_loss', loss)
        
        wer_value = self.compute_wer(logits, labels)[0]
        self.log('val_wer', wer_value, prog_bar=True)
        
        return loss
    
    def test_step(self, batch):
        input_ids, decoder_input_ids, labels = batch
        self.test_results = []
        with torch.no_grad():
            outputs = self(input_ids, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            
            # Reshape logits to (batch_size * sequence_length, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            test_loss = self.loss_function(logits, labels)
            self.log('test_loss', test_loss)

            wer_value, pred_str, label_str = self.compute_wer(logits, labels)

            # Store results
            for pred, actual in zip(pred_str, label_str):
                self.test_results.append({'predicted': pred, 'actual': actual})
            
            wer_value = self.compute_wer(logits, labels)[0]
            self.log('test_wer', wer_value, prog_bar=True)
            
            return test_loss

    def configure_optimizers(self):
        # Implement your optimizer configuration here
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    
class ASRIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer):
        self.type_dir, self.num_batches = data
        self.tokenizer = tokenizer

    def __iter__(self):
        device = torch.device("cuda")  # Define the device as GPU
        for batch_idx in range(self.num_batches):
            batch_output_dir = os.path.join(self.type_dir, f"batch_{batch_idx}")

            # Load input_ids
            input_ids_path = os.path.join(batch_output_dir, "input_ids.npy")
            input_ids_arr = np.load(input_ids_path)

            # Load decoder_input_ids
            decoded_input_ids_path = os.path.join(batch_output_dir, "decoder_input_ids.npy")
            decoded_input_ids_arr = np.load(decoded_input_ids_path)

            # Load labels
            labels_path = os.path.join(batch_output_dir, "labels.npy")
            labels_arr = np.load(labels_path)

            # Convert to tensors, adjust data types, and move to GPU
            input_ids = torch.tensor(input_ids_arr, dtype=torch.float16).to(device)
            decoder_input_ids = torch.tensor(decoded_input_ids_arr, dtype=torch.long).to(device)
            labels = torch.tensor(labels_arr, dtype=torch.long).to(device)

            yield input_ids, decoder_input_ids, labels

class ASRDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_data, val_data, test_data, num_workers=0):
        super().__init__()
        self.tokenizer = tokenizer # can just use the global one?
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ASRIterableDataset(self.train_data, self.tokenizer)
        self.val_dataset = ASRIterableDataset(self.val_data, self.tokenizer,)
        self.test_dataset = ASRIterableDataset(self.test_data, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None, num_workers=self.num_workers)
    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        patience=3,          # no of epochs with no improvement to wait before stopping
        verbose=True,        # logging
        mode='min'           # minimize or maximize the monitored metric
    )

    # Initialize Trainer with model checkpointing
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_checkpoints',
        filename='asr_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        max_steps=700*100,  # Maximum number of steps (batches) to train for
        callbacks=[checkpoint_callback, early_stopping_callback],
        val_check_interval=700,
        limit_val_batches=88,  # Limit the number of validation batches
    )

    torch.set_float32_matmul_precision('medium')

    model_path = "../models/whisper"  # Path where the model and processor are saved
    # Load the model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_path)

    data_module = ASRDataModule(
        tokenizer=processor.tokenizer,
        train_data=(train_dir, 700),
        val_data=(val_dir, 88),
        test_data=(test_dir, 88),
        num_workers=4,
        # batch_size=1, # Removed param as setting to 2 causes errors, probably due to IterableDataset? Perhaps need to manually handle using arrays in Dataset class and update collate function.
    )

    asr_model = ASRModel()
    asr_model.to('cuda')

    # Train the model
    trainer.fit(asr_model, data_module) # pl.LightningDataModule can be 2nd parameter

    # Test the model
    trainer.test(asr_model, data_module)