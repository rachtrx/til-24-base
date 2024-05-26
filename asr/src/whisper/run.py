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
import jiwer
import whisper

import torch
from torch.utils.data import IterableDataset, DataLoader
import torchaudio
from torchaudio import transforms
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from tqdm import tqdm
from torch.nn import CrossEntropyLoss

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

class ASRIterableDataset(IterableDataset):
    def __init__(self, data, model_name):
        self.type_dir, self.num_batches = data
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            batch_output_dir = os.path.join(self.type_dir, f"batch_{batch_idx}")
            input_ids_path = os.path.join(batch_output_dir, "input_ids.npy")
            input_ids_arr = np.load(input_ids_path)
            decoded_input_ids_path = os.path.join(batch_output_dir, "decoder_input_ids.npy")
            decoded_input_ids_arr = np.load(decoded_input_ids_path)
            labels_path = os.path.join(batch_output_dir, "labels.npy")
            labels_arr = np.load(labels_path)

            yield (torch.tensor(input_ids_arr, dtype=torch.float32),  # Change float16 to float32
                torch.tensor(decoded_input_ids_arr, dtype=torch.long),
                torch.tensor(labels_arr, dtype=torch.long))

class ASRModel(pl.LightningModule):
    def __init__(self, train_data, val_data, test_data, num_workers=0, model_name="distil-whisper/distil-medium.en", lr=1e-3, checkpoint_path=None):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_workers = num_workers
        self.loss_function = CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def setup(self, stage=None):
        self.train_dataset = ASRIterableDataset(self.train_data, self.model_name)
        self.val_dataset = ASRIterableDataset(self.val_data, self.model_name)
        self.test_dataset = ASRIterableDataset(self.test_data, self.model_name)

    def forward(self, input_ids, decoder_input_ids=None):
        return self.model(input_features=input_ids, decoder_input_ids=decoder_input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = map(lambda t: t.to(self.device), batch)

        outputs = self(input_ids, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits
        logits = logits.view(-1, logits.size(-1))

        labels = labels.view(-1)
        loss = self.loss_function(logits, labels)
        print(loss)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, 'test')

    def _shared_eval_step(self, batch, prefix):
        input_ids, decoder_input_ids, labels = map(lambda t: t.to(self.device), batch)
        outputs = self(input_ids, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.loss_function(logits, labels)
        self.log(f'{prefix}_loss', loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None, num_workers=self.num_workers)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn')

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        patience=3,          # no of epochs with no improvement to wait before stopping
        verbose=True,        # logging
        mode='min'           # minimize or maximize the monitored metric
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='model_checkpoints',
        filename='asr_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        precision=16,
        max_steps=700*100,  # Maximum number of steps (batches) to train for
        callbacks=[checkpoint_callback, early_stopping_callback],
        val_check_interval=700,
        limit_val_batches=88,  # Limit the number of validation batches
    )

    asr_model = ASRModel(
        train_data=(train_dir, 700),
        val_data=(val_dir, 88),
        test_data=(test_dir, 88),
        num_workers=2,
        # checkpoint_path='./models/asr_model-epoch=04-val_loss=0.61.ckpt'
    )
    # Train the model
    trainer.fit(asr_model)

    # Test the model
    trainer.test(asr_model)