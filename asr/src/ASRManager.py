import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import whisper
import numpy as np
from jiwer import wer

  # You can change this to any model you want to use
save_directory = "../models/whisper"  # Path to save the model and processor

class ASRManager:
    def __init__(self, model_path="./models/asr_model-epoch=04-val_loss=0.61.ckpt"):
        self.model_name = "distil-whisper/distil-medium.en"
        # self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model = ASRModel.load_from_checkpoint(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to('cuda')  # Move model to GPU

    def load_audio(self, audio_bytes: bytes):
        # Load audio file from bytes, resample if necessary
        waveform, sample_rate = torchaudio.load(audio_bytes)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        return waveform, 16000

    def transcribe(self, audio_bytes: bytes) -> str:
        # Load and process the audio
        waveform, sample_rate = self.load_audio(audio_bytes)
        # Convert audio to features
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values
        # Move inputs to GPU
        inputs = inputs.to('cuda')
        # Generate transcription
        with torch.no_grad():
            logits = self.model.generate(inputs)
        transcription = self.processor.batch_decode(logits, skip_special_tokens=True)
        return transcription[0]
    
class ASRModel(pl.LightningModule):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(self, input_ids, decoder_input_ids=None):
        return self.model(input_features=input_ids, decoder_input_ids=decoder_input_ids)
    
    def compute_wer(self, logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
        return wer(label_str, pred_str), pred_str, label_str
    
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