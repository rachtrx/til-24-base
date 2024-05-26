import base64
import torch 
import whisper
import torchaudio
import numpy as np
from jiwer import wer
import io
import pytorch_lightning as pl
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import io
import soundfile as sf
import whisper

# You can change this to any model you want to use
    
class ASRModel(pl.LightningModule):
    def __init__(self, model_name="openai/whisper-medium"):
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def forward(self, input_values):
        return self.model.generate(input_values)

class ASRManager:
    def __init__(self, model_name="openai/whisper-medium"):
        self.model = ASRModel(model_name=model_name)
        self.model.eval()
        self.model.to('cuda')  # Ensure the model is on GPU

    def load_audio(self, audio_bytes: bytes):
        audio_buffer = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(audio_buffer, dtype='float32')

        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if necessary (Whisper models expect 16 kHz)
        if sample_rate != 16000:
            data = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(torch.tensor(data)).numpy()

        # Process audio to model's expected format using processor
        audio_input = self.model.processor(data, sampling_rate=16000, return_tensors="pt").input_values
        return audio_input

    def transcribe(self, audio_bytes: bytes) -> str:
        # Load and process the audio
        input_values = self.load_audio(audio_bytes)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model(input_values)
        transcription = self.model.processor.batch_decode(predicted_ids)[0]
        return transcription