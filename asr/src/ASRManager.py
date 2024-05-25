import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import whisper

class ASRManager:
    def __init__(self, model_path):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.to('cuda')  # Move model to GPU

    def transcribe(self, audio_bytes: bytes) -> str:
        waveform, sample_rate = self.load_audio(audio_bytes)
        input_features = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_features
        input_features = input_features.to('cuda')
        
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    def load_audio(self, audio_bytes: bytes):
        waveform, sample_rate = torchaudio.load(audio_bytes)
        waveform = waveform.numpy().flatten()

        # Apply augmentations
        waveform = self.augmentations(samples=waveform, sample_rate=sample_rate)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(torch.tensor(waveform)).numpy().flatten()
            
        # Compute log-mel spectrogram
        input_ids = self.to_pad_to_mel(waveform)

        return input_ids, 16000
    
    @staticmethod
    def to_pad_to_mel(array):
        """Static function to pad/trim and compute log-mel spectrogram."""
        padded_input = whisper.pad_or_trim(np.asarray(array, dtype=np.float32))
        input_ids = whisper.log_mel_spectrogram(padded_input)
        return input_ids