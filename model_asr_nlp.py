from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

print(f"📦 Đang load ASR model lên {'GPU' if device == 0 else 'CPU'}...")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="tuan8p/whisper-small-vi",
    device=device
)

print("✅ ASR model đã sẵn sàng.")