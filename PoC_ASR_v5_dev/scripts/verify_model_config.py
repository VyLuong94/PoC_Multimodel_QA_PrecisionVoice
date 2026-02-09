import os
from app.core.config import get_settings
from app.services.transcription import TranscriptionService

def verify_stt_model():
    settings = get_settings()
    print(f"Current Whisper Model: {settings.whisper_model}")
    print(f"Device: {settings.resolved_device}")
    print(f"Compute Type: {settings.resolved_compute_type}")
    
    expected_model = "kiendt/PhoWhisper-large-ct2"
    if settings.whisper_model == expected_model:
        print("✅ SUCCESS: Model configuration updated correctly.")
    else:
        print(f"❌ FAILURE: Expected {expected_model}, got {settings.whisper_model}")

if __name__ == "__main__":
    verify_stt_model()
