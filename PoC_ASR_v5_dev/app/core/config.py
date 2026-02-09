"""
Application configuration using Pydantic Settings.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Literal, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # HuggingFace
    hf_token: str = ""
    enable_noise_reduction: bool = True
    
    # Denoising (Speech Enhancement)
    enable_denoiser: bool = True
    denoiser_model: str = "dns64"
    

    # MDX-Net Vocal Separation
    enable_vocal_separation: bool = True
    mdx_model: str = "Kim_Vocal_2.onnx"  # High quality vocal isolation
    
    available_whisper_models: Dict[str, str] = {
        "EraX-WoW-Turbo": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
        "PhoWhisper Large": "kiendt/PhoWhisper-large-ct2",
        "PhoWhisper Lora Finetuned": "vyluong/pho-whisper-vi-ct2"
    }
    
    default_whisper_model: str = "vyluong/pho-whisper-vi-ct2"
    

    # Diarization model
    diarization_model: str = "pyannote/speaker-diarization-community-1"

    # Device settings
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: str = "float16"  # float16 for GPU, int8 for CPU
    
    # Upload settings
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = ["mp3", "wav", "m4a", "ogg", "flac", "webm"]
    
    # Audio processing settings
    sample_rate: int = 16000
    channels: int = 1  # Mono
    

    enable_loudnorm: bool = True
    
    # VAD parameters
    vad_threshold: float = 0.55
    vad_min_speech_duration_ms: int = 200
    vad_min_silence_duration_ms: int = 450
    vad_speech_pad_ms: int = 250
    
        
    # Post-processing
    merge_threshold_s: float = 0.35  # Merge segments from same speaker if gap < this
    min_segment_duration_s: float = 0.85 # Remove segments shorter than this
    
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 7860
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    upload_dir: Path = data_dir / "uploads"
    processed_dir: Path = data_dir / "processed"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
    
    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device
    
    @property
    def resolved_compute_type(self) -> str:
        """Get appropriate compute type for device."""
        if self.resolved_device == "cuda":
            return "float16"
        return "int8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
