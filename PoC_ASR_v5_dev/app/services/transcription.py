"""
Transcription service using faster-whisper.
Supports multiple Vietnamese Whisper models with caching.
"""
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from typing import Tuple
import re
import librosa

import numpy as np
from faster_whisper import WhisperModel


from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Available Whisper models for Vietnamese
AVAILABLE_MODELS = {
    "EraX-WoW-Turbo": "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
    "PhoWhisper Large": "kiendt/PhoWhisper-large-ct2",
    "PhoWhisper Lora Finetuned": "vyluong/pho-whisper-vi-ct2"
    
}

@dataclass
class WordTimestamp:
    """A single word with precise timestamp."""
    word: str
    start: float
    end: float
    speaker: Optional[str] = None

class TranscriptionService:
    """
    Service for speech-to-text transcription using faster-whisper.
    Supports multiple models with caching.
    """
    
    _models: Dict[str, WhisperModel] = {}
    
    @classmethod
    def get_model(cls, model_name: str = None) -> WhisperModel:
        """
        Get or load a Whisper model (lazy loading with caching).
        
        Args:
            model_name: Name of the model from AVAILABLE_MODELS
            
        Returns:
            Loaded WhisperModel instance
        """
        
        if model_name is None:
            model_name = settings.default_whisper_model
            
        cache_key = f"{model_name}_{settings.resolved_compute_type}"
        
        if cache_key in cls._models:
            return cls._models[cache_key]
        
        # Get model path
        if model_name in AVAILABLE_MODELS:
            model_path = AVAILABLE_MODELS[model_name]
        else:
            # Fallback to first available model
            model_name = list(AVAILABLE_MODELS.keys())[0]
            model_path = AVAILABLE_MODELS[model_name]
        
        logger.info(f"Loading Whisper model: {model_name} ({model_path})")
        logger.debug(f"Device: {settings.resolved_device}, Compute type: {settings.resolved_compute_type}")
        
        model = WhisperModel(
            model_path,
            device=settings.resolved_device,
            compute_type=settings.resolved_compute_type,
        )
        
        cls._models[cache_key] = model
        logger.info(f"Whisper model loaded: {model_name}")
        
        return model
    
    @classmethod
    def is_loaded(cls, model_name: str = None) -> bool:
        """Check if a model is loaded."""
        if model_name is None:
            model_name = settings.default_whisper_model

        cache_key = f"{model_name}_{settings.resolved_compute_type}"
        return cache_key in cls._models
    
    @classmethod
    def preload_model(cls, model_name: str = None) -> None:
        """Preload a model during startup."""
        if model_name is None:
            model_name = settings.default_whisper_model
        try:
            cls.get_model(model_name)
        except Exception as e:
            logger.error(f"Failed to preload Whisper model: {e}")
            raise
    

    @classmethod
    def transcribe_with_words(
        cls,
        audio_array: np.ndarray,
        model_name: str = None,
        language: str = "vi",
        vad_options: Optional[dict | bool] = None,
        beam_size: int = 3,
        temperature: float = 0.0,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,

        # Prompting
        initial_prompt: str = "Hội thoại tổng đài. Chỉ ghi lại đúng lời nói trong audio.",

        prefix_text: Optional[str] = None,
        
        # Stability / filtering
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.70,
        log_prob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.4
        
    ) -> Dict:
        """
        Transcribe audio and return word-level timestamps.
        """
        model = cls.get_model(model_name)
        
        if vad_options is None or vad_options is False:
            use_vad = False
            vad_parameters = None

        elif vad_options is True:
            use_vad = True
            vad_parameters = {
                "threshold": settings.vad_threshold,
                "min_speech_duration_ms": settings.vad_min_speech_duration_ms,
                "min_silence_duration_ms": settings.vad_min_silence_duration_ms,
            }

        elif isinstance(vad_options, dict):
            use_vad = True
            vad_parameters = vad_options

        else:
            use_vad = False
            vad_parameters = None

        
        prompt = (
            initial_prompt.strip()
            if isinstance(initial_prompt, str) and initial_prompt.strip()
            else None
        )

        prefix = (
            prefix_text.strip()
            if isinstance(prefix_text, str) and prefix_text.strip()
            else None
        )

        segments_gen, info = model.transcribe(
            audio_array,
            language=language if language != "auto" else None,
            
            # decoding
            beam_size=beam_size,
            temperature=temperature,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            
            # prompting
            prefix=prefix,

            # QA / Stability
            condition_on_previous_text=condition_on_previous_text,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,

            word_timestamps=True,

            # VAD
            vad_filter=use_vad,
            vad_parameters=vad_parameters,
            initial_prompt=prompt,
        )

        words = []
        full_text = []

        for seg in segments_gen:
            if seg.text:
                full_text.append(seg.text.strip())

            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    if not w.word.strip():
                        continue
                    words.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                    })

        return {
            "text": " ".join(full_text).strip(),
            "words": words,
            "info": info,
        }

    
    @classmethod
    async def transcribe_with_words_async(
        cls,
        audio_array: np.ndarray,
        model_name: str = None,
        language: str = "vi",
        vad_options: Optional[dict | bool] = None,
        beam_size: int = 5,
        temperature: float = 0.0,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        initial_prompt: Optional[str] = None,
        prefix_text: Optional[str] = None,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.70,
        log_prob_threshold: float = -1.0,
        # text repetitive / nonsense
        compression_ratio_threshold: float = 2.4
    ) -> Dict:
        """
        Async wrapper for transcription (runs in thread pool).
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.transcribe_with_words(
                audio_array=audio_array,
                model_name=model_name,
                language=language,
                vad_options=vad_options,
                beam_size=beam_size,
                temperature=temperature,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                initial_prompt=initial_prompt,
                prefix_text=prefix_text,
                condition_on_previous_text=condition_on_previous_text,
                no_speech_threshold=no_speech_threshold,
                log_prob_threshold=log_prob_threshold,
                compression_ratio_threshold=compression_ratio_threshold

            )
        )

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        
        
        """Return list of available models."""
        return AVAILABLE_MODELS.copy()
    

