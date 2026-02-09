"""
Speech Enhancement Service using Facebook's Denoiser.
Removes background noise and enhances speech quality.
"""
import os
import asyncio
import logging
from pathlib import Path

import torch
import torchaudio

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DenoiserError(Exception):
    """Custom exception for denoiser errors."""
    pass


class DenoiserService:
    """
    Service for enhancing speech using Facebook's Denoiser models.
    Supports dns48, dns64, master64, etc.
    """
    
    _model = None
    _model_name: str = None
    
    @classmethod
    def _get_model(cls):
        """Lazy load the Denoiser model."""
        if cls._model is None or cls._model_name != settings.denoiser_model:
            from denoiser.pretrained import dns48, dns64, master64
            
            model_map = {
                "dns48": dns48,
                "dns64": dns64,
                "master64": master64
            }
            
            model_func = model_map.get(settings.denoiser_model, dns64)
            logger.debug(f"Loading Denoiser model: {settings.denoiser_model}")
            
            model = model_func()
            device = settings.resolved_device
            model.to(device)
            model.eval()
            
            cls._model = model
            cls._model_name = settings.denoiser_model
            logger.debug(f"Denoiser model loaded on {device}")
            
        return cls._model

    @classmethod
    async def enhance_audio(cls, input_path: Path) -> Path:
        """
        Enhance audio by removing noise.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to enhanced WAV file
        """
        if not settings.enable_denoiser:
            logger.debug("Denoiser disabled, skipping...")
            return input_path
            
        logger.debug(f"Starting speech enhancement for: {input_path.name}")
        
        try:
            # Run enhancement in executor to not block
            loop = asyncio.get_event_loop()
            enhanced_path = await loop.run_in_executor(
                None, 
                lambda: cls._run_enhancement(input_path)
            )
            
            logger.info(f"Speech enhancement complete: {enhanced_path.name}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Speech enhancement failed: {e}")
            # Fallback to original on failure rather than failing the whole pipeline
            logger.warning("Falling back to original audio.")
            return input_path

    @classmethod
    def _run_enhancement(cls, input_path: Path) -> Path:
        """Run the actual denoiser enhancement (blocking)."""
        from denoiser.enhance import enhance
        
        model = cls._get_model()
        device = settings.resolved_device
        
        # Load audio
        wav, sr = torchaudio.load(str(input_path))
        wav = wav.to(device)
        
        # Ensure correct sample rate for the model
        if sr != model.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, model.sample_rate).to(device)
            wav = resampler(wav)
            sr = model.sample_rate
            
        # Enhance
        # wav shape: [channels, time]
        from types import SimpleNamespace
        
        args = SimpleNamespace(
            streaming=False,
            dry=0.0,
            sample_rate=sr
        )
        
        with torch.no_grad():
            # denoiser.enhance.enhance(args, model, wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)
            elif wav.dim() == 2:
                wav = wav.unsqueeze(0)
                
            enhanced = enhance(args, model, wav)
            # remove batch dim
            enhanced = enhanced.squeeze(0)
            
        # Save enhanced audio
        output_filename = f"{input_path.stem}_denoised.wav"
        output_path = settings.processed_dir / output_filename
        
        torchaudio.save(
            str(output_path),
            enhanced.cpu(),
            sr
        )
        
        return output_path
