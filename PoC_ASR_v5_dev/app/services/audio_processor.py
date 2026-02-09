"""
Audio processing utilities.
Simple validation and file handling.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from app.core.config import get_settings
import ffmpeg
import asyncio

from app.services.vocal_separator import VocalSeparator
from app.services.denoiser import DenoiserService

logger = logging.getLogger(__name__)
settings = get_settings()


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


class AudioProcessor:
    ALLOWED_EXTENSIONS = settings.allowed_extensions
    TARGET_SAMPLE_RATE = settings.sample_rate
    TARGET_CHANNELS = settings.channels
    
    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> None:
        """
        Validate uploaded file.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            
        Raises:
            AudioProcessingError: If validation fails
        """
        # Check extension
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in settings.allowed_extensions:
            raise AudioProcessingError(
                f"File type '.{ext}' not supported. "
                f"Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Check size
        if file_size > settings.max_upload_size_bytes:
            raise AudioProcessingError(
                f"File too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum size: {settings.max_upload_size_mb}MB"
            )
    
    @classmethod
    async def save_upload(cls, file_content: bytes, original_filename: str) -> Path:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: Raw file bytes
            original_filename: Original filename for extension
            
        Returns:
            Path to saved file
        """
        import aiofiles
        
        # Generate unique filename
        ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else 'wav'
        unique_filename = f"{uuid.uuid4()}.{ext}"
        file_path = settings.upload_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"Saved upload: {file_path} ({len(file_content) / 1024:.1f}KB)")
        return file_path
    
    @classmethod
    async def convert_to_wav(cls, input_path: Path) -> Path:
        """
        Convert audio to 16kHz mono WAV using FFmpeg.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to converted WAV file
        """
        output_filename = f"{input_path.stem}_processed.wav"
        output_path = settings.processed_dir / output_filename
        
        try:
            # Run ffmpeg conversion in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: cls._run_ffmpeg_conversion(input_path, output_path))
            
            logger.info(f"Converted to WAV: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise AudioProcessingError(f"Audio conversion failed: {error_msg}")
    
    @staticmethod
    def _run_ffmpeg_conversion(input_path: Path, output_path: Path) -> None:
        """Run the actual FFmpeg conversion (blocking)."""
        stream = ffmpeg.input(str(input_path))
        
        # Apply normalization if enabled (loudnorm is best for speech consistency)
        if settings.enable_loudnorm:
            logger.debug("Applying loudnorm normalization...")
            stream = stream.filter('loudnorm', I=-20, TP=-2, LRA=7)
            
        # Apply noise reduction if enabled (Note: basic filters are kept as minor cleanup)
        if settings.enable_noise_reduction:
            logger.debug("Applying subtle highpass filter...")
            stream = (
                stream
                .filter('highpass', f=60)
                .filter('lowpass', f=7500)
                .filter(
                    #  Silence trimming
                    'silenceremove',
                    stop_periods=-1,
                    stop_duration=0.4,
                    stop_threshold='-45dB'
                )
            )

            (
                stream.output(
                    str(output_path),
                    acodec='pcm_s16le',
                    ar=16000,
                    ac=1
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
        
    @classmethod
    async def get_audio_duration(cls, filepath: Path) -> float:
        """
        Get audio file duration in seconds.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            loop = asyncio.get_event_loop()
            probe = await loop.run_in_executor(
                None, 
                lambda: ffmpeg.probe(str(filepath))
            )
            
            duration = float(probe['format'].get('duration', 0))
            return duration
            
        except ffmpeg.Error as e:
            logger.warning(f"Could not probe audio duration: {e}")
            return 0.0
    @classmethod
    async def cleanup_files(cls, *paths: Path) -> None:
        """Remove temporary files."""
        import asyncio
        
        for path in paths:
            try:
                if path and path.exists():
                    path.unlink()
                    logger.debug(f"Cleaned up: {path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {path}: {e}")



    @classmethod
    async def process_upload(cls, file_content: bytes, filename: str) -> Tuple[Path, float]:
        """
        Full upload processing pipeline: validate, save, convert.
        
        Args:
            file_content: Uploaded file bytes
            filename: Original filename
            
        Returns:
            Tuple of (processed WAV path, duration in seconds)
        """
        # Validate
        cls.validate_file(filename, len(file_content))
        
        # Save original
        original_path = await cls.save_upload(file_content, filename)
        vocals_path = None
        
        try:
            # Step 1: Denoising (Speech Enhancement)
            if settings.enable_denoiser:
                denoised_path = await DenoiserService.enhance_audio(original_path)
                source_for_separation = denoised_path
            else:
                source_for_separation = original_path
                denoised_path = None
                
            # Step 2: Vocal separation using MDX-Net
            if settings.enable_vocal_separation:
                vocals_path = await VocalSeparator.separate_vocals(source_for_separation)
                source_for_conversion = vocals_path
            else:
                source_for_conversion = source_for_separation
                vocals_path = None
            
            # Step 3: Convert to 16kHz mono WAV (includes normalization)
            wav_path = await cls.convert_to_wav(source_for_conversion)
            
            # Get duration
            duration = await cls.get_audio_duration(wav_path)
            
            # Cleanup intermediate files
            to_cleanup = [original_path]
            if denoised_path and denoised_path != original_path:
                to_cleanup.append(denoised_path)
            if vocals_path and vocals_path not in [original_path, denoised_path]:
                to_cleanup.append(vocals_path)
                
            await cls.cleanup_files(*to_cleanup)
            
            return wav_path, duration
            
        except Exception as e:
            # Cleanup on error
            await cls.cleanup_files(original_path)
            if 'denoised_path' in locals() and denoised_path and denoised_path != original_path:
                await cls.cleanup_files(denoised_path)
            if 'vocals_path' in locals() and vocals_path and vocals_path not in [original_path, denoised_path]:
                await cls.cleanup_files(vocals_path)
            raise