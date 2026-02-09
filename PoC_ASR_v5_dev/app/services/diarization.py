# The `DiarizationService` class provides a production-grade speaker diarization service for call
# centers, including role inference based on speaking duration and asynchronous diarization
# capabilities.
"""
Speaker diarization service using pyannote.audio.
QA / Production optimized diarization for call center.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

import torch
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# =========================
# Data model
# =========================
@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start

@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    speaker_count: int
    speakers: List[str]
    roles: Dict[str, str] 
    
# =========================
# Diarization Service
# =========================
class DiarizationService:
    """
    Production-grade speaker diarization service.
    """

    _instance: Optional["DiarizationService"] = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # -------------------------
    # Pipeline loading
    # -------------------------
    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            from pyannote.audio import Pipeline

            if not settings.hf_token:
                raise ValueError("HF_TOKEN is required for diarization")

            logger.info(
                f"Loading diarization model: {settings.diarization_model}"
            )

            pipeline = Pipeline.from_pretrained(
                settings.diarization_model,
                token=settings.hf_token
            )

            pipeline.instantiate({
                "clustering": {
                    "threshold": 0.65
                },
                "segmentation": {
                    "min_duration_off": 0.4  # avoid fragment explosion
                }
            })

            device = torch.device(settings.resolved_device)
            if device.type == "cuda":
                pipeline = pipeline.to(device)
                logger.info("Diarization pipeline moved to GPU")

            cls._pipeline = pipeline

        return cls._pipeline


    # -------------------------
    # Role inference (CALL CENTER)
    # -------------------------
    @staticmethod
    def infer_roles(segments: List[SpeakerSegment]) -> Dict[str, str]:
        """
        Infer Agent / Customer roles based on total speaking duration.
        Agent usually speaks the most.
        """
        duration_map: Dict[str, float] = {}

        for seg in segments:
            duration_map[seg.speaker] = (
                duration_map.get(seg.speaker, 0.0) + seg.duration
            )

        if not duration_map:
            return {}

        # Agent = speaker with max duration
        agent = max(duration_map, key=duration_map.get)

        roles = {}
        for speaker in duration_map:
            roles[speaker] = "NV" if speaker == agent else "KH"

        return roles
    # -------------------------
    # Main diarization
    # -------------------------
    @classmethod
    def diarize(
        cls,
        audio_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> DiarizationResult:

        pipeline = cls.get_pipeline()
        logger.debug(f"Diarizing file: {audio_path}")

        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        diarization = pipeline(str(audio_path), **params)

        annotation = (
            diarization.speaker_diarization
            if hasattr(diarization, "speaker_diarization")
            else diarization
        )

        # step 1: diarize
        raw_segments: List[SpeakerSegment] = []
        speaker_map = {}
        speaker_idx = 1

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {speaker_idx}"
                speaker_idx += 1

            raw_segments.append(
                SpeakerSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=speaker_map[speaker]
                )
            )

        raw_segments.sort(key=lambda s: s.start)
        unique_speakers = []
        for seg in raw_segments:
            if seg.speaker not in unique_speakers:
                unique_speakers.append(seg.speaker)

        roles = cls.infer_roles(raw_segments)

        logger.info(
            f"Diarization done | "
            f"Segments: {len(raw_segments)} | "
            f"Speakers: {len(unique_speakers)} | "
            f"Roles: {roles}"
        )
        
        return DiarizationResult(
            segments=raw_segments,
            speaker_count=len(unique_speakers),
            speakers=unique_speakers,
            roles=roles
        )

    # -------------------------
    # Async
    # -------------------------
    @classmethod
    async def diarize_async(
        cls,
        audio_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> DiarizationResult:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.diarize(
                audio_path,
                num_speakers,
                min_speakers,
                max_speakers
            )
        )

    @classmethod
    def preload_pipeline(cls) -> None:
        try:
            cls.get_pipeline()
        except Exception as e:
            logger.warning(
                f"Failed to preload diarization pipeline: {e}"
            )
