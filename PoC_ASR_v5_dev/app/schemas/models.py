"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of the transcription process."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptSegment(BaseModel):
    start: float
    end: float

    speaker: Optional[str] = Field(
        default=None,
        description="Internal speaker id (debug only)"
    )

    role: str = Field(
        ...,
        description="Conversation role (NV = agent, KH = customer)"
    )

    text: str = Field(
        ...,
        description="Transcribed text"
    )

    
    @property
    def start_formatted(self) -> str:
        """Format start time as HH:MM:SS."""
        return self._format_time(self.start)
    
    @property
    def end_formatted(self) -> str:
        """Format end time as HH:MM:SS."""
        return self._format_time(self.end)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TranscriptionRequest(BaseModel):
    """Request model for transcription settings."""
    language: str = Field(default="vi", description="Language code for transcription")
    num_speakers: Optional[int] = Field(default=None, description="Expected number of speakers (None for auto-detect)")
    output_format: str = Field(default="json", description="Output format: json, txt, srt")


class TranscriptionResponse(BaseModel):
    """Response containing the transcription results."""
    success: bool = Field(..., description="Whether transcription succeeded")
    message: str = Field(default="", description="Status message")
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Transcript segments with speaker and role")

    duration: float = Field(default=0.0, description="Audio duration in seconds")
    speaker_count: int = Field(default=0, description="Number of detected speakers")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    speakers: Optional[list[str]] = None
    
    roles: Optional[dict[str, str]] = Field(
        default=None,
        description="Internal mapping speaker_id → role (debug / audit only)"
    )

    download_txt: Optional[str] = Field(default=None, description="Download URL for TXT file")
    download_csv: Optional[str] = Field(default=None, description="Download URL for CSV file")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    models_loaded: bool = False
    device: str = "cpu"
