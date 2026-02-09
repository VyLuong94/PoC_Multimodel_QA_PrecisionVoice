"""
Precision alignment service - Word-center-based speaker assignment.
Merges word-level transcription with speaker diarization using precise timestamps.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.core.config import get_settings
from app.services.transcription import WordTimestamp
from app.services.diarization import SpeakerSegment
from app.schemas.models import TranscriptSegment



logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class WordWithSpeaker:
    """A word with assigned speaker."""
    word: str
    start: float
    end: float
    speaker: str


class AlignmentService:
    """
    Precision alignment service.
    Uses word-center-based algorithm for accurate speaker-to-text mapping.
    """
    
    PAUSE_THRESHOLD = 0.45
    CENTER_TOL = 0.15 # s (150 ms)
    OVERLAP_TH = 0.12 # > x% segments
    DIA_MERGE_GAP = 0.25
    MAX_SEGMENT_DURATION = 7.5
    
    @staticmethod
    def get_word_center(word: WordTimestamp) -> float:
        """Calculate the center time of a word."""
        return (word.start + word.end) / 2
    
    
    @staticmethod
    def overlap_ratio(w_start, w_end, s_start, s_end):
        overlap = max(0.0, min(w_end, s_end) - max(w_start, s_start))
        dur = max(1e-6, w_end - w_start)
        return overlap / dur
    
    
    # Diarization merge
    @classmethod
    def merge_dia_segments(cls, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        if not segments:
            return []

        segments = sorted(segments, key=lambda s: s.start)
        merged = [segments[0]]

        for s in segments[1:]:
            p = merged[-1]
            if s.speaker == p.speaker and (s.start - p.end) <= cls.DIA_MERGE_GAP:
                p.end = s.end
            else:
                merged.append(s)

        return merged
    

    @classmethod
    def find_speaker_center(
        cls,
        time: float,
        speaker_segments: List[SpeakerSegment],
    ) -> Optional[str]:

        for seg in speaker_segments:
            if seg.start - cls.CENTER_TOL <= time <= seg.end + cls.CENTER_TOL:
                return seg.speaker
        return None

    @staticmethod
    def find_closest_speaker(time: float, speaker_segments: List[SpeakerSegment]) -> str:
        if not speaker_segments:
            return "Unknown"

        min_dist = float("inf")
        closest = "Unknown"

        for seg in speaker_segments:
            d = min(abs(time - seg.start), abs(time - seg.end))
            if d < min_dist:
                min_dist = d
                closest = seg.speaker

        return closest
    
    
    @classmethod
    def assign_speakers_to_words(
        cls,
        words: List[WordTimestamp],
        speaker_segments: List[SpeakerSegment],
    ) -> List[WordWithSpeaker]:

        words = [w for w in words if w.word and w.word.strip()]

        if not speaker_segments:
            logger.warning("No diarization, fallback single speaker")
            return [
                WordWithSpeaker(w.word, w.start, w.end, "Speaker 1")
                for w in words
            ]

        speaker_segments = cls.merge_dia_segments(speaker_segments)

        results = []

        for word in words:
            center = cls.get_word_center(word)

            # 1. CENTER
            speaker = cls.find_speaker_center(center, speaker_segments)

            if speaker is None:
                # 2. OVERLAP
                best_ratio = 0
                best_spk = None

                for seg in speaker_segments:
                    r = cls.overlap_ratio(word.start, word.end, seg.start, seg.end)
                    if r > best_ratio:
                        best_ratio = r
                        best_spk = seg.speaker

                if best_ratio >= cls.OVERLAP_TH:
                    speaker = best_spk
                else:
                    # 3. CLOSEST
                    speaker = cls.find_closest_speaker(center, speaker_segments)

            results.append(
                WordWithSpeaker(word.word, word.start, word.end, speaker)
            )

        return results
    
    @classmethod
    def reconstruct_segments(
        cls,
        words_with_speakers: List[WordWithSpeaker]
    ) -> List[TranscriptSegment]:
        """
        Step 3d: Reconstruct sentence segments from words.
        
        Groups consecutive words of the same speaker into segments.
        Creates new segment when:
        - Speaker changes
        - Pause > PAUSE_THRESHOLD between words
        
        Args:
            words_with_speakers: List of words with speaker assignments
            
        Returns:
            List of TranscriptSegment with complete sentences
        """
        if not words_with_speakers:
            return []
        
        segments = []
        
        # Start first segment
        current_speaker = words_with_speakers[0].speaker
        current_start = words_with_speakers[0].start
        current_end = words_with_speakers[0].end
        current_words = [words_with_speakers[0].word]
        
        for i in range(1, len(words_with_speakers)):
            word = words_with_speakers[i]
            prev_word = words_with_speakers[i - 1]
            
            # Calculate pause between words
            pause = word.start - prev_word.end
            
            # Check if we need to start a new segment
            speaker_changed = word.speaker != current_speaker
            significant_pause = pause > cls.PAUSE_THRESHOLD
            
            segment_duration = current_end - current_start
            too_long = segment_duration > cls.MAX_SEGMENT_DURATION and pause > 0.15
            
            if speaker_changed or significant_pause or too_long:
                # Save current segment
                segments.append(TranscriptSegment(
                    start=current_start,
                    end=current_end,
                    speaker=current_speaker,
                    role="UNKNOWN", 
                    text=" ".join(current_words)
                ))
                
                # Start new segment
                current_speaker = word.speaker
                current_start = word.start
                current_end = word.end
                current_words = [word.word]
            else:
                # Continue current segment
                current_end = word.end
                current_words.append(word.word)
        

        if current_words:
            segments.append(TranscriptSegment(
                start=current_start,
                end=current_end,
                speaker=current_speaker,
                role="UNKNOWN", 
                text=" ".join(current_words)
            ))
        
        logger.debug(f"Reconstructed {len(segments)} segments from {len(words_with_speakers)} words")
        return segments
    
    @classmethod
    def resize_and_merge_segments(
        cls,
        segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """
        Merge consecutive segments of the same speaker if the gap is small.
        Also filters out extremely short segments.
        """
        if not segments:
            return []
            
        # Filter 1: Remove extremely short blips (noise)
        segments = [s for s in segments if (s.end - s.start) >= settings.min_segment_duration_s]
        
        if not segments:
            return []
            
        merged = []
        curr = segments[0]
        
        for i in range(1, len(segments)):
            next_seg = segments[i]
            
            # If same speaker and gap is small, merge
            gap = next_seg.start - curr.end
            if next_seg.speaker == curr.speaker and gap < settings.merge_threshold_s:
                curr.end = next_seg.end
                curr.text += " " + next_seg.text
            else:
                merged.append(curr)
                curr = next_seg
                
        merged.append(curr)
        
        logger.debug(f"Merged segments: {len(segments)} -> {len(merged)}")
        return merged

    @classmethod
    def align_precision(
        cls,
        words: List[WordTimestamp],
        speaker_segments: List[SpeakerSegment]
    ) -> List[TranscriptSegment]:
        """
        Full precision alignment pipeline.
        
        Args:
            words: Word-level timestamps from transcription
            speaker_segments: Speaker segments from diarization
            
        Returns:
            List of TranscriptSegment with proper speaker assignments
        """
        # Step 3c: Assign speakers to words
        words_with_speakers = cls.assign_speakers_to_words(words, speaker_segments)
        
        # Step 3d: Reconstruct segments
        segments = cls.reconstruct_segments(words_with_speakers)
        
        # Step 3e: Clustering/Merging (Optimization)
        segments = cls.resize_and_merge_segments(segments)
        
        return segments

 