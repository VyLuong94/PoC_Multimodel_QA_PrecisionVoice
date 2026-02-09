
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import librosa
import torch

from app.core.config import get_settings
from app.services.transcription import TranscriptionService
from app.services.alignment import AlignmentService
from app.services.transcription import WordTimestamp


from app.services.diarization import DiarizationService, SpeakerSegment, DiarizationResult

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TranscriptSegment:
    """A transcribed segment with speaker info."""
    start: float
    end: float
    speaker: str
    role: Optional[str]
    text: str


@dataclass
class ProcessingResult:
    """Result of audio processing."""
    segments: List[TranscriptSegment]
    speaker_count: int
    duration: float
    processing_time: float
    speakers: List[str]
    roles: Dict[str, str]

    txt_content: str = ""
    csv_content: str = ""


def pad_and_refine_tensor(
    waveform: torch.Tensor,
    sr: int,
    start_s: float,
    end_s: float,
    pad_ms: int = 250,
) -> Tuple[float, float]:

    total_len = waveform.shape[1]
    s = max(int((start_s - pad_ms / 1000) * sr), 0)
    e = min(int((end_s + pad_ms / 1000) * sr), total_len)

    if e <= s:
        return start_s, end_s

    return s / sr, e / sr


def normalize_asr_result(result: dict):

    words = []

    for w in result.get("words", []):

        word = w.get("word", "").strip()
        if not word:
            continue

        words.append(
            {
                "word": word,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "speaker": w.get("speaker"),
            }
        )

    text = result.get("text", "").strip()
    return text, words


def guess_speaker_by_overlap(start, end, diar_segments):

    best_spk = None
    best_overlap = 0.0

    for seg in diar_segments:

        overlap = max(0.0, min(end, seg.end) - max(start, seg.start))

        if overlap > best_overlap:
            best_overlap = overlap
            best_spk = seg.speaker

    return best_spk or diar_segments[0].speaker



def convert_audio_to_wav(audio_path: Path) -> Path:
    """Convert any audio to WAV 16kHz Mono using ffmpeg."""
    output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
    if output_path.exists():
        output_path.unlink()
    command = ["ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-y", str(output_path)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Converted audio to WAV: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return audio_path


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def merge_consecutive_segments(
    segments: List[SpeakerSegment], 
    max_gap: float = 0.8,
    min_duration: float = 0.15,
) -> List[SpeakerSegment]:
    """Merge consecutive segments from same speaker."""
    if not segments:
        return []
    
    merged = []
    current = SpeakerSegment(
        start=segments[0].start,
        end=segments[0].end,
        speaker=segments[0].speaker
    )
    
    for seg in segments[1:]:
        seg_dur = seg.end - seg.start
        if (seg.speaker == current.speaker and (seg.start - current.end) <= max_gap
            or seg_dur < min_duration):
            # Merge: extend current segment
                current.end = seg.end
        else:
            # New speaker or gap too large
            merged.append(current)
            current = SpeakerSegment(
                start=seg.start,
                end=seg.end,
                speaker=seg.speaker
            )
    
    merged.append(current)
    return merged


def overlap_prefix(a: str, b: str, n: int = 12) -> bool:
    if not a or not b:
        return False

    a = a.strip().lower()
    b = b.strip().lower()

    return a[:n] in b or b[:n] in a


class Processor:
    @classmethod
    async def process_audio(
        cls,
        audio_path: Path,
        model_name: str = "PhoWhisper Lora Finetuned",
        language="vi",
        merge_segments: bool = True,
           
    ) -> ProcessingResult:

        import asyncio

        t0= time.time()

        # 1: Convert to WAV
        logger.info("Step 1: Converting audio to WAV 16kHz...")
        wav_path = await asyncio.get_event_loop().run_in_executor(None, convert_audio_to_wav, audio_path)

        # 2: Load audio
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        waveform = torch.from_numpy(y).unsqueeze(0)
        if y.size == 0:
            raise ValueError("Empty audio")
        duration = len(y) / sr
        
        # 3: Diarization
        logger.info("Step 3: Running diarization...")
        
        diarization: DiarizationResult = await DiarizationService.diarize_async(wav_path)

        diarization_segments = diarization.segments or []
        speakers = diarization.speakers or []
        roles = diarization.roles or {}

        if not diarization_segments:
            diarization_segments = [SpeakerSegment(0.0, duration, "SPEAKER_0")]
            speakers = ["SPEAKER_0"]
            roles = {"SPEAKER_0": "KH"}

        diarization_segments.sort(key=lambda x: x.start)
        
        diarization_segments = [
            SpeakerSegment(
                *pad_and_refine_tensor(waveform, sr, s.start, s.end),
                speaker=s.speaker,
            )
            for s in diarization_segments
        ]
        
        diarization_segments.sort(key=lambda x: x.start)

        
        if merge_segments and diarization_segments:
            logger.info("Step 4: Merging consecutive segments...")
            diarization_segments = merge_consecutive_segments(diarization_segments)
    
        # 4. Normalize speakers
        raw_speakers = sorted({seg.speaker for seg in diarization_segments})

        speaker_map = {
            spk: f"Speaker {i+1}"
            for i, spk in enumerate(raw_speakers)
        }

        speakers = list(speaker_map.values())

        # 5. NORMALIZE ROLES
        speaker_duration = defaultdict(float)
        for seg in diarization_segments:
            speaker_duration[seg.speaker] += seg.end - seg.start

        logger.info(f"speaker_duration(raw) = {speaker_duration}")

        if speaker_duration:
            agent_raw = max(speaker_duration, key=speaker_duration.get)

            roles = {
                speaker_map[spk]: ("NV" if spk == agent_raw else "KH")
                for spk in speaker_duration
            }
        else:
            roles = {}

        # Default fallback
        for label in speakers:
            roles.setdefault(label, "KH")

        logger.info(f"roles(mapped) = {roles}")

        # 7: Transcribe segments after diarization
        logger.info("Step 7: Running ASR with external VAD batch...")

        asr_result = await TranscriptionService.transcribe_with_words_async(
            audio_array=y,
            model_name=model_name,
            language=language,
            vad_options=True
        )

        text, raw_words = normalize_asr_result(asr_result)

        processed_segments: List[TranscriptSegment] = []

        if not raw_words:
            processed_segments = [
                TranscriptSegment(
                    start=0.0,
                    end=duration,
                    speaker=speakers[0],
                    role=roles[speakers[0]],
                    text="(No speech detected)"
                )
            ]
            
        else:

            # ===== CONVERT TO WordTimestamp =====
            word_objs: List[WordTimestamp] = []

            for w in raw_words:

                spk = w.get("speaker")

                if spk is None:
                    spk = guess_speaker_by_overlap(
                        w["start"], w["end"], diarization_segments
                    )

                word_objs.append(
                    WordTimestamp(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        speaker=spk,
                    )
                )

            word_objs.sort(key=lambda x: x.start)
            
            # ===== ALIGNMENT =====
            aligned_segments = AlignmentService.align_precision(
                word_objs,
                diarization_segments
            )

            processed_segments = []

            if not aligned_segments:

                vote = [w.speaker for w in word_objs if w.speaker]

                if vote:
                    raw_spk = Counter(vote).most_common(1)[0][0]
                else:
                    raw_spk = diarization_segments[0].speaker

                label = speaker_map.get(raw_spk, "Speaker 1")

                processed_segments.append(
                    TranscriptSegment(0, duration, label, roles[label], text)
                )

            else:

                for seg in aligned_segments:

                    raw_spk = seg.speaker
                    label = speaker_map.get(raw_spk, "Speaker 1")
                    role = roles.get(label, "KH")

                    processed_segments.append(
                        TranscriptSegment(
                            start=seg.start,
                            end=seg.end,
                            speaker=label,
                            role=role,
                            text=seg.text,
                        )
                    )

        
        processed_segments = cls._merge_adjacent_segments(
            processed_segments
        )
        processed_segments.sort(key=lambda x: x.start)

        processing_time = time.time() - t0
        
        txt_content = cls._generate_txt(
            processed_segments,
            len(speakers),
            processing_time,
            duration,
            roles
        )

        csv_content = cls._generate_csv(processed_segments)

        return ProcessingResult(
            segments=processed_segments,
            speaker_count=len(speakers),
            duration=duration,
            processing_time=processing_time,
            speakers=speakers,
            roles=roles,
            txt_content=txt_content,
            csv_content=csv_content,
        )
    
    
    @staticmethod
    def _merge_adjacent_segments(
        segments: List[TranscriptSegment],
        max_gap_s: float = 0.8,
        max_segment_duration: float = 9.0
    ) -> List[TranscriptSegment]:
        """
        Merge adjacent segments if:
        - same speaker
        - gap <= max_gap_s
        """
        if not segments:
            return segments

        segments = sorted(segments, key=lambda s: s.start)
        merged = [segments[0]]

        for seg in segments[1:]:
            prev = merged[-1]

            gap = seg.start - prev.end
            combined_duration = seg.end - prev.start

            if (
                seg.speaker == prev.speaker  and seg.role == prev.role
                and gap <= max_gap_s
                and combined_duration <= max_segment_duration
                and not overlap_prefix(seg.text, prev.text)
            ):
                # MERGE
                prev.text = f"{prev.text} {seg.text}".strip()
                prev.end = max(prev.end, seg.end)
            else:
                merged.append(seg)

        return merged

        
    @classmethod
    def _generate_txt(
            cls,
            segments: List[TranscriptSegment],
            speaker_count: int,
            processing_time: float,
            duration: float,
            roles: Dict[str, str],
        ) -> str:

        segments = sorted(segments, key=lambda s: s.start)
        speakers = []
        for seg in segments:
            if seg.speaker and seg.speaker not in speakers:
                speakers.append(seg.speaker)
                
        lines = [
            "# Transcription Result",
            f"# Duration: {format_timestamp(duration)}",
            f"# Speakers: {speaker_count}",
            f"# Roles: {roles}",
            f"# Processing time: {processing_time:.1f}s",
            "",
        ]
        icon_pool = ["🔵", "🟢", "🟡", "🟠", "🔴", "🟣"]
        speaker_icons = {
            spk: icon_pool[i % len(icon_pool)]
            for i, spk in enumerate(speakers)
        }


        for seg in segments:
            ts = f"[{format_timestamp(seg.start)} → {format_timestamp(seg.end)}]"
            role = seg.role or "UNKNOWN"

            speaker_icon = speaker_icons.get(seg.speaker, "⚪")
            lines.append(
                f"{ts} {speaker_icon} [{seg.speaker}|{role}] {seg.text}"
            )

        return "\n".join(lines)

    @classmethod
    def _generate_csv(cls, segments: List[TranscriptSegment]) -> str:
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["start", "end", "speaker", "text"])
        for seg in segments:
            writer.writerow([round(seg.start, 3), round(seg.end, 3), seg.speaker, seg.text])
        return output.getvalue()
