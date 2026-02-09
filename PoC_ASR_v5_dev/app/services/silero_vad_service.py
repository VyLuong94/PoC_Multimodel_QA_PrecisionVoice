import numpy as np
import librosa
from typing import List, Tuple
from silero_vad import load_silero_vad, get_speech_timestamps


class SileroVADService:

    _model = None

    @classmethod
    def load_model(cls):

        if cls._model is None:
            cls._model = load_silero_vad()

        return cls._model

    @classmethod
    def get_speech_timestamps(
        cls,
        audio: np.ndarray,
        sr: int
    ) -> List[Tuple[float, float]]:

        model = cls.load_model()

        audio = audio.astype(np.float32)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        speech = get_speech_timestamps(
            audio,
            model,
            sampling_rate=sr
        )

        # convert 
        segments = [
            (seg["start"] / sr, seg["end"] / sr)
            for seg in speech
            if seg["end"] > seg["start"]
        ]

        MIN_SPEECH_SEC = 0.25
        segments = [
            (s, e) for s, e in segments
            if (e - s) >= MIN_SPEECH_SEC
        ]

        # merge close 
        MERGE_GAP = 0.15
        merged = []

        for s, e in segments:
            if not merged:
                merged.append([s, e])
                continue

            prev = merged[-1]

            if s - prev[1] < MERGE_GAP:
                prev[1] = e
            else:
                merged.append([s, e])

        return [(s, e) for s, e in merged]
