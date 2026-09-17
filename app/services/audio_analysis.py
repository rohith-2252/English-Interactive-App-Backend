
import librosa
import numpy as np
import re


FILLER_WORDS = {
    "um",
    "uh",
    "er",
    "ah",
    "like",
    "you know",
    "actually",
    "basically"
}


def analyze_audio(audio_path: str, transcript: str = "") -> dict:

    # Load audio
    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    # Duration
    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # -------------------------
    # Word count
    # -------------------------

    words = re.findall(
        r"\b[\w']+\b",
        transcript.lower()
    )

    word_count = len(words)

    # -------------------------
    # Words per minute
    # -------------------------

    if duration > 0:
        wpm = (word_count / duration) * 60
    else:
        wpm = 0

    # -------------------------
    # Silence detection
    # -------------------------

    intervals = librosa.effects.split(
        y,
        top_db=30
    )

    speech_samples = sum(
        end - start
        for start, end in intervals
    )

    total_samples = len(y)

    if total_samples > 0:
        silence_ratio = 1 - (
            speech_samples / total_samples
        )
    else:
        silence_ratio = 0

    # -------------------------
    # Pause count
    # -------------------------

    pause_count = max(
        0,
        len(intervals) - 1
    )

    # -------------------------
    # Filler words
    # -------------------------

    filler_count = 0

    transcript_lower = transcript.lower()

    for filler in FILLER_WORDS:

        if " " in filler:

            filler_count += transcript_lower.count(
                filler
            )

        else:

            filler_count += len(
                re.findall(
                    rf"\b{re.escape(filler)}\b",
                    transcript_lower
                )
            )

    return {
        "duration_seconds": round(
            duration,
            2
        ),

        "word_count": word_count,

        "words_per_minute": round(
            wpm,
            2
        ),

        "pause_count": pause_count,

        "silence_ratio": round(
            silence_ratio,
            3
        ),

        "filler_word_count": filler_count
    }
