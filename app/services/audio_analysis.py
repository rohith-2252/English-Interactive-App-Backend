import librosa
import numpy as np
import re
from pydub import AudioSegment


FILLER_WORDS = {
    "um", "uh", "erm", "ah",
    "like", "you know",
    "actually", "basically"
}


def analyze_audio(audio_path, transcript=""):

    # Convert M4A to WAV
    if audio_path.lower().endswith(".m4a"):

        wav_path = audio_path.rsplit(".", 1)[0] + ".wav"

        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")

        audio_path = wav_path

    # Load WAV
    y, sr = librosa.load(
        audio_path,
        sr=None
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    rms = librosa.feature.rms(y=y)[0]

    threshold = np.mean(rms) * 0.3

    silent_frames = rms < threshold

    silence_ratio = np.mean(silent_frames)

    words = re.findall(
        r"\b[\w']+\b",
        transcript.lower()
    )

    word_count = len(words)

    wpm = (
        (word_count / duration) * 60
        if duration > 0
        else 0
    )

    filler_count = sum(
        1 for word in words
        if word in FILLER_WORDS
    )

    silence_changes = np.diff(
        silent_frames.astype(int)
    )

    pause_count = np.sum(
        silence_changes == 1
    )

    return {
        "duration_seconds": round(float(duration), 2),
        "word_count": word_count,
        "words_per_minute": round(float(wpm), 2),
        "pause_count": int(pause_count),
        "silence_ratio": round(float(silence_ratio), 3),
        "filler_word_count": filler_count
    }