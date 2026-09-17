
from faster_whisper import WhisperModel

print("Loading Whisper model...")

# Good balance for your CPU-based laptop
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded successfully.")


def transcribe_audio(audio_path: str) -> str:
    """
    Convert an audio file into text using Faster-Whisper.
    """

    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True
    )

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return transcript.strip()
