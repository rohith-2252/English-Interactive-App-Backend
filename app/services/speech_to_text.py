from faster_whisper import WhisperModel


print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded successfully.")


def transcribe_audio(audio_path):

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        vad_filter=True
    )

    transcript = []

    for segment in segments:
        transcript.append(segment.text.strip())

    return " ".join(transcript)