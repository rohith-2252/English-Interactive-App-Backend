from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.speech_to_text import transcribe_audio
from app.services.audio_analysis import analyze_audio
from app.services.text_analysis import analyze_text
from app.services.scoring import calculate_score
from app.services.feedback import generate_feedback

import os
import uuid

app = FastAPI(
    title="AI Speech Analyzer",
    version="1.0.0"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "AI Speech Analyzer Backend",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_audio_file(audio: UploadFile = File(...)):

    try:
        extension = os.path.splitext(audio.filename)[1].lower()

        if extension not in [".wav", ".mp3", ".m4a", ".ogg", ".webm"]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format"
            )

        filename = f"{uuid.uuid4()}{extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        contents = await audio.read()

        with open(file_path, "wb") as file:
            file.write(contents)

        print(f"Audio saved: {file_path}")

        # Speech → Text
        transcript = transcribe_audio(file_path)
        print(f"Transcript: {transcript}")

        audio_features = analyze_audio(file_path, transcript)
        print(f"Audio features: {audio_features}")

        text_features = analyze_text(transcript)

        scores = calculate_score(
            audio_features,
            text_features
        )
        feedback = generate_feedback(
            audio_features,
            text_features,
            scores
        )

        print(f"Feedback: {feedback}")

        print(f"Scores: {scores}")
        print(f"Text features: {text_features}")

        print(f"Transcript: {transcript}")

        # Audio analysis
        features = analyze_audio(
                    file_path,
                    transcript
                )

        print(f"Audio features: {features}")
        
        return {
            "success": True,
            "transcript": transcript,
            "audio_features": audio_features,
            "text_features": text_features,
            "scores": scores,
            "feedback": feedback
        }

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )