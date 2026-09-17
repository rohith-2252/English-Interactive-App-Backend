import os
import uuid
import traceback
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
cors_origins_str = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

import app.db as db
from app.services.whisper_service import transcribe_audio
from app.services.audio_analysis import analyze_audio
from app.services.text_analysis import analyze_text
from app.services.ml_scoring import predict_score
from app.services.feedback_service import (
    calculate_scores,
    generate_feedback
)

# ==========================================
# FastAPI application
# ==========================================

app = FastAPI(
    title="Speech AI Backend",
    description="AI-powered English speech analysis API with SQLite persistence",
    version="1.1.0"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Upload directory & Static files
# ==========================================

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ==========================================
# Auth Helper
# ==========================================

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Extract user from Bearer token in Authorization header.
    Falls back to a persistent guest account if no token is provided.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        user = db.get_user_by_token(token)
        if user:
            return user
    return db.get_or_create_guest_user()


# ==========================================
# Root & Health Endpoints
# ==========================================

@app.get("/")
def home():
    return {
        "success": True,
        "message": "Speech AI Backend is running with SQLite database support"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==========================================
# Authentication Endpoints
# ==========================================

@app.post("/api/auth/register")
async def api_register(payload: dict = Body(...)):
    """
    Sign up a new user with email, password, name, and optional phone number.
    Persists credentials to SQLite with PBKDF2 HMAC-SHA256 password hashing.
    """
    email = payload.get("email")
    password = payload.get("password")
    name = payload.get("name")
    phone_no = payload.get("phone_no") or payload.get("phone_number") or payload.get("phone")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
    
    if len(password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters long.")

    try:
        user = db.create_user(
            email=email,
            password=password,
            name=name or "",
            phone_no=phone_no
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = db.create_session(user["id"])
    return {
        "access_token": token,
        "name": user["name"],
        "email": user["email"],
        "phone_no": user.get("phone_no"),
        "id": user["id"]
    }


@app.post("/api/auth/login")
async def api_login(payload: dict = Body(...)):
    """
    Authenticate user using email and password against SQLite database.
    """
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    user = db.authenticate_user(email, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = db.create_session(user["id"])
    return {
        "access_token": token,
        "name": user["name"],
        "email": user["email"],
        "phone_no": user.get("phone_no"),
        "id": user["id"]
    }


@app.get("/api/auth/me")
async def api_me(authorization: Optional[str] = Header(None)):
    """Get profile of currently logged-in user."""
    user = get_current_user(authorization)
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "phone_no": user.get("phone_no"),
        "created_at": user.get("created_at")
    }


# ==========================================
# Topics
# ==========================================

TOPICS = [
    {
        "id": "1",
        "topic": "Describe your favorite hobby",
        "prompt": "Talk about what you enjoy doing in your free time, why you like it, and how often you do it."
    },
    {
        "id": "2",
        "topic": "A memorable travel experience",
        "prompt": "Describe a trip you took that was memorable to you. Where did you go, who did you go with, and what made it special?"
    },
    {
        "id": "3",
        "topic": "The importance of learning English",
        "prompt": "Explain why learning English is important for your personal growth, career, and communication."
    },
    {
        "id": "4",
        "topic": "Your favorite book or movie",
        "prompt": "Talk about a book or movie that influenced you. What is the story about and why did you enjoy it?"
    }
]


@app.get("/api/topics/random")
async def get_random_topic():
    import random
    return random.choice(TOPICS)


# ==========================================
# Audio Streaming / Serving Endpoint
# ==========================================

@app.get("/api/practice/audio/{filename}")
async def get_practice_audio(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/wav" if filename.endswith(".wav") else "audio/m4a"
    return FileResponse(file_path, media_type=media_type)


# ==========================================
# Analyze Practice (Mobile & Web Integration)
# ==========================================

@app.post("/api/practice/analyze")
async def analyze_practice(
    topic: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    duration_ms: Optional[str] = Form(None),
    word_count: Optional[str] = Form(None),
    pause_count: Optional[str] = Form(None),
    total_pause_ms: Optional[str] = Form(None),
    longest_pause_ms: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    try:
        user = get_current_user(authorization)
        saved_path = None
        saved_filename = None
        extracted_transcript = transcript or ""

        if audio and audio.filename:
            ext = os.path.splitext(audio.filename)[1] or ".m4a"
            saved_filename = f"{uuid.uuid4()}{ext}"
            saved_path = os.path.join(UPLOAD_DIR, saved_filename)
            contents = await audio.read()
            with open(saved_path, "wb") as f:
                f.write(contents)

            # Whisper STT if audio provided
            try:
                whisper_text = transcribe_audio(saved_path)
                if whisper_text and len(whisper_text.strip()) > 0:
                    extracted_transcript = whisper_text
            except Exception as stt_err:
                print(f"Whisper STT fallback: {stt_err}")

        if not extracted_transcript and not saved_path:
            raise HTTPException(status_code=400, detail="Provide either audio or transcript.")

        # Audio analysis
        dur_secs = 0.0
        wpm_val = 0.0
        if saved_path:
            audio_feats = analyze_audio(saved_path, extracted_transcript)
            dur_secs = audio_feats.get("duration_seconds", 0.0)
            wpm_val = audio_feats.get("words_per_minute", 0.0)
        else:
            w_cnt = int(word_count) if word_count else len(extracted_transcript.split())
            dur = float(duration_ms) / 1000.0 if duration_ms else 30.0
            dur_secs = dur
            wpm_val = round((w_cnt / max(dur, 1.0)) * 60, 2)
            audio_feats = {
                "duration_seconds": dur,
                "words_per_minute": wpm_val,
                "silence_ratio": (float(total_pause_ms) / 1000.0 / max(dur, 1.0)) if total_pause_ms else 0.1,
                "pause_count": int(pause_count) if pause_count else 0,
                "filler_word_count": 0
            }

        # Text analysis
        text_feats = analyze_text(extracted_transcript)

        # ML Scoring & Feedback
        ml_score = predict_score(audio_feats, text_feats)
        scores = calculate_scores(audio_feats, text_feats, ml_score)
        feedback = generate_feedback(audio_feats, text_feats, scores)

        session_id = str(uuid.uuid4())
        overall_fb = " ".join(feedback)
        audio_url = f"/uploads/{saved_filename}" if saved_filename else None

        session_dict = {
            "id": session_id,
            "session_id": session_id,
            "topic": topic or "General Speech Practice",
            "transcript": extracted_transcript,
            "audio_filename": saved_filename,
            "audio_url": audio_url,
            "overall_score": int(scores.get("overall", 80)),
            "vocabulary_score": int(scores.get("vocabulary", 80)),
            "sentence_formation_score": int(scores.get("grammar", 80)),
            "fluency_score": int(scores.get("fluency", 80)),
            "confidence_score": int(scores.get("pronunciation", 80)),
            "corrections": [],
            "vocabulary_tips": [f for f in feedback if "vocabulary" in f.lower() or "word" in f.lower()],
            "overall_feedback": overall_fb,
            "duration_seconds": dur_secs,
            "words_per_minute": wpm_val,
            "scores": scores,
            "feedback": feedback
        }

        # Persist session with audio reference and scores to SQLite for the respective user
        persisted = db.save_practice_session(user["id"], session_dict)
        return persisted

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /api/practice/analyze:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ==========================================
# User Practice History
# ==========================================

@app.get("/api/practice/history")
async def get_practice_history(authorization: Optional[str] = Header(None)):
    """Retrieve speech practice sessions and evaluation scores for the current user."""
    user = get_current_user(authorization)
    return db.get_user_practice_history(user["id"])


# ==========================================
# Root /analyze endpoint (Compatibility)
# ==========================================

@app.post("/analyze")
async def analyze_audio_file(
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    try:
        user = get_current_user(authorization)

        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")

        extension = os.path.splitext(audio.filename)[1] or ".wav"
        filename = f"{uuid.uuid4()}{extension}"
        saved_path = os.path.join(UPLOAD_DIR, filename)

        contents = await audio.read()
        with open(saved_path, "wb") as f:
            f.write(contents)

        # 1. Speech -> Text
        transcript = transcribe_audio(saved_path)

        # 2. Audio analysis
        audio_features = analyze_audio(saved_path, transcript)

        # 3. Text analysis
        text_features = analyze_text(transcript)

        # 4. ML score
        score = predict_score(audio_features, text_features)
        scores = calculate_scores(audio_features, text_features, score)
        feedback = generate_feedback(audio_features, text_features, scores)

        session_id = str(uuid.uuid4())
        overall_feedback = " ".join(feedback)
        audio_url = f"/uploads/{filename}"

        session_dict = {
            "id": session_id,
            "session_id": session_id,
            "topic": "Speech AI Analysis",
            "transcript": transcript,
            "audio_filename": filename,
            "audio_url": audio_url,
            "overall_score": int(scores.get("overall", 80)),
            "vocabulary_score": int(scores.get("vocabulary", 80)),
            "sentence_formation_score": int(scores.get("grammar", 80)),
            "fluency_score": int(scores.get("fluency", 80)),
            "confidence_score": int(scores.get("pronunciation", 80)),
            "corrections": [],
            "vocabulary_tips": [f for f in feedback if "vocabulary" in f.lower() or "word" in f.lower()],
            "overall_feedback": overall_feedback,
            "duration_seconds": audio_features.get("duration_seconds", 0.0),
            "words_per_minute": audio_features.get("words_per_minute", 0.0),
            "scores": scores,
            "feedback": feedback
        }

        db.save_practice_session(user["id"], session_dict)

        return {
            "success": True,
            "session_id": session_id,
            "id": session_id,
            "overall_score": int(scores.get("overall", 80)),
            "vocabulary_score": int(scores.get("vocabulary", 80)),
            "sentence_formation_score": int(scores.get("grammar", 80)),
            "fluency_score": int(scores.get("fluency", 80)),
            "confidence_score": int(scores.get("pronunciation", 80)),
            "corrections": [],
            "vocabulary_tips": [f for f in feedback if "vocabulary" in f.lower() or "word" in f.lower()],
            "overall_feedback": overall_feedback,
            "audio_url": audio_url,
            "transcript": transcript,
            "audio_features": audio_features,
            "text_features": text_features,
            "scores": scores,
            "feedback": feedback
        }

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR in /analyze:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ==========================================
# Interview Bot Message
# ==========================================

@app.post("/api/interview/message")
async def interview_message(payload: dict = Body(...)):
    history = payload.get("history", [])
    last_user_msg = ""
    for msg in reversed(history):
        if msg.get("is_user") or msg.get("role") == "user" or msg.get("isUser"):
            last_user_msg = msg.get("text") or msg.get("content") or ""
            break

    reply = f"Thank you for sharing. In response to your point ('{last_user_msg if last_user_msg else 'your statement'}'), could you elaborate further on your experience and how it affected your goals?"
    return {"reply": reply}
