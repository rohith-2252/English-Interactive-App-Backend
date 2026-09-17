import os
import sqlite3
import hashlib
import secrets
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

# SQLite DB Path
DB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(DB_DIR, "speech_app.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash password using PBKDF2 HMAC SHA-256 with 100,000 iterations.
    Returns (salt_hex, hash_hex).
    """
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        bytes.fromhex(salt),
        100000
    )
    return salt, key.hex()

def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    _, key_hex = hash_password(password, salt)
    return secrets.compare_digest(key_hex, expected_hash)

def init_db():
    """Create the necessary database tables if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                name TEXT NOT NULL,
                phone_no TEXT,
                created_at TEXT NOT NULL
            );
        """)
        
        # 2. User Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        
        # 3. Practice Sessions (Audio & Scores) Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS practice_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                transcript TEXT,
                audio_filename TEXT,
                audio_url TEXT,
                overall_score INTEGER,
                vocabulary_score INTEGER,
                sentence_formation_score INTEGER,
                fluency_score INTEGER,
                confidence_score INTEGER,
                corrections TEXT,
                vocabulary_tips TEXT,
                overall_feedback TEXT,
                scores_json TEXT,
                feedback_json TEXT,
                duration_seconds REAL,
                words_per_minute REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_practice_user_id ON practice_sessions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_practice_created_at ON practice_sessions(created_at DESC);")
        
        conn.commit()

def create_user(email: str, password: str, name: str, phone_no: Optional[str] = None) -> Dict[str, Any]:
    """Create a new user in SQLite database."""
    email_clean = email.strip().lower()
    name_clean = name.strip() if name else email_clean.split('@')[0].capitalize()
    phone_clean = phone_no.strip() if phone_no else None
    
    salt, pwd_hash = hash_password(password)
    now = datetime.utcnow().isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO users (email, password_hash, salt, name, phone_no, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (email_clean, pwd_hash, salt, name_clean, phone_clean, now)
            )
            user_id = cursor.lastrowid
            conn.commit()
            return {
                "id": user_id,
                "email": email_clean,
                "name": name_clean,
                "phone_no": phone_clean,
                "created_at": now
            }
        except sqlite3.IntegrityError:
            raise ValueError("An account with this email already exists.")

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify email & password against SQLite database."""
    email_clean = email.strip().lower()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
        row = cursor.fetchone()
        if not row:
            return None
        
        if verify_password(password, row["salt"], row["password_hash"]):
            return {
                "id": row["id"],
                "email": row["email"],
                "name": row["name"],
                "phone_no": row["phone_no"],
                "created_at": row["created_at"]
            }
        return None

def create_session(user_id: int) -> str:
    """Generate and store an access token for user."""
    token = f"token-{uuid.uuid4().hex}"
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, now)
        )
        conn.commit()
    return token

def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Look up user by bearer session token."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.id, u.email, u.name, u.phone_no, u.created_at
            FROM user_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def get_or_create_guest_user() -> Dict[str, Any]:
    """Retrieve or create a default guest user for unauthenticated requests."""
    guest_email = "guest@englishcoach.local"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (guest_email,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    
    return create_user(
        email=guest_email,
        password=secrets.token_hex(16),
        name="Guest Learner",
        phone_no=None
    )

def save_practice_session(user_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
    """Save an analyzed speech practice session with audio and scores."""
    session_id = data.get("id") or data.get("session_id") or str(uuid.uuid4())
    now = data.get("created_at") or datetime.utcnow().isoformat()
    
    corrections_json = json.dumps(data.get("corrections", []))
    vocabulary_tips_json = json.dumps(data.get("vocabulary_tips", []))
    scores_json = json.dumps(data.get("scores", {}))
    feedback_json = json.dumps(data.get("feedback", []))
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO practice_sessions (
                id, user_id, topic, transcript, audio_filename, audio_url,
                overall_score, vocabulary_score, sentence_formation_score,
                fluency_score, confidence_score, corrections, vocabulary_tips,
                overall_feedback, scores_json, feedback_json,
                duration_seconds, words_per_minute, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                data.get("topic", "General English Practice"),
                data.get("transcript", ""),
                data.get("audio_filename"),
                data.get("audio_url"),
                int(data.get("overall_score", 0)),
                int(data.get("vocabulary_score", 0)),
                int(data.get("sentence_formation_score", 0)),
                int(data.get("fluency_score", 0)),
                int(data.get("confidence_score", 0)),
                corrections_json,
                vocabulary_tips_json,
                data.get("overall_feedback", ""),
                scores_json,
                feedback_json,
                float(data.get("duration_seconds") or 0.0),
                float(data.get("words_per_minute") or 0.0),
                now
            )
        )
        conn.commit()
    
    return {
        "id": session_id,
        "session_id": session_id,
        "topic": data.get("topic", "General English Practice"),
        "transcript": data.get("transcript", ""),
        "overall_score": int(data.get("overall_score", 0)),
        "vocabulary_score": int(data.get("vocabulary_score", 0)),
        "sentence_formation_score": int(data.get("sentence_formation_score", 0)),
        "fluency_score": int(data.get("fluency_score", 0)),
        "confidence_score": int(data.get("confidence_score", 0)),
        "corrections": data.get("corrections", []),
        "vocabulary_tips": data.get("vocabulary_tips", []),
        "overall_feedback": data.get("overall_feedback", ""),
        "audio_url": data.get("audio_url"),
        "created_at": now,
        "scores": data.get("scores", {}),
        "feedback": data.get("feedback", [])
    }

def get_user_practice_history(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all practice sessions for a specific user ordered by created_at DESC."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM practice_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            corrections = []
            try:
                if r["corrections"]:
                    corrections = json.loads(r["corrections"])
            except Exception:
                pass
                
            vocab_tips = []
            try:
                if r["vocabulary_tips"]:
                    vocab_tips = json.loads(r["vocabulary_tips"])
            except Exception:
                pass

            scores = {}
            try:
                if r["scores_json"]:
                    scores = json.loads(r["scores_json"])
            except Exception:
                pass

            feedback = []
            try:
                if r["feedback_json"]:
                    feedback = json.loads(r["feedback_json"])
            except Exception:
                pass

            results.append({
                "id": r["id"],
                "session_id": r["id"],
                "topic": r["topic"],
                "transcript": r["transcript"] or "",
                "overall_score": r["overall_score"],
                "vocabulary_score": r["vocabulary_score"],
                "sentence_formation_score": r["sentence_formation_score"],
                "fluency_score": r["fluency_score"],
                "confidence_score": r["confidence_score"],
                "corrections": corrections,
                "vocabulary_tips": vocab_tips,
                "overall_feedback": r["overall_feedback"] or "",
                "audio_url": r["audio_url"],
                "created_at": r["created_at"],
                "scores": scores,
                "feedback": feedback
            })
        return results

# Initialize DB on module import
init_db()
