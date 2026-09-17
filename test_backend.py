import os
import sys
import unittest
import io
import json
from fastapi.testclient import TestClient

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
import app.db as db

class TestBackendIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure fresh DB tables
        db.init_db()

    def test_01_signup_with_phone(self):
        email = f"testuser_{os.urandom(4).hex()}@example.com"
        payload = {
            "email": email,
            "password": "SecretPassword123!",
            "name": "Jane Doe",
            "phone_no": "+1234567890"
        }
        res = self.client.post("/api/auth/register", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["email"], email)
        self.assertEqual(data["name"], "Jane Doe")
        self.assertEqual(data["phone_no"], "+1234567890")
        TestBackendIntegration.test_email = email
        TestBackendIntegration.test_token = data["access_token"]

    def test_02_login(self):
        payload = {
            "email": TestBackendIntegration.test_email,
            "password": "SecretPassword123!"
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["email"], TestBackendIntegration.test_email)
        self.assertEqual(data["name"], "Jane Doe")
        self.assertEqual(data["phone_no"], "+1234567890")

    def test_03_login_invalid_password(self):
        payload = {
            "email": TestBackendIntegration.test_email,
            "password": "WrongPassword!"
        }
        res = self.client.post("/api/auth/login", json=payload)
        self.assertEqual(res.status_code, 401)

    def test_04_practice_analyze_and_store(self):
        headers = {
            "Authorization": f"Bearer {TestBackendIntegration.test_token}"
        }
        # Create a dummy 1-second silent WAV in-memory
        import wave
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00\x00' * 16000)
        wav_buf.seek(0)

        files = {
            "audio": ("sample.wav", wav_buf, "audio/wav")
        }
        data = {
            "topic": "Describe your favorite hobby",
            "transcript": "I love playing basketball with my friends on weekends because it keeps me active and healthy.",
            "duration_ms": "3000",
            "word_count": "16",
            "pause_count": "1"
        }

        res = self.client.post("/api/practice/analyze", headers=headers, data=data, files=files)
        self.assertEqual(res.status_code, 200, res.text)
        result = res.json()
        self.assertIn("overall_score", result)
        self.assertIn("vocabulary_score", result)
        self.assertIn("sentence_formation_score", result)
        self.assertIn("fluency_score", result)
        self.assertIn("confidence_score", result)
        self.assertIsNotNone(result.get("audio_url"))
        self.assertTrue(result["audio_url"].startswith("/uploads/"))
        TestBackendIntegration.audio_url = result["audio_url"]

    def test_05_audio_file_served(self):
        res = self.client.get(TestBackendIntegration.audio_url)
        self.assertEqual(res.status_code, 200)
        self.assertGreater(len(res.content), 0)

    def test_06_practice_history(self):
        headers = {
            "Authorization": f"Bearer {TestBackendIntegration.test_token}"
        }
        res = self.client.get("/api/practice/history", headers=headers)
        self.assertEqual(res.status_code, 200, res.text)
        history = res.json()
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 1)
        first_item = history[0]
        self.assertEqual(first_item["topic"], "Describe your favorite hobby")
        self.assertIn("basketball", first_item["transcript"])
        self.assertIsNotNone(first_item["audio_url"])

if __name__ == "__main__":
    unittest.main()
