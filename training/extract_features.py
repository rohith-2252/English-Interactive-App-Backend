import os
import csv
import sys

# Allow importing from app/
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.services.whisper_service import transcribe_audio
from app.services.audio_analysis import analyze_audio
from app.services.text_analysis import analyze_text


INPUT_DIR = "datasets/common_voice"
OUTPUT_FILE = "datasets/speech_features.csv"


rows = []

wav_files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".wav")
)

print(f"Found {len(wav_files)} audio files")


for index, filename in enumerate(wav_files):

    audio_path = os.path.join(
        INPUT_DIR,
        filename
    )

    print(
        f"\n[{index + 1}/{len(wav_files)}] "
        f"Processing {filename}"
    )

    try:

        # -------------------------
        # 1. Whisper transcription
        # -------------------------

        transcript = transcribe_audio(audio_path)

        print("Transcript:", transcript)


        # -------------------------
        # 2. Audio features
        # -------------------------

        audio_features = analyze_audio(
            audio_path,
            transcript
        )

        print("Audio:", audio_features)


        # -------------------------
        # 3. Text features
        # -------------------------

        text_features = analyze_text(
            transcript
        )

        print("Text:", text_features)


        # -------------------------
        # 4. Combine features
        # -------------------------

        row = {
            "filename": filename,
            "transcript": transcript,

            "duration_seconds":
                audio_features.get("duration_seconds", 0),

            "word_count":
                audio_features.get("word_count", 0),

            "words_per_minute":
                audio_features.get("words_per_minute", 0),

            "pause_count":
                audio_features.get("pause_count", 0),

            "silence_ratio":
                audio_features.get("silence_ratio", 0),

            "filler_word_count":
                audio_features.get("filler_word_count", 0),

            "sentence_count":
                text_features.get("sentence_count", 0),

            "unique_word_count":
                text_features.get("unique_word_count", 0),

            "vocabulary_richness":
                text_features.get("vocabulary_richness", 0),

            "average_sentence_length":
                text_features.get(
                    "average_sentence_length", 0
                ),

            "noun_count":
                text_features.get("noun_count", 0),

            "verb_count":
                text_features.get("verb_count", 0),

            "adjective_count":
                text_features.get("adjective_count", 0),

            "adverb_count":
                text_features.get("adverb_count", 0)
        }

        rows.append(row)

    except Exception as e:

        print(
            f"ERROR processing {filename}: {e}"
        )


# -------------------------
# Save CSV
# -------------------------

if rows:

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    fieldnames = rows[0].keys()

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n==============================")
    print("FEATURE EXTRACTION COMPLETED")
    print("==============================")
    print(f"Samples processed: {len(rows)}")
    print(f"Dataset: {OUTPUT_FILE}")

else:

    print("\nNo samples were successfully processed.")