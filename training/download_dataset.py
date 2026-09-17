from huggingface_hub import hf_hub_download
import pandas as pd
import os
import io
import soundfile as sf

REPO_ID = "saeedzou/common-voice-17-en-age-gender-accent-sampled"

OUTPUT_DIR = "datasets/common_voice"
TARGET_SAMPLES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading Parquet file...")

parquet_file = hf_hub_download(
    repo_id=REPO_ID,
    filename="data/train-00000-of-00003.parquet",
    repo_type="dataset"
)

print("Reading dataset...")

df = pd.read_parquet(parquet_file)

print("Columns:")
print(df.columns.tolist())

print(f"Total rows available: {len(df)}")

count = 0

for _, row in df.iterrows():

    if count >= TARGET_SAMPLES:
        break

    try:
        audio = row["audio"]
        sentence = row["sentence"]

        # Audio may be stored as bytes
        if isinstance(audio, dict):

            audio_bytes = audio.get("bytes")

            if audio_bytes is not None:

                audio_data, sample_rate = sf.read(
                    io.BytesIO(audio_bytes)
                )

                audio_path = os.path.join(
                    OUTPUT_DIR,
                    f"sample_{count:04d}.wav"
                )

                sf.write(
                    audio_path,
                    audio_data,
                    sample_rate
                )

        # Save transcript
        text_path = os.path.join(
            OUTPUT_DIR,
            f"sample_{count:04d}.txt"
        )

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(str(sentence))

        count += 1

        print(f"Saved {count}/{TARGET_SAMPLES}")

    except Exception as e:
        print("Skipping sample:", e)

print("\n================================")
print("DATASET DOWNLOAD COMPLETED")
print("================================")
print(f"Samples saved: {count}")