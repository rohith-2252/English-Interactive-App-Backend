
import pandas as pd
import numpy as np
import os


INPUT_FILE = "datasets/speech_features.csv"
OUTPUT_FILE = "datasets/training_data.csv"


df = pd.read_csv(INPUT_FILE)


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def fluency_score(row):
    score = 100.0

    wpm = row["words_per_minute"]

    # Comfortable speaking range
    if wpm < 90:
        score -= (90 - wpm) * 0.25
    elif wpm > 180:
        score -= (wpm - 180) * 0.25

    # Penalize excessive pauses
    pause_rate = (
        row["pause_count"] /
        max(row["duration_seconds"], 1)
    )

    score -= max(0, pause_rate - 0.8) * 12

    # Penalize excessive silence
    score -= row["silence_ratio"] * 20

    # Penalize fillers
    if row["word_count"] > 0:
        filler_rate = (
            row["filler_word_count"] /
            row["word_count"]
        )

        score -= filler_rate * 100

    return clamp(score)


def vocabulary_score(row):
    richness = row["vocabulary_richness"]

    # Vocabulary richness between 0 and 1
    score = richness * 100

    # Very short responses should not receive
    # artificially high vocabulary scores
    if row["word_count"] < 10:
        score *= 0.8

    return clamp(score)


def grammar_score(row):
    score = 100.0

    words = max(row["word_count"], 1)

    # Basic linguistic complexity indicators
    noun_ratio = row["noun_count"] / words
    verb_ratio = row["verb_count"] / words

    # Very low verb usage can indicate weak sentence structure
    if verb_ratio < 0.05:
        score -= 15

    # Reasonable noun usage
    if noun_ratio == 0:
        score -= 10

    # Penalize excessive repetition
    repeated = row.get("repeated_words", 0)

    if isinstance(repeated, str):
        repeated = 0

    score -= min(float(repeated) * 2, 20)

    return clamp(score)


def overall_score(row):

    fluency = fluency_score(row)
    vocabulary = vocabulary_score(row)
    grammar = grammar_score(row)

    return (
        fluency * 0.45 +
        vocabulary * 0.25 +
        grammar * 0.30
    )


print("Generating labels...")

df["fluency_score"] = df.apply(
    fluency_score,
    axis=1
)

df["vocabulary_score"] = df.apply(
    vocabulary_score,
    axis=1
)

df["grammar_score"] = df.apply(
    grammar_score,
    axis=1
)

df["overall_score"] = df.apply(
    overall_score,
    axis=1
)


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================")
print("LABEL GENERATION COMPLETED")
print("==============================")

print(f"Samples: {len(df)}")

print("\nScore ranges:")

print(
    "Fluency:",
    round(df["fluency_score"].min(), 2),
    "-",
    round(df["fluency_score"].max(), 2)
)

print(
    "Vocabulary:",
    round(df["vocabulary_score"].min(), 2),
    "-",
    round(df["vocabulary_score"].max(), 2)
)

print(
    "Grammar:",
    round(df["grammar_score"].min(), 2),
    "-",
    round(df["grammar_score"].max(), 2)
)

print(
    "Overall:",
    round(df["overall_score"].min(), 2),
    "-",
    round(df["overall_score"].max(), 2)
)

print(f"\nSaved: {OUTPUT_FILE}")
