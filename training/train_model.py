
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATASET = "datasets/training_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "speech_score_model.pkl"
)


# -------------------------
# Load dataset
# -------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET)

print("Dataset shape:", df.shape)


# -------------------------
# Features
# -------------------------

FEATURES = [
    "duration_seconds",
    "word_count",
    "words_per_minute",
    "pause_count",
    "silence_ratio",
    "filler_word_count",
    "sentence_count",
    "unique_word_count",
    "vocabulary_richness",
    "average_sentence_length",
    "noun_count",
    "verb_count",
    "adjective_count",
    "adverb_count"
]


TARGET = "overall_score"


# -------------------------
# Prepare data
# -------------------------

X = df[FEATURES].fillna(0)
y = df[TARGET].fillna(0)


# -------------------------
# Train / test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# -------------------------
# Model
# -------------------------

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# -------------------------
# Evaluation
# -------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    "Mean Absolute Error:",
    round(mae, 2)
)

print(
    "R² Score:",
    round(r2, 3)
)


# -------------------------
# Save model
# -------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_PATH
)


print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(MODEL_PATH)
