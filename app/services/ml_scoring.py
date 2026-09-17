
import os
import joblib


MODEL_PATH = os.path.join(
    "models",
    "speech_score_model.pkl"
)


# Load model once when backend starts
print("Loading ML scoring model...")

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
FEATURES = model_data["features"]

print("ML scoring model loaded successfully.")


def predict_score(audio_features, text_features):

    feature_values = {
        **audio_features,
        **text_features
    }

    values = []

    for feature in FEATURES:
        values.append(
            feature_values.get(feature, 0)
        )

    import pandas as pd
    df = pd.DataFrame([values], columns=FEATURES)
    score = model.predict(df)[0]

    score = max(
        0,
        min(100, float(score))
    )

    return round(score, 2)

