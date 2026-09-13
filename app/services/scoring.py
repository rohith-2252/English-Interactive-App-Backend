def calculate_score(audio_features, text_features):

    # -------------------------
    # SPEAKING SPEED
    # -------------------------

    wpm = audio_features["words_per_minute"]

    if 120 <= wpm <= 160:
        speed_score = 100
    elif 100 <= wpm < 120 or 160 < wpm <= 180:
        speed_score = 85
    elif 80 <= wpm < 100 or 180 < wpm <= 200:
        speed_score = 70
    else:
        speed_score = 50

    # -------------------------
    # PAUSES
    # -------------------------

    pause_count = audio_features["pause_count"]

    if pause_count <= 5:
        pause_score = 95
    elif pause_count <= 10:
        pause_score = 85
    elif pause_count <= 15:
        pause_score = 70
    else:
        pause_score = 50

    # -------------------------
    # FILLERS
    # -------------------------

    filler_count = audio_features["filler_word_count"]

    if filler_count == 0:
        filler_score = 100
    elif filler_count <= 2:
        filler_score = 90
    elif filler_count <= 5:
        filler_score = 75
    else:
        filler_score = 50

    # -------------------------
    # VOCABULARY
    # -------------------------

    richness = text_features["vocabulary_richness"]

    vocabulary_score = min(
        100,
        max(0, richness * 100)
    )

    # -------------------------
    # GRAMMAR
    # -------------------------

    grammar_errors = text_features[
        "grammar_error_count"
    ]

    word_count = text_features[
        "word_count"
    ]

    if word_count > 0:

        error_rate = (
            grammar_errors / word_count
        )

        grammar_score = max(
            0,
            100 - (error_rate * 200)
        )

    else:
        grammar_score = 0

    # -------------------------
    # FLUENCY
    # -------------------------

    fluency_score = (
        speed_score * 0.4 +
        pause_score * 0.3 +
        filler_score * 0.3
    )

    # -------------------------
    # OVERALL
    # -------------------------

    overall_score = (
        fluency_score * 0.40 +
        vocabulary_score * 0.25 +
        grammar_score * 0.35
    )

    return {

        "speaking_speed": round(
            speed_score, 2
        ),

        "pause_control": round(
            pause_score, 2
        ),

        "filler_control": round(
            filler_score, 2
        ),

        "fluency": round(
            fluency_score, 2
        ),

        "vocabulary": round(
            vocabulary_score, 2
        ),

        "grammar": round(
            grammar_score, 2
        ),

        "overall": round(
            overall_score, 2
        )
    }