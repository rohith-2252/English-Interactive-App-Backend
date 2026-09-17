
def clamp(value, minimum=0, maximum=100):
    return round(
        max(minimum, min(maximum, value)),
        2
    )


def calculate_scores(
    audio_features,
    text_features,
    ml_score
):

    wpm = audio_features.get(
        "words_per_minute", 0
    )

    silence = audio_features.get(
        "silence_ratio", 0
    )

    pauses = audio_features.get(
        "pause_count", 0
    )

    fillers = audio_features.get(
        "filler_word_count", 0
    )

    richness = text_features.get(
        "vocabulary_richness", 0
    )

    avg_sentence = text_features.get(
        "average_sentence_length", 0
    )

    verbs = text_features.get(
        "verb_count", 0
    )

    words = max(
        text_features.get("word_count", 1),
        1
    )


    # ==============================
    # FLUENCY
    # ==============================

    fluency = 100

    if wpm < 100:
        fluency -= 15

    elif wpm > 180:
        fluency -= 10


    if silence > 0.30:
        fluency -= 15

    elif silence > 0.20:
        fluency -= 7


    if pauses > 15:
        fluency -= 15

    elif pauses > 10:
        fluency -= 7


    if fillers > 3:
        fluency -= 15

    elif fillers > 0:
        fluency -= 5


    fluency = clamp(fluency)


    # ==============================
    # VOCABULARY
    # ==============================

    vocabulary = clamp(
        richness * 100
    )


    # ==============================
    # GRAMMAR
    # ==============================

    grammar = 80

    verb_ratio = verbs / words

    if verb_ratio < 0.05:
        grammar -= 15

    if avg_sentence < 5:
        grammar -= 5

    grammar = clamp(grammar)


    # ==============================
    # PRONUNCIATION
    # ==============================

    pronunciation = 100

    if silence > 0.35:
        pronunciation -= 10

    if wpm > 200:
        pronunciation -= 10

    pronunciation = clamp(
        pronunciation
    )


    # ==============================
    # OVERALL
    # ==============================

    overall = (
        fluency * 0.30
        + grammar * 0.25
        + vocabulary * 0.20
        + pronunciation * 0.10
        + ml_score * 0.15
    )

    overall = clamp(overall)


    return {
        "fluency": fluency,
        "grammar": grammar,
        "vocabulary": vocabulary,
        "pronunciation": pronunciation,
        "overall": overall
    }


def generate_feedback(
    audio_features,
    text_features,
    scores
):

    feedback = []


    # ==============================
    # SPEAKING SPEED
    # ==============================

    wpm = audio_features.get(
        "words_per_minute", 0
    )

    if wpm > 180:

        feedback.append(
            "Try speaking slightly slower "
            "for better clarity."
        )

    elif wpm < 100:

        feedback.append(
            "Try speaking a little faster "
            "to improve fluency."
        )

    else:

        feedback.append(
            "Your speaking speed is comfortable."
        )


    # ==============================
    # PAUSES
    # ==============================

    silence = audio_features.get(
        "silence_ratio", 0
    )

    if silence > 0.30:

        feedback.append(
            "There are several long pauses. "
            "Try to maintain a smoother flow."
        )


    # ==============================
    # FILLER WORDS
    # ==============================

    fillers = audio_features.get(
        "filler_word_count", 0
    )

    if fillers > 0:

        feedback.append(
            f"You used {fillers} filler words. "
            "Try replacing them with short pauses."
        )

    else:

        feedback.append(
            "Good control of filler words."
        )


    # ==============================
    # VOCABULARY
    # ==============================

    richness = text_features.get(
        "vocabulary_richness", 0
    )

    if richness < 0.50:

        feedback.append(
            "Try using a wider range of vocabulary."
        )

    else:

        feedback.append(
            "Your vocabulary variety is good."
        )


    # ==============================
    # GRAMMAR
    # ==============================

    if scores["grammar"] < 70:

        feedback.append(
            "Focus on forming complete sentences "
            "with clear verb usage."
        )

    else:

        feedback.append(
            "Your basic sentence structure looks good."
        )


    return feedback
