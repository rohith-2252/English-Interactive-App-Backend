def generate_feedback(audio_features, text_features, scores):

    feedback = []
    recommendations = []

    # Speaking speed
    wpm = audio_features["words_per_minute"]

    if wpm < 100:
        feedback.append("Your speaking speed is slow.")
        recommendations.append(
            "Try to speak a little faster while maintaining clarity."
        )
    elif wpm <= 160:
        feedback.append("Your speaking speed is good.")
    else:
        feedback.append("Your speaking speed is fast.")
        recommendations.append(
            "Try slowing down slightly to improve clarity."
        )

    # Pauses
    pause_count = audio_features["pause_count"]

    if pause_count <= 10:
        feedback.append("Your pause control is good.")
    else:
        feedback.append("You are taking many pauses.")
        recommendations.append(
            "Practice speaking continuously and reduce unnecessary pauses."
        )

    # Fillers
    filler_count = audio_features["filler_word_count"]

    if filler_count == 0:
        feedback.append("No filler words detected.")
    else:
        feedback.append(
            f"{filler_count} filler words detected."
        )
        recommendations.append(
            "Reduce filler words such as um, uh, like, and basically."
        )

    # Vocabulary
    vocabulary = scores["vocabulary"]

    if vocabulary >= 80:
        feedback.append("Your vocabulary variety is good.")
    elif vocabulary >= 60:
        feedback.append("Your vocabulary variety is average.")
        recommendations.append(
            "Try using a wider range of words."
        )
    else:
        feedback.append("Your vocabulary variety is low.")
        recommendations.append(
            "Learn and practice new words regularly."
        )

    # Grammar
    grammar_errors = text_features[
        "grammar_error_count"
    ]

    if grammar_errors == 0:
        feedback.append("No major grammar errors detected.")
    elif grammar_errors <= 2:
        feedback.append(
            f"{grammar_errors} grammar issues detected."
        )
        recommendations.append(
            "Review the grammar mistakes identified in your transcript."
        )
    else:
        feedback.append(
            f"{grammar_errors} grammar issues detected."
        )
        recommendations.append(
            "Focus on sentence construction and basic grammar."
        )

    # Overall
    overall = scores["overall"]

    if overall >= 85:
        level = "Excellent"
    elif overall >= 70:
        level = "Good"
    elif overall >= 50:
        level = "Needs Improvement"
    else:
        level = "Beginner"

    return {
        "level": level,
        "feedback": feedback,
        "recommendations": recommendations
    }