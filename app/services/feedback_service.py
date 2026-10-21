"""
feedback_service.py
===================
Dynamic, text-aware scoring and feedback generation.

calculate_scores():
    Uses REAL grammar error counts (from spaCy analysis) + audio metrics.
    Grammar score starts at 100 and deducts per detected error.
    Vocabulary score uses richness + sophistication (advanced words, avg length).

generate_feedback():
    References ACTUAL issues found: names the grammar errors, calls out
    repeated words by name, notes missing transitions, etc.

generate_recommendations():
    Returns actionable practice tips customised to the scores and features.
    Always distinct from generate_feedback() output.
"""

from __future__ import annotations


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 40.0, hi: float = 100.0) -> int:
    """Clamp a float to [lo, hi] and return as int."""
    return int(max(lo, min(hi, value)))


# ─── Score calculation ────────────────────────────────────────────────────────

def calculate_scores(audio_features: dict, text_features: dict, ml_score: float) -> dict:
    """
    Produce per-dimension scores (0-100) from audio + text features.
    All scores reflect ACTUAL measured quality, not hardcoded baselines.
    """

    # ── Fluency ───────────────────────────────────────────────────────────
    wpm      = audio_features.get("words_per_minute", 120)
    silence  = audio_features.get("silence_ratio", 0.1)
    pauses   = audio_features.get("pause_count", 0)
    fillers  = audio_features.get("filler_word_count", 0)

    fluency = 100.0
    if wpm > 200:   fluency -= 20
    elif wpm > 180: fluency -= 10
    elif wpm < 80:  fluency -= 20
    elif wpm < 100: fluency -= 10

    if silence > 0.40:   fluency -= 20
    elif silence > 0.25: fluency -= 10

    if pauses > 10:  fluency -= 15
    elif pauses > 6: fluency -= 8

    if fillers > 5:   fluency -= 20
    elif fillers > 2: fluency -= 10
    elif fillers > 0: fluency -= 5
    fluency = _clamp(fluency, 40, 100)

    # ── Grammar ───────────────────────────────────────────────────────────
    grammar_errors  = text_features.get("grammar_error_count", 0)
    fragment_count  = text_features.get("fragment_count", 0)
    avg_sent        = text_features.get("average_sentence_length", 0)
    word_count      = text_features.get("word_count", 0)
    verb_count      = text_features.get("verb_count", 0)

    grammar = 100.0
    grammar -= min(grammar_errors  * 18, 54)  # up to -54 for grammar errors
    grammar -= min(fragment_count  * 12, 24)  # up to -24 for fragments

    verb_ratio = verb_count / max(word_count, 1)
    if verb_ratio < 0.04: grammar -= 12
    elif verb_ratio < 0.08: grammar -= 5

    if avg_sent < 4 and word_count > 0: grammar -= 8
    grammar = _clamp(grammar, 40, 100)

    # ── Vocabulary ────────────────────────────────────────────────────────
    richness       = text_features.get("vocabulary_richness", 0.5)
    advanced_count = text_features.get("advanced_word_count", 0)
    basic_ratio    = text_features.get("basic_word_ratio", 0.3)
    avg_wlen       = text_features.get("avg_word_length", 4.0)
    trans_count    = text_features.get("transition_word_count", 0)

    vocabulary = (richness * 60)
    vocabulary += min(advanced_count * 5, 25)
    vocabulary += min(avg_wlen * 2, 10)
    vocabulary += min(trans_count * 3, 10)
    vocabulary -= min(basic_ratio * 30, 15)
    vocabulary = _clamp(vocabulary, 40, 100)

    # ── Pronunciation (confidence proxy) ─────────────────────────────────
    pronunciation = 100.0
    if wpm > 190 or (wpm < 85 and wpm > 0):  pronunciation -= 15
    if silence > 0.35:         pronunciation -= 10
    if fillers > 3:            pronunciation -= 10
    pronunciation = _clamp(pronunciation, 40, 100)

    # ── Overall (weighted) ────────────────────────────────────────────────
    overall = int(
        fluency      * 0.30
        + grammar    * 0.30
        + vocabulary * 0.25
        + pronunciation * 0.15
    )
    overall = _clamp(overall, 40, 100)

    return {
        "fluency":      fluency,
        "grammar":      grammar,
        "vocabulary":   vocabulary,
        "pronunciation": pronunciation,
        "overall":      overall,
    }


# ─── Feedback generation ──────────────────────────────────────────────────────

def generate_feedback(audio_features: dict, text_features: dict, scores: dict) -> list[str]:
    """
    Generate specific, text-aware feedback sentences.
    References ACTUAL errors found — not generic boilerplate.
    """
    feedback: list[str] = []

    wpm        = audio_features.get("words_per_minute", 0)
    fillers    = audio_features.get("filler_word_count", 0)

    grammar_errors  = text_features.get("grammar_errors", [])
    error_count     = text_features.get("grammar_error_count", 0)
    fragment_count  = text_features.get("fragment_count", 0)
    richness        = text_features.get("vocabulary_richness", 0)
    repeated_words  = text_features.get("repeated_words", {})
    trans_count     = text_features.get("transition_word_count", 0)
    advanced_count  = text_features.get("advanced_word_count", 0)
    avg_sent        = text_features.get("average_sentence_length", 0)

    # ── Speaking pace ──────────────────────────────────────────────────────
    if wpm > 180:
        feedback.append(
            f"You spoke at {int(wpm)} words/min — slightly fast. "
            "Aim for 120-160 wpm to improve clarity."
        )
    elif wpm < 100 and wpm > 0:
        feedback.append(
            f"Your pace ({int(wpm)} wpm) is slow. "
            "Try speaking with more fluency and fewer long pauses."
        )
    else:
        if wpm > 0:
            feedback.append(f"Good speaking pace ({int(wpm)} wpm) — comfortable to follow.")

    # ── Grammar errors ─────────────────────────────────────────────────────
    if error_count > 0:
        ex = grammar_errors[0] if grammar_errors else ""
        detail = f" Example: {ex}" if ex else ""
        feedback.append(
            f"Found {error_count} grammar error(s).{detail}. "
            "Focus on correct verb forms and complete sentences."
        )
    if fragment_count > 0:
        feedback.append(
            f"{fragment_count} incomplete sentence(s) detected — "
            "make sure every sentence has both a subject and a verb."
        )
    if error_count == 0 and fragment_count == 0:
        feedback.append("Your sentences are grammatically well-formed.")

    # ── Vocabulary & repetition ────────────────────────────────────────────
    top_rep = sorted(repeated_words.items(), key=lambda x: -x[1])[:3]
    if top_rep:
        rw_str = ", ".join(f'"{w}" ({c}x)' for w, c in top_rep)
        feedback.append(f"Repeated words: {rw_str}. Use synonyms to show richer vocabulary.")
    if richness < 0.50 and richness > 0:
        feedback.append(
            f"Vocabulary richness is {int(richness*100)}% — many words reused. "
            "Expand by learning topic-specific vocabulary."
        )
    if advanced_count >= 3:
        feedback.append(f"Good use of {advanced_count} advanced/academic word(s).")
    elif advanced_count == 0:
        feedback.append(
            "No advanced vocabulary detected. "
            "Try using words like 'significant', 'demonstrate', 'foundation' etc."
        )

    # ── Sentence structure ─────────────────────────────────────────────────
    if trans_count == 0:
        feedback.append(
            "No linking words found. Add transitions like 'however', "
            "'therefore', 'furthermore' to connect ideas naturally."
        )
    elif trans_count >= 2:
        feedback.append(f"Good use of {trans_count} linking word(s) to connect ideas.")

    if avg_sent > 25:
        feedback.append(
            f"Your sentences are long (avg {avg_sent:.0f} words). "
            "Break them into shorter, cleaner sentences."
        )

    # ── Filler words ───────────────────────────────────────────────────────
    if fillers > 2:
        feedback.append(
            f"You used {fillers} filler sounds. "
            "Replace them with a brief pause instead."
        )

    return feedback


# ─── Recommendations generation ───────────────────────────────────────────────

def generate_recommendations(
    audio_features: dict,
    text_features: dict,
    scores: dict,
) -> list[str]:
    """
    Generate specific, actionable improvement tips based on ACTUAL scores.
    Always returns at least 4 tips. Content is DIFFERENT from generate_feedback().
    """
    tips: list[str] = []

    wpm           = audio_features.get("words_per_minute", 0)
    fillers       = audio_features.get("filler_word_count", 0)
    grammar_errs  = text_features.get("grammar_errors", [])
    corrections   = text_features.get("corrections", [])
    error_count   = text_features.get("grammar_error_count", 0)
    fragment_count = text_features.get("fragment_count", 0)
    word_count    = text_features.get("word_count", 0)
    trans_count   = text_features.get("transition_word_count", 0)

    vocab_score    = scores.get("vocabulary", 100)
    overall_score  = scores.get("overall", 100)

    # ── Grammar-specific tips ──────────────────────────────────────────────
    if corrections:
        first_c = corrections[0]
        tips.append(
            f"Review grammar correction: '{first_c['mistake']}' -> '{first_c['correction']}'. "
            f"{first_c.get('explanation', '')}"
        )
    elif error_count > 0 and grammar_errs:
        first_err = grammar_errs[0]
        tips.append(
            f"Fix detected grammar issue: '{first_err}'. "
            "Practise this pattern: Subject + auxiliary verb + verb-ing. "
            "E.g. 'I am currently studying' NOT 'I am currently study'."
        )
    elif fragment_count > 0:
        tips.append(
            "Practise forming COMPLETE sentences: every sentence needs "
            "a subject AND a verb. Write 5 complete sentences before "
            "your next speaking session."
        )
    else:
        tips.append(
            "Challenge yourself with complex sentence structures. "
            "Try starting sentences with 'Although...', 'Despite...', "
            "or 'Not only... but also...'."
        )

    # ── Fluency / pace tips ────────────────────────────────────────────────
    if wpm > 180:
        tips.append(
            "Read a paragraph aloud slowly and deliberately. "
            "Record yourself and aim for 130-150 wpm in your next attempt."
        )
    elif wpm < 100 and wpm > 0:
        tips.append(
            "Practise speaking continuously for 60 seconds on any topic "
            "without stopping. This builds fluency and reduces long pauses."
        )
    else:
        tips.append(
            "Maintain your pace. Try shadow-reading a podcast or news "
            "article aloud to sharpen rhythm and intonation."
        )

    # ── Vocabulary tips ────────────────────────────────────────────────────
    if vocab_score < 55:
        tips.append(
            "Learn 5 new words daily from your study/tech domain and "
            "use each one in a spoken sentence the same day."
        )
    elif vocab_score < 75:
        tips.append(
            "When you repeat a word, swap it for a synonym. "
            "Instead of 'good' try 'effective', 'significant', or 'substantial'."
        )
    else:
        tips.append(
            "Expand into collocations (words that go naturally together): "
            "'strong foundation', 'critical thinking', 'problem-solving skills'."
        )

    # ── Transition / structure tips ────────────────────────────────────────
    if trans_count == 0:
        tips.append(
            "Add ONE linking word to every 2-3 sentences: 'however', "
            "'therefore', 'furthermore', 'in addition'. "
            "This alone makes speech sound 50% more structured."
        )
    else:
        tips.append(
            "Use the PEEL method: Point -> Example -> Explanation -> Link. "
            "This keeps longer answers organised and coherent."
        )

    # ── Content depth ─────────────────────────────────────────────────────
    if word_count < 40 and word_count > 0:
        tips.append(
            "Speak for longer — aim for at least 60 seconds / 100 words. "
            "Elaborate with: reason, example, and a conclusion."
        )

    # ── Filler tips ────────────────────────────────────────────────────────
    if fillers > 2:
        tips.append(
            "Replace filler sounds (um, uh, er) with a confident 1-second "
            "pause. Silence sounds professional; fillers sound hesitant."
        )

    # ── Overall motivational tip ───────────────────────────────────────────
    if overall_score >= 85:
        tips.append(
            "Excellent performance! Try answering unfamiliar questions "
            "within 5 seconds — this is the hallmark of advanced fluency."
        )
    elif overall_score >= 65:
        tips.append(
            "You are on the right track. Practise 10 minutes daily "
            "— consistency improves spoken English faster than occasional long sessions."
        )
    else:
        tips.append(
            "Focus on ONE area this week: grammar. "
            "Speak slowly, check each sentence has a subject and a verb, "
            "and gradually add more complex structures."
        )

    return tips
