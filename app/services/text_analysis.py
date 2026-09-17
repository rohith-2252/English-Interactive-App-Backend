
import re
from collections import Counter

import spacy


# Load English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy English model is missing.\n"
        "Run:\n"
        "python -m spacy download en_core_web_sm"
    )


def analyze_text(text: str) -> dict:

    if not text or not text.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "unique_word_count": 0,
            "vocabulary_richness": 0,
            "average_sentence_length": 0,
            "repeated_words": {},
            "noun_count": 0,
            "verb_count": 0,
            "adjective_count": 0,
            "adverb_count": 0
        }

    doc = nlp(text)

    # -------------------------
    # Words
    # -------------------------

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]

    word_count = len(words)

    unique_words = set(words)

    unique_word_count = len(unique_words)

    # -------------------------
    # Vocabulary richness
    # -------------------------

    if word_count > 0:
        vocabulary_richness = (
            unique_word_count / word_count
        )
    else:
        vocabulary_richness = 0

    # -------------------------
    # Sentences
    # -------------------------

    sentences = list(doc.sents)

    sentence_count = len(sentences)

    if sentence_count > 0:
        average_sentence_length = (
            word_count / sentence_count
        )
    else:
        average_sentence_length = 0

    # -------------------------
    # Repeated words
    # -------------------------

    word_frequency = Counter(words)

    repeated_words = {
        word: count
        for word, count in word_frequency.items()
        if count > 1
    }

    # -------------------------
    # POS counts
    # -------------------------

    noun_count = sum(
        1 for token in doc
        if token.pos_ in {"NOUN", "PROPN"}
    )

    verb_count = sum(
        1 for token in doc
        if token.pos_ == "VERB"
    )

    adjective_count = sum(
        1 for token in doc
        if token.pos_ == "ADJ"
    )

    adverb_count = sum(
        1 for token in doc
        if token.pos_ == "ADV"
    )

    return {
        "word_count": word_count,

        "sentence_count": sentence_count,

        "unique_word_count": unique_word_count,

        "vocabulary_richness": round(
            vocabulary_richness,
            3
        ),

        "average_sentence_length": round(
            average_sentence_length,
            2
        ),

        "repeated_words": repeated_words,

        "noun_count": noun_count,

        "verb_count": verb_count,

        "adjective_count": adjective_count,

        "adverb_count": adverb_count
    }

