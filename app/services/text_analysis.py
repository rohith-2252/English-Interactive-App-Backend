import spacy
from collections import Counter
import language_tool_python

nlp = spacy.load("en_core_web_sm")

grammar_tool = language_tool_python.LanguageTool("en-US")


def analyze_text(text):

    doc = nlp(text)

    # -------------------------
    # WORDS
    # -------------------------

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]

    word_count = len(words)

    unique_words = set(words)

    vocabulary_richness = (
        len(unique_words) / word_count
        if word_count > 0
        else 0
    )

    # -------------------------
    # SENTENCES
    # -------------------------

    sentences = list(doc.sents)

    sentence_count = len(sentences)

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0
    )

    # -------------------------
    # REPETITION
    # -------------------------

    frequency = Counter(words)

    repeated_words = {
        word: count
        for word, count in frequency.items()
        if count > 2
    }

    # -------------------------
    # PARTS OF SPEECH
    # -------------------------

    nouns = sum(
        1 for token in doc
        if token.pos_ == "NOUN"
    )

    verbs = sum(
        1 for token in doc
        if token.pos_ == "VERB"
    )

    adjectives = sum(
        1 for token in doc
        if token.pos_ == "ADJ"
    )

    adverbs = sum(
        1 for token in doc
        if token.pos_ == "ADV"
    )

    # -------------------------
    # GRAMMAR
    # -------------------------

    matches = grammar_tool.check(text)

    grammar_errors = len(matches)

    grammar_error_details = []

    for match in matches[:10]:

        grammar_error_details.append({
            "message": match.message,
            "error": text[
                match.offset:
                match.offset + match.errorLength
            ],
            "suggestions": match.replacements[:3]
        })

    # -------------------------
    # RESULT
    # -------------------------

    return {

        "word_count": word_count,

        "sentence_count": sentence_count,

        "unique_word_count": len(unique_words),

        "vocabulary_richness": round(
            vocabulary_richness, 3
        ),

        "average_sentence_length": round(
            average_sentence_length, 2
        ),

        "repeated_words": repeated_words,

        "noun_count": nouns,

        "verb_count": verbs,

        "adjective_count": adjectives,

        "adverb_count": adverbs,

        "grammar_error_count": grammar_errors,

        "grammar_errors": grammar_error_details
    }