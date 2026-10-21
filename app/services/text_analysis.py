"""
text_analysis.py
================
Advanced NLP Spoken English Analyzer powered by spaCy POS Tagging and Dependency Parsing.

Features:
  - Bagging/boosting transcript correction and domain disambiguation
  - Grammar error detection (tense, agreement, auxiliary usage, verb form)
  - Sentence structure analysis (run-ons, fragments, punctuation)
  - Spoken English vocabulary metrics (richness, academic words, connectors)
  - Emits structured corrections: [{"mistake": "...", "correction": "...", "explanation": "..."}]
"""

from __future__ import annotations
import re
from collections import Counter
import spacy

from app.services.transcript_corrector import optimize_transcript

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# Discourse & transition markers
_TRANSITION_WORDS = {
    "however", "therefore", "moreover", "furthermore", "consequently",
    "nevertheless", "meanwhile", "additionally", "subsequently",
    "although", "despite", "whereas", "thus", "hence", "accordingly",
    "firstly", "secondly", "finally", "also", "besides", "otherwise",
    "for example", "for instance", "in conclusion", "to summarize",
    "in addition", "as a result", "on the other hand", "in contrast",
    "in fact", "that is", "such as", "in other words",
}

# Advanced / Academic & Technical vocabulary
_ADVANCED_WORDS = {
    "substantial", "significant", "fundamental", "demonstrate", "illustrate",
    "analyze", "analyse", "evaluate", "implement", "establish", "contribute",
    "emphasize", "facilitate", "generate", "incorporate", "investigate",
    "maintain", "obtain", "participate", "perceive", "require", "acquire",
    "sufficient", "appropriate", "efficient", "effective", "comprehensive",
    "structure", "develop", "enhance", "foundation", "logical", "critical",
    "problem", "solution", "strategy", "approach", "objective", "concept",
    "principle", "methodology", "framework", "perspective", "capability",
    "competency", "proficiency", "articulate", "coherent", "elaborate",
    "distinguish", "identify", "clarify", "justify", "prioritize", "sharpen",
    "strengthen", "engineering", "technology", "programming", "foundation",
}

# Overused / basic vocabulary
_BASIC_WORDS = {
    "good", "bad", "nice", "thing", "stuff", "get", "got", "make",
    "do", "did", "very", "really", "just", "like", "said", "big",
    "small", "go", "come", "put", "give", "take", "know", "feel",
    "want", "need", "use", "used", "lots", "kind", "sort", "way",
}


def analyze_text(text: str) -> dict:
    """
    Comprehensive NLP analysis of spoken transcript.
    Returns feature dictionary containing metrics, structured corrections, and polished text.
    """
    empty = {
        "word_count": 0, "sentence_count": 0, "unique_word_count": 0,
        "vocabulary_richness": 0.0, "average_sentence_length": 0.0,
        "repeated_words": {}, "noun_count": 0, "verb_count": 0,
        "adjective_count": 0, "adverb_count": 0,
        "grammar_error_count": 0, "grammar_errors": [],
        "corrections": [],
        "fragment_count": 0,
        "transition_word_count": 0, "advanced_word_count": 0,
        "basic_word_ratio": 0.0, "avg_word_length": 0.0,
        "sentence_length_variance": 0.0, "content_word_ratio": 0.0,
        "optimized_transcript": text or "",
    }
    if not text or not text.strip():
        return empty

    # 1. Multi-Stage Transcript Optimization & Corrections
    optimized_text, corrections = optimize_transcript(text, nlp)

    # 2. Linguistic analysis of both original and optimized text
    doc_orig = nlp(text)
    doc_opt = nlp(optimized_text)

    words = [t.text.lower() for t in doc_orig if t.is_alpha]
    word_count = len(words)
    if word_count == 0:
        return empty

    unique_words = set(words)
    unique_word_count = len(unique_words)
    vocabulary_richness = round(unique_word_count / word_count, 3)

    sentences = list(doc_opt.sents)
    sentence_count = len(sentences)
    sent_lens = [len([t for t in s if t.is_alpha]) for s in sentences]
    average_sentence_length = round(sum(sent_lens) / sentence_count, 2) if sentence_count else 0.0
    if len(sent_lens) > 1:
        mean_sl = sum(sent_lens) / len(sent_lens)
        variance = round(sum((l - mean_sl) ** 2 for l in sent_lens) / len(sent_lens), 2)
    else:
        variance = 0.0

    word_freq = Counter(words)
    stop_words = {t.text.lower() for t in doc_orig if t.is_stop}
    repeated_words = {w: c for w, c in word_freq.items() if c > 1 and w not in stop_words and len(w) > 2}

    noun_count = sum(1 for t in doc_opt if t.pos_ in {"NOUN", "PROPN"})
    verb_count = sum(1 for t in doc_opt if t.pos_ == "VERB")
    adjective_count = sum(1 for t in doc_opt if t.pos_ == "ADJ")
    adverb_count = sum(1 for t in doc_opt if t.pos_ == "ADV")
    content_count = noun_count + verb_count + adjective_count + adverb_count
    content_word_ratio = round(content_count / word_count, 3)

    lemmas = {t.lemma_.lower() for t in doc_opt if t.is_alpha}
    advanced_word_count = len(lemmas & _ADVANCED_WORDS)
    basic_word_count = sum(1 for w in words if w in _BASIC_WORDS)
    basic_word_ratio = round(basic_word_count / word_count, 3)
    avg_word_length = round(sum(len(w) for w in words) / word_count, 2)

    text_lower = text.lower()
    transition_word_count = sum(
        1 for tw in _TRANSITION_WORDS
        if re.search(r"\b" + re.escape(tw) + r"\b", text_lower)
    )

    # 3. Detect additional syntactic issues & sentence structure
    grammar_error_descriptions = []

    # Run-on check
    if word_count > 25 and len(sentences) <= 1:
        corrections.append({
            "mistake": "Unpunctuated continuous speech",
            "correction": "Divided into distinct sentences",
            "explanation": f"Sentence structure: You spoke a long run-on clause ({word_count} words). Use pauses and full stops between thoughts."
        })
        grammar_error_descriptions.append("Run-on sentence structure")

    # Fragment check on sentences
    fragment_count = 0
    for s in sentences:
        alphas = [t for t in s if t.is_alpha]
        if len(alphas) >= 3:
            root_tokens = [t for t in s if t.dep_ == "ROOT"]
            if root_tokens and root_tokens[0].pos_ not in ("VERB", "AUX"):
                fragment_count += 1

    for c in corrections:
        grammar_error_descriptions.append(f"{c['mistake']} -> {c['correction']}")

    grammar_error_count = len(corrections)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_word_count": unique_word_count,
        "vocabulary_richness": vocabulary_richness,
        "average_sentence_length": average_sentence_length,
        "repeated_words": repeated_words,
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adjective_count": adjective_count,
        "adverb_count": adverb_count,
        "grammar_error_count": grammar_error_count,
        "grammar_errors": grammar_error_descriptions,
        "corrections": corrections,
        "fragment_count": fragment_count,
        "transition_word_count": transition_word_count,
        "advanced_word_count": advanced_word_count,
        "basic_word_ratio": basic_word_ratio,
        "avg_word_length": avg_word_length,
        "sentence_length_variance": variance,
        "content_word_ratio": content_word_ratio,
        "optimized_transcript": optimized_text,
    }
