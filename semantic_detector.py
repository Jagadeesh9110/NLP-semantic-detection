"""
Semantic Field Detection — Production Module
=============================================
Course  : DS357 – Natural Language Processing (NLP)
Author  : Manyam Jagadeeswar Reddy (23BDS033)
Institution : IIIT Dharwad

Description:
    This module implements a context-aware semantic field detection system
    that classifies English sentences into one of five domains: Medical,
    Technology, Finance, Sports, or Food. It resolves polysemy (words with
    multiple meanings) by analysing surrounding context via a sliding window
    algorithm, and provides explainable confidence scoring.

Dependencies:
    - nltk
    - wn  (Open English WordNet Python API)
"""

# =============================================================================
# 1. IMPORTS & SETUP
# =============================================================================

import re
import string

import nltk
import wn

# ── NLTK data downloads (idempotent; no-op if already present) ──────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Open English WordNet 2023 download (idempotent) ─────────────────────────
try:
    _oewn = wn.Wordnet("oewn:2023")
except wn.Error:
    wn.download("oewn:2023")
    _oewn = wn.Wordnet("oewn:2023")


# =============================================================================
# 2. WORDNET SYNONYM EXPANSION
# =============================================================================

def get_wordnet_synonyms(word: str) -> list[str]:
    """
    Retrieve all synonyms (lemmas) for *word* from the Open English WordNet
    2023 edition.  Multi-word lemmas have underscores replaced by spaces and
    are lowercased for consistent matching.

    Parameters
    ----------
    word : str
        The seed word to expand.

    Returns
    -------
    list[str]
        Deduplicated list of synonyms discovered via synset traversal.
    """
    synonyms: set[str] = set()
    for w in _oewn.words(word):
        for sense in w.senses():
            synset = sense.synset()
            for lemma in synset.lemmas():
                synonyms.add(lemma.lower().replace("_", " "))
    return list(synonyms)


# =============================================================================
# 3. DOMAIN MAP (curated seed keywords including polysemy traps)
# =============================================================================

DOMAIN_MAP: dict[str, list[str]] = {
    "food": [
        "apple", "banana", "rice", "bread", "fruit", "vegetable",
        "cook", "meal", "eat", "drink", "recipe", "sugar", "salt",
        "chicken", "fish", "server",
    ],
    "technology": [
        "apple", "screen", "battery", "software", "hardware",
        "computer", "phone", "app", "code", "program", "data",
        "server", "internet", "device", "virus", "monitor",
    ],
    "medical": [
        "hospital", "doctor", "nurse", "patient", "medicine",
        "surgery", "diagnosis", "drug", "virus", "treatment",
        "symptom", "disease", "health", "monitor",
    ],
    "sports": [
        "football", "goal", "player", "match", "team", "score",
        "stadium", "coach", "tournament", "athlete", "run",
        "cricket", "basketball", "bat",
    ],
    "finance": [
        "bank", "money", "loan", "interest", "stock", "invest",
        "market", "trade", "profit", "tax", "currency", "fund",
        "budget", "economy", "price",
    ],
}


# =============================================================================
# 4. TEXT PREPROCESSING
# =============================================================================

# Pre-compile resources to avoid repeated construction inside hot loops.
_STOP_WORDS: set[str] = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def preprocess_text(sentence: str) -> list[str]:
    """
    Apply a standard NLP preprocessing pipeline to *sentence*.

    Steps:
        1. Lowercase and strip punctuation.
        2. Tokenise on whitespace.
        3. Remove English stop-words.
        4. Lemmatise each remaining token.

    Parameters
    ----------
    sentence : str
        Raw English sentence.

    Returns
    -------
    list[str]
        Cleaned, lemmatised token list.
    """
    text = sentence.lower()
    text = _PUNCTUATION_RE.sub("", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS]

    clean_tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return clean_tokens


# =============================================================================
# 5. APPROACH 1 — BASELINE DICTIONARY LOOKUP
# =============================================================================

def approach1_baseline(
    sentence: str,
) -> tuple[str, str | None]:
    """
    Naïve baseline: scan tokens left-to-right and return the domain of the
    first keyword match.  This approach is context-blind and therefore
    vulnerable to polysemy (e.g. 'apple' always maps to Food because it
    appears in that domain list first).

    Parameters
    ----------
    sentence : str
        Raw English sentence.

    Returns
    -------
    tuple[str, str | None]
        (predicted_domain, trigger_word).  Both are ``"unknown"`` / ``None``
        when no domain keyword is found.
    """
    clean_words = preprocess_text(sentence)

    for word in clean_words:
        for domain, keywords in DOMAIN_MAP.items():
            if word in keywords:
                return domain, word

    return "unknown", None


# =============================================================================
# 6. APPROACH 2 — CONTEXT-AWARE SLIDING WINDOW
# =============================================================================

def approach2_context_window(
    sentence: str,
    window_size: int = 5,
) -> tuple[str, dict[str, int], list[str]]:
    """
    Context-aware algorithm that resolves polysemy by evaluating the
    aggregate semantic 'pull' of neighbouring words within a sliding window.

    Inspired by the FIRE model (NeurIPS 2022), which represents word meaning
    as a *field* rather than a point — the realised domain of a polysemous
    word is determined by the field contributions of its neighbours.

    Parameters
    ----------
    sentence : str
        Raw English sentence.
    window_size : int, default 5
        Width of the sliding context window (centred on the current token).

    Returns
    -------
    tuple[str, dict[str, int], list[str]]
        * **best_domain** – the domain with the highest accumulated score.
        * **domain_scores** – vote tallies per domain.
        * **contributors** – keywords that voted for the winning domain.
    """
    clean_words = preprocess_text(sentence)
    domain_scores: dict[str, int] = {}
    word_contributions: dict[str, set[str]] = {}

    # Slide a window over the token sequence.
    for i, _word in enumerate(clean_words):
        start = max(0, i - window_size // 2)
        end = min(len(clean_words), i + window_size // 2 + 1)
        window = clean_words[start:end]

        # Accumulate domain votes from every keyword in the window.
        for w in window:
            for domain, keywords in DOMAIN_MAP.items():
                if w in keywords:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 1

                    if domain not in word_contributions:
                        word_contributions[domain] = set()
                    word_contributions[domain].add(w)

    if not domain_scores:
        return "unknown", {}, []

    best_domain = max(domain_scores, key=domain_scores.get)  # type: ignore[arg-type]
    contributors = list(word_contributions.get(best_domain, []))

    return best_domain, domain_scores, contributors


# =============================================================================
# 7. EXPLAINABLE SEMANTIC DETECTION (XAI + CONFIDENCE SCORING)
# =============================================================================

def explainable_semantic_detection(sentence: str) -> dict:
    """
    Wrap Approach 2 with an Explainable AI (XAI) layer.

    Enhancements over the raw context-window output:
        * **Confidence score** — percentage of total votes captured by the
          winning domain.
        * **Ambiguity flag** — when confidence falls below 60 %, the result
          is marked ``AMBIGUOUS``, signalling that multiple domains are
          strongly represented (inspired by ACL 2023 uncertainty estimation
          research).
        * **Feature-importance trace** — per-keyword contribution
          percentages so the user can inspect *why* a domain was selected.

    Parameters
    ----------
    sentence : str
        Raw English sentence.

    Returns
    -------
    dict
        Keys: ``sentence``, ``domain``, ``confidence``, ``status``,
        ``explanation``.
    """
    domain, scores, keywords = approach2_context_window(sentence, window_size=5)

    # ── No signal path ──────────────────────────────────────────────────
    if not scores:
        return {
            "sentence": sentence,
            "domain": "UNKNOWN",
            "confidence": 0,
            "status": "WARNING: No semantic signals detected",
            "explanation": [],
        }

    # ── Confidence calculation ──────────────────────────────────────────
    total_votes = sum(scores.values())
    domain_votes = scores[domain]
    confidence = round((domain_votes / total_votes) * 100, 1)

    # ── XAI feature-importance trace ────────────────────────────────────
    explanation: list[str] = []
    if keywords:
        weight_per_word = round(100 / len(keywords), 1)
        for kw in keywords:
            explanation.append(f"'{kw}' ({weight_per_word}%)")

    # ── Ambiguity detection (threshold: 60 %) ───────────────────────────
    if confidence < 60:
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        status = (
            f"AMBIGUOUS ({sorted_domains[0][0].upper()} "
            f"vs {sorted_domains[1][0].upper()})"
        )
    else:
        status = "High confidence"

    return {
        "sentence": sentence,
        "domain": domain.upper(),
        "confidence": confidence,
        "status": status,
        "explanation": explanation,
    }


