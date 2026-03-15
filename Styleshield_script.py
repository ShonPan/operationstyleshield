"""
StyleShield: Bot Network Detection via Stylometric Anomaly Detection
=====================================================================
Adapted from Xenarch Mk14 (planetary technosignature detection).
Combined pipeline: base + enhanced extractor + enhanced clustering.

Core principle: Train on natural human writing diversity. Flag artificial
uniformity. A bot farm's output is the textual equivalent of a lunar lander
sitting in a field of craters — it doesn't belong, and a properly tuned
anomaly detector will find it.

Scoring formula:
    B = 0.20·VU + 0.45·SR + 0.15·HF + 0.15·TC + 0.05·CA

Xenarch → StyleShield metric mapping:
    MSE (recon error)   → Vocabulary Uniformity (VU)
    Edge Regularity     → Structural Regularity  (SR)
    Latent Density      → Hedging & Filler Sig.  (HF)
    Contextual Metric   → Temporal-Contextual    (TC)
    Gradient Anomaly    → Cross-Account Corr.    (CA)

Usage:
    from Styleshield_script import StyleShieldScorer

    # From dict
    scorer = StyleShieldScorer()
    results, clusters = scorer.analyze_accounts(accounts_dict)

    # From CSV (auto-detects long or wide format)
    results, clusters = scorer.analyze_csv(["training.csv"], ["test.csv"])

    # Generate a blank CSV template
    StyleShieldScorer().loader.save_template("template.csv")

CSV formats supported
---------------------
Long format (one post per row) — recommended:
    account_id, post_text, posting_hour (optional)

Wide format (one account per row):
    account_id, post_1, post_2, ..., posting_hours (optional, semicolon-separated)

Column names are matched case-insensitively and common aliases are accepted
(user_id, username, handle, text, content, tweet, timestamp, created_at, etc.)
"""

import re
import sys
import json
import math
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# ============================================================
# CONSTANTS: LLM SIGNATURES & KEYBOARD LAYOUT
# ============================================================

QWERTY_NEIGHBORS = {
    'q': set('wa'), 'w': set('qeas'), 'e': set('wrds'), 'r': set('etf'),
    't': set('ryg'), 'y': set('tuh'), 'u': set('yij'), 'i': set('uok'),
    'o': set('ipl'), 'p': set('ol'), 'a': set('qwsz'), 's': set('awedxz'),
    'd': set('serfcx'), 'f': set('drtgvc'), 'g': set('ftyhbv'), 'h': set('gyujnb'),
    'j': set('huiknm'), 'k': set('jiolm'), 'l': set('kop'), 'z': set('asx'),
    'x': set('zsdc'), 'c': set('xdfv'), 'v': set('cfgb'), 'b': set('vghn'),
    'n': set('bhjm'), 'm': set('njk'),
}

GPT4_MARKERS = [
    "certainly", "certainly!", "absolutely", "it's worth noting",
    "it's important to note", "it's worth mentioning", "furthermore",
    "moreover", "in terms of", "when it comes to", "that being said",
    "having said that", "it's also worth", "on the other hand",
    "at the end of the day", "in a nutshell",
]

CLAUDE_MARKERS = [
    "i'd be happy to", "that's a great question", "great question",
    "i think", "i believe", "it seems", "it appears", "to be honest",
    "honestly", "frankly", "i should note", "i should mention",
    "i appreciate", "happy to help", "let me", "shall i",
    "i understand", "that makes sense",
]

LLAMA_MARKERS = [
    "as an ai", "as a language model", "as an artificial intelligence",
    "i don't have personal", "i cannot", "i'm not able to",
    "please note that", "it's essential to", "keep in mind",
    "here's the thing", "let's dive in", "buckle up",
]

GENERIC_LLM_MARKERS = [
    "delve", "delving", "tapestry", "landscape", "multifaceted",
    "nuanced", "comprehensive", "robust", "leverage", "utilize",
    "facilitate", "implement", "paradigm", "synergy", "holistic",
    "cutting-edge", "game-changer", "groundbreaking",
]

EPISTEMIC_QUALIFIERS = [
    "perhaps", "possibly", "probably", "likely", "unlikely",
    "might", "could", "may", "seems", "appears",
    "arguably", "presumably", "supposedly", "allegedly",
    "in my opinion", "from my perspective",
]

COMMON_WORDS_1000 = set([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
    "had", "did", "got", "am", "very", "much", "more", "still", "really",
    "right", "too", "here", "thing", "things", "going", "been", "being",
    "does", "don't", "didn't", "won't", "can't", "isn't", "aren't", "wasn't",
    "it's", "i'm", "i've", "i'll", "i'd", "that's", "there's", "what's",
    "let's", "he's", "she's", "we're", "they're", "you're", "you've",
])


def count_syllables(word):
    word = word.lower().strip()
    if len(word) <= 2:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# Features that best separate bots from humans (used for clustering)
DISCRIMINATING_FEATURES = [
    # LLM signature (strongest separators)
    "llm_phrase_density",
    "gpt4_marker_rate",
    "claude_marker_rate",
    "model_prob_human",
    # Human informality signals
    "typo_rate",
    "contraction_rate",
    "all_caps_rate",
    # Vocabulary complexity
    "avg_syllables",
    "jargon_density",
    # Structure
    "paragraph_rhythm_score",
    "sentence_length_variance",
    "opener_diversity",
    # Intra-account consistency (bots = low variance)
    "intra_vocab_variance",
    "intra_structure_variance",
    "intra_hedge_variance",
    "intra_punctuation_variance",
    "intra_jargon_variance",
]

# Features used for stealth bot second-pass detection
STEALTH_FEATURES = [
    # Naturalness metrics (stealth bots are "too clean casual")
    "formality_variance",
    "naturalness_score",
    # Character-level fingerprint
    "char_trigram_entropy",
    "punctuation_sequence_entropy",
    # Structural micro-patterns
    "sentence_length_variance",
    "paragraph_rhythm_score",
    "opener_diversity",
    "intra_vocab_variance",
    "intra_structure_variance",
    "intra_hedge_variance",
    "intra_punctuation_variance",
    "intra_jargon_variance",
    "intra_typo_variance",
    # Vocabulary fingerprint
    "type_token_ratio",
    "avg_syllables",
    "contraction_rate",
    "avg_word_length",
]


# ============================================================
# 1. STYLOMETRIC FEATURE EXTRACTOR (BASE + ENHANCED)
# ============================================================

class StylometricExtractor:
    """
    Extracts writing fingerprint features from a corpus of posts.
    Combines base Xenarch-adapted features with enhanced LLM signature
    detection, typo analysis, sentence openers, and temporal features.
    """

    HEDGE_PHRASES = [
        "i think", "i believe", "in my opinion", "it seems", "it appears",
        "arguably", "perhaps", "possibly", "probably", "likely",
        "certainly", "absolutely", "definitely", "of course", "to be honest",
        "honestly", "frankly", "needless to say", "it's worth noting",
        "it's important to note", "it's worth mentioning", "as an ai",
        "as a language model", "i'd be happy to", "great question",
        "that's a great", "fascinating", "delve", "certainly!", "sure!"
    ]

    TRANSITION_WORDS = [
        "furthermore", "moreover", "additionally", "however", "nevertheless",
        "consequently", "therefore", "thus", "hence", "in conclusion",
        "in summary", "to summarize", "overall", "ultimately", "notably",
        "importantly", "significantly", "interestingly"
    ]

    def extract(self, posts: List[str], posting_hours: Optional[List[int]] = None) -> Dict:
        if not posts:
            return self._empty_fingerprint()

        full_text = " ".join(posts)
        full_lower = full_text.lower()
        tokens = self._tokenize(full_text)
        tokens_lower = [t.lower() for t in tokens]
        sentences = self._split_sentences(full_text)
        per_post = [self._extract_single(p) for p in posts if p.strip()]
        per_post_enhanced = [self._extract_post_enhanced(p) for p in posts if p.strip()]

        features = {}

        # ---- BASE FEATURES (Xenarch-adapted) ----

        # Vocabulary (→ VU)
        features["type_token_ratio"] = self._type_token_ratio(tokens)
        features["vocabulary_size"] = len(set(t.lower() for t in tokens))
        features["hapax_ratio"] = self._hapax_ratio(tokens)
        features["top10_concentration"] = self._top_n_concentration(tokens, 10)
        features["avg_word_length"] = float(np.mean([len(t) for t in tokens])) if tokens else 0.0

        # Structure (→ SR)
        features["avg_sentence_length"] = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0
        features["sentence_length_variance"] = float(np.var([len(s.split()) for s in sentences])) if sentences else 0.0
        features["avg_post_length"] = float(np.mean([len(p.split()) for p in posts]))
        features["post_length_variance"] = float(np.var([len(p.split()) for p in posts]))
        features["paragraph_rhythm_score"] = self._paragraph_rhythm(posts)

        # Hedging (→ HF)
        features["hedge_rate"] = self._phrase_rate(full_lower, self.HEDGE_PHRASES)
        features["transition_rate"] = self._phrase_rate(full_lower, self.TRANSITION_WORDS)
        features["exclamation_rate"] = full_text.count("!") / max(len(sentences), 1)
        features["question_rate"] = full_text.count("?") / max(len(sentences), 1)
        features["contraction_rate"] = self._contraction_rate(full_text)

        # Punctuation DNA
        features["comma_rate"] = full_text.count(",") / max(len(tokens), 1)
        features["semicolon_rate"] = full_text.count(";") / max(len(tokens), 1)
        features["dash_rate"] = (full_text.count("—") + full_text.count(" - ")) / max(len(sentences), 1)
        features["ellipsis_rate"] = full_text.count("...") / max(len(sentences), 1)
        features["emoji_rate"] = self._emoji_rate(full_text, len(tokens))

        # Intra-account variance (bots are unnaturally consistent)
        features["intra_vocab_variance"] = self._intra_var(per_post, "type_token_ratio")
        features["intra_structure_variance"] = self._intra_var(per_post, "avg_sentence_length")
        features["intra_hedge_variance"] = self._intra_var(per_post, "hedge_rate")

        # Meta
        features["post_count"] = len(posts)
        features["total_tokens"] = len(tokens)

        # ---- ENHANCED FEATURES ----

        # LLM Model Signatures
        features["gpt4_marker_rate"] = self._marker_rate(full_lower, GPT4_MARKERS, len(tokens))
        features["claude_marker_rate"] = self._marker_rate(full_lower, CLAUDE_MARKERS, len(tokens))
        features["llama_marker_rate"] = self._marker_rate(full_lower, LLAMA_MARKERS, len(tokens))
        features["generic_llm_rate"] = self._marker_rate(full_lower, GENERIC_LLM_MARKERS, len(tokens))
        features["epistemic_qualifier_rate"] = self._marker_rate(full_lower, EPISTEMIC_QUALIFIERS, len(tokens))

        # Individual high-signal markers
        features["certainly_rate"] = full_lower.count("certainly") / max(len(sentences), 1)
        features["furthermore_rate"] = full_lower.count("furthermore") / max(len(sentences), 1)
        features["delve_rate"] = full_lower.count("delve") / max(len(tokens), 1)
        features["moreover_rate"] = full_lower.count("moreover") / max(len(sentences), 1)

        # Combined LLM density
        features["llm_phrase_density"] = (
            features["gpt4_marker_rate"] +
            features["claude_marker_rate"] +
            features["llama_marker_rate"] +
            features["generic_llm_rate"]
        )

        # Model probability
        model_scores = {
            "gpt4": features["gpt4_marker_rate"],
            "claude": features["claude_marker_rate"],
            "llama": features["llama_marker_rate"],
            "human": max(0.01, 1.0 - features["llm_phrase_density"]),
        }
        total = sum(model_scores.values()) or 1
        features["model_prob_gpt4"] = model_scores["gpt4"] / total
        features["model_prob_claude"] = model_scores["claude"] / total
        features["model_prob_llama"] = model_scores["llama"] / total
        features["model_prob_human"] = model_scores["human"] / total

        # Sentence Opener Analysis
        openers = []
        for s in sentences:
            words = s.strip().split()
            if words:
                openers.append(words[0].lower())

        if openers:
            opener_freq = Counter(openers)
            features["opener_diversity"] = len(set(openers)) / max(len(openers), 1)
            features["opener_repetition_rate"] = max(opener_freq.values()) / len(openers)
            features["question_opener_rate"] = sum(
                1 for o in openers if o in (
                    "what", "why", "how", "when", "where", "who",
                    "is", "are", "do", "does", "can", "could", "would", "should"
                )
            ) / max(len(openers), 1)
            features["i_opener_rate"] = sum(1 for o in openers if o == "i") / max(len(openers), 1)
            features["the_opener_rate"] = sum(1 for o in openers if o == "the") / max(len(openers), 1)
        else:
            features["opener_diversity"] = 0
            features["opener_repetition_rate"] = 0
            features["question_opener_rate"] = 0
            features["i_opener_rate"] = 0
            features["the_opener_rate"] = 0

        features["list_marker_rate"] = len(re.findall(r'(?:^|\n)\s*[\-\*\d]+[\.\)]\s', full_text)) / max(len(sentences), 1)

        # Vocabulary: Jargon & Syllables
        if tokens_lower:
            uncommon = [t for t in tokens_lower if t not in COMMON_WORDS_1000 and len(t) > 3]
            features["jargon_density"] = len(uncommon) / max(len(tokens_lower), 1)
            features["avg_syllables"] = float(np.mean([count_syllables(t) for t in tokens_lower]))
            freq = Counter(tokens_lower)
            sorted_freqs = sorted(freq.values(), reverse=True)
            if len(sorted_freqs) > 1:
                top_ratio = sorted_freqs[0] / max(sorted_freqs[-1], 1)
                features["zipf_ratio"] = min(top_ratio / 50.0, 1.0)
            else:
                features["zipf_ratio"] = 0
        else:
            features["jargon_density"] = 0
            features["avg_syllables"] = 0
            features["zipf_ratio"] = 0

        # Extended Punctuation
        features["parenthetical_rate"] = (full_text.count("(") + full_text.count(")")) / max(len(tokens), 1)
        features["colon_rate"] = full_text.count(":") / max(len(sentences), 1)
        features["all_caps_rate"] = sum(1 for t in tokens if t.isupper() and len(t) > 1) / max(len(tokens), 1)

        # Typo / Device Fingerprinting
        typo_data = self._analyze_typos(full_text, tokens)
        features["typo_rate"] = typo_data["typo_rate"]
        features["adjacent_key_error_rate"] = typo_data["adjacent_key_rate"]
        features["dropped_char_rate"] = typo_data["dropped_char_rate"]
        features["double_char_rate"] = typo_data["double_char_rate"]
        mobile_signal = typo_data["adjacent_key_rate"] * 2 + typo_data["dropped_char_rate"]
        features["mobile_device_signal"] = float(np.clip(mobile_signal, 0, 1))

        # Temporal / Shift Detection
        if posting_hours and len(posting_hours) >= 2:
            hours = np.array(posting_hours)
            features["posting_hour_mean"] = float(np.mean(hours))
            features["posting_hour_variance"] = float(np.var(hours))
            bins = np.zeros(3)
            for h in hours:
                bins[int(h) // 8] += 1
            bins /= max(bins.sum(), 1)
            features["shift_concentration"] = float(np.max(bins))
            features["dominant_shift"] = int(np.argmax(bins))
            biz = sum(1 for h in hours if 9 <= h <= 17) / len(hours)
            features["business_hours_ratio"] = biz
            features["posting_regularity"] = 1.0 / (1.0 + float(np.var(hours)) / 50.0)
            mode_hour = int(Counter(posting_hours).most_common(1)[0][0])
            tz_offset = (mode_hour - 19) % 24
            if tz_offset > 12:
                tz_offset -= 24
            features["inferred_tz_offset"] = tz_offset
        else:
            features["posting_hour_mean"] = 12.0
            features["posting_hour_variance"] = 50.0
            features["shift_concentration"] = 0.33
            features["dominant_shift"] = 1
            features["business_hours_ratio"] = 0.5
            features["posting_regularity"] = 0.5
            features["inferred_tz_offset"] = 0

        # Enhanced Intra-Account Variance
        if len(per_post_enhanced) >= 2:
            features["intra_punctuation_variance"] = float(np.var([p["comma_rate"] for p in per_post_enhanced]))
            features["intra_opener_variance"] = float(np.var([p["opener_diversity"] for p in per_post_enhanced]))
            features["intra_jargon_variance"] = float(np.var([p["jargon_density"] for p in per_post_enhanced]))
            features["intra_syllable_variance"] = float(np.var([p["avg_syllables"] for p in per_post_enhanced]))
            features["intra_typo_variance"] = float(np.var([p["typo_rate"] for p in per_post_enhanced]))
        else:
            features["intra_punctuation_variance"] = 1.0
            features["intra_opener_variance"] = 1.0
            features["intra_jargon_variance"] = 1.0
            features["intra_syllable_variance"] = 1.0
            features["intra_typo_variance"] = 1.0

        # ---- STEALTH BOT DETECTION FEATURES ----

        # Formality variance: real humans shift formality across posts;
        # LLM personas maintain consistent formality level
        per_post_formality = []
        for post in posts:
            if not post.strip():
                continue
            p_tokens = re.findall(r"\b[a-zA-Z']+\b", post)
            if not p_tokens:
                continue
            # Formality signals: contractions (informal), long words (formal),
            # slang markers (informal), transition words (formal)
            contractions = len(re.findall(r"\b\w+'\w+\b", post)) / max(len(p_tokens), 1)
            long_words = sum(1 for t in p_tokens if len(t) > 8) / max(len(p_tokens), 1)
            slang = len(re.findall(
                r'\b(u|ur|r|n|thru|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|lol|omg|tbh|idk|imo|smh|btw|ngl|rn|fr|ong|lowkey|highkey|bruh|fam|lit|bet|slay|cap|deadass)\b',
                post.lower()
            )) / max(len(p_tokens), 1)
            transitions = sum(post.lower().count(t) for t in self.TRANSITION_WORDS) / max(len(p_tokens) / 10, 1)
            formality = (long_words + transitions) - (contractions + slang)
            per_post_formality.append(formality)

        if len(per_post_formality) >= 2:
            features["formality_variance"] = float(np.var(per_post_formality))
            features["formality_range"] = float(max(per_post_formality) - min(per_post_formality))
        else:
            features["formality_variance"] = 0.1
            features["formality_range"] = 0.2

        # Naturalness score: combines multiple signals that separate
        # "genuine casual" from "LLM-generated casual"
        # Real casual text has: varied sentence lengths, occasional typos,
        # inconsistent punctuation, mixed formality
        genuine_typo = features.get("typo_rate", 0) > 0
        has_variance = features.get("intra_structure_variance", 0) > 2.0
        mixed_formality = features.get("formality_variance", 0) > 0.005
        low_rhythm = features.get("paragraph_rhythm_score", 1.0) < 0.7
        naturalness = sum([
            0.25 * float(genuine_typo),
            0.25 * float(has_variance),
            0.25 * float(mixed_formality),
            0.25 * float(low_rhythm),
        ])
        features["naturalness_score"] = naturalness

        # Character-level trigram entropy: catches shared LLM "writing DNA"
        # Same model produces similar character-level distributions
        features["char_trigram_entropy"] = self._char_trigram_entropy(full_text)

        # Punctuation sequence entropy: how predictable is punctuation ordering?
        # LLMs produce more regular punctuation patterns
        features["punctuation_sequence_entropy"] = self._punctuation_entropy(full_text)

        return features

    # ---- Base helpers ----

    def _tokenize(self, text):
        return re.findall(r"\b[a-zA-Z']+\b", text)

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]

    def _type_token_ratio(self, tokens):
        if not tokens:
            return 0.0
        w = min(len(tokens), 100)
        return len(set(t.lower() for t in tokens[:w])) / w

    def _hapax_ratio(self, tokens):
        if not tokens:
            return 0.0
        freq = Counter(t.lower() for t in tokens)
        return sum(1 for v in freq.values() if v == 1) / len(freq)

    def _top_n_concentration(self, tokens, n):
        if not tokens:
            return 0.0
        freq = Counter(t.lower() for t in tokens)
        return sum(v for _, v in freq.most_common(n)) / len(tokens)

    def _phrase_rate(self, text, phrases):
        words = text.split()
        if not words:
            return 0.0
        return sum(text.count(p) for p in phrases) / max(len(words) / 10, 1)

    def _marker_rate(self, text, markers, token_count):
        if token_count == 0:
            return 0.0
        hits = sum(text.count(p) for p in markers)
        return hits / max(token_count / 10, 1)

    def _contraction_rate(self, text):
        contractions = re.findall(r"\b\w+'\w+\b", text)
        return len(contractions) / max(len(text.split()), 1)

    def _emoji_rate(self, text, token_count):
        return sum(1 for c in text if ord(c) > 127000) / max(token_count, 1)

    def _paragraph_rhythm(self, posts):
        if len(posts) < 2:
            return 0.0
        lengths = [len(p.split()) for p in posts]
        mean = np.mean(lengths)
        if mean == 0:
            return 0.0
        return float(max(0.0, 1.0 - np.std(lengths) / mean))

    def _extract_single(self, post):
        tokens = self._tokenize(post)
        sentences = self._split_sentences(post)
        return {
            "type_token_ratio":    self._type_token_ratio(tokens),
            "avg_sentence_length": float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0,
            "hedge_rate":          self._phrase_rate(post.lower(), self.HEDGE_PHRASES),
        }

    def _extract_post_enhanced(self, post):
        """Per-post features for enhanced intra-account variance."""
        tokens = re.findall(r"\b[a-zA-Z']+\b", post)
        tokens_lower = [t.lower() for t in tokens]
        sentences = [s.strip() for s in re.split(r'[.!?]+', post) if len(s.strip()) > 3]
        openers = []
        for s in sentences:
            w = s.strip().split()
            if w:
                openers.append(w[0].lower())
        uncommon = [t for t in tokens_lower if t not in COMMON_WORDS_1000 and len(t) > 3]
        return {
            "comma_rate": post.count(",") / max(len(tokens), 1),
            "opener_diversity": len(set(openers)) / max(len(openers), 1) if openers else 0,
            "jargon_density": len(uncommon) / max(len(tokens_lower), 1) if tokens_lower else 0,
            "avg_syllables": float(np.mean([count_syllables(t) for t in tokens_lower])) if tokens_lower else 0,
            "typo_rate": len(re.findall(
                r'\b(u|ur|r|n|thru|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|lol|omg|tbh|idk|imo|smh|btw)\b',
                post.lower()
            )) / max(len(tokens), 1),
        }

    def _analyze_typos(self, text, tokens):
        words = text.split()
        if not words:
            return {"typo_rate": 0, "adjacent_key_rate": 0, "dropped_char_rate": 0, "double_char_rate": 0}
        adjacent_key_errors = 0
        dropped_chars = 0
        double_chars = 0
        for word in words:
            w = word.lower().strip(".,!?;:'\"()[]{}@#$%^&*")
            if len(w) < 2:
                continue
            common_doubles = set("lseptofn")
            for i in range(len(w) - 1):
                if w[i] == w[i+1] and w[i].isalpha() and w[i] not in common_doubles:
                    double_chars += 1
            for i in range(len(w) - 1):
                c1, c2 = w[i], w[i+1]
                if c1 in QWERTY_NEIGHBORS and c2 in QWERTY_NEIGHBORS.get(c1, set()):
                    adjacent_key_errors += 0.1
        total_words = len(words)
        informal_markers = len(re.findall(
            r'\b(u|ur|r|n|thru|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|lol|omg|tbh|idk|imo|smh|btw)\b',
            text.lower()
        ))
        return {
            "typo_rate": informal_markers / max(total_words, 1),
            "adjacent_key_rate": adjacent_key_errors / max(total_words, 1),
            "dropped_char_rate": dropped_chars / max(total_words, 1),
            "double_char_rate": double_chars / max(total_words, 1),
        }

    def _char_trigram_entropy(self, text):
        """Shannon entropy of character trigram distribution."""
        text = text.lower()
        if len(text) < 3:
            return 0.0
        trigrams = Counter()
        for i in range(len(text) - 2):
            trigrams[text[i:i+3]] += 1
        total = sum(trigrams.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in trigrams.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _punctuation_entropy(self, text):
        """Entropy of punctuation character sequence."""
        punct_seq = [c for c in text if c in '.,!?;:—-()[]"\'']
        if len(punct_seq) < 3:
            return 0.0
        bigrams = Counter()
        for i in range(len(punct_seq) - 1):
            bigrams[punct_seq[i] + punct_seq[i+1]] += 1
        total = sum(bigrams.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in bigrams.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _intra_var(self, per_post, key):
        vals = [p[key] for p in per_post if key in p]
        return float(np.var(vals)) if len(vals) >= 2 else 1.0

    def _empty_fingerprint(self):
        base_keys = [
            "type_token_ratio", "vocabulary_size", "hapax_ratio", "top10_concentration",
            "avg_word_length", "avg_sentence_length", "sentence_length_variance",
            "avg_post_length", "post_length_variance", "paragraph_rhythm_score",
            "hedge_rate", "transition_rate", "exclamation_rate", "question_rate",
            "contraction_rate", "comma_rate", "semicolon_rate", "dash_rate",
            "ellipsis_rate", "emoji_rate", "intra_vocab_variance",
            "intra_structure_variance", "intra_hedge_variance", "post_count", "total_tokens",
        ]
        enhanced_keys = [
            "gpt4_marker_rate", "claude_marker_rate", "llama_marker_rate",
            "generic_llm_rate", "epistemic_qualifier_rate", "certainly_rate",
            "furthermore_rate", "delve_rate", "moreover_rate", "llm_phrase_density",
            "model_prob_gpt4", "model_prob_claude", "model_prob_llama", "model_prob_human",
            "opener_diversity", "opener_repetition_rate", "question_opener_rate",
            "i_opener_rate", "the_opener_rate", "list_marker_rate",
            "jargon_density", "avg_syllables", "zipf_ratio",
            "parenthetical_rate", "colon_rate", "all_caps_rate",
            "typo_rate", "adjacent_key_error_rate", "dropped_char_rate",
            "double_char_rate", "mobile_device_signal",
            "posting_hour_mean", "posting_hour_variance", "shift_concentration",
            "dominant_shift", "business_hours_ratio", "posting_regularity",
            "inferred_tz_offset",
            "intra_punctuation_variance", "intra_opener_variance",
            "intra_jargon_variance", "intra_syllable_variance", "intra_typo_variance",
        ]
        stealth_keys = [
            "formality_variance", "formality_range", "naturalness_score",
            "char_trigram_entropy", "punctuation_sequence_entropy",
        ]
        return {k: 0.0 for k in base_keys + enhanced_keys + stealth_keys}


# ============================================================
# 2. MULTI-METRIC BOT SCORER
# ============================================================

class MultiMetricBotScorer:
    """
    Five normalized metrics combined into a single bot score.
    Weights updated from script-1: 0.20·VU + 0.45·SR + 0.15·HF + 0.15·TC + 0.05·CA
    """

    def compute_vocabulary_uniformity(self, fp):
        ttr_signal     = 1.0 - fp.get("type_token_ratio", 0.5)
        concentration  = fp.get("top10_concentration", 0.3)
        hapax_signal   = 1.0 - fp.get("hapax_ratio", 0.5)
        intra_var      = fp.get("intra_vocab_variance", 0.01)
        consistency    = 1.0 / (1.0 + intra_var * 100)
        return float(np.clip(
            0.30 * ttr_signal + 0.25 * concentration + 0.25 * hapax_signal + 0.20 * consistency,
            0, 1
        ))

    def compute_structural_regularity(self, fp):
        sent_var       = fp.get("sentence_length_variance", 10.0)
        rhythm_signal  = 1.0 / (1.0 + sent_var / 20.0)
        post_var       = fp.get("post_length_variance", 100.0)
        template       = 1.0 / (1.0 + post_var / 500.0)
        rhythm_unif    = fp.get("paragraph_rhythm_score", 0.0)
        intra_var      = fp.get("intra_structure_variance", 5.0)
        intra_signal   = 1.0 / (1.0 + intra_var / 10.0)
        return float(np.clip(
            0.30 * rhythm_signal + 0.25 * template + 0.25 * rhythm_unif + 0.20 * intra_signal,
            0, 1
        ))

    def compute_hedging_signature(self, fp):
        hedge        = fp.get("hedge_rate", 0.0)
        transition   = fp.get("transition_rate", 0.0)
        llm_signal   = float(np.clip((hedge + transition) / 0.5, 0, 1))
        excl         = fp.get("exclamation_rate", 0.0)
        enthusiasm   = float(np.clip(excl / 2.0, 0, 1))
        no_contraction = 1.0 - float(np.clip(fp.get("contraction_rate", 0.05) / 0.1, 0, 1))
        intra_var    = fp.get("intra_hedge_variance", 0.001)
        consistency  = 1.0 / (1.0 + intra_var * 1000)
        return float(np.clip(
            0.35 * llm_signal + 0.20 * enthusiasm + 0.25 * no_contraction + 0.20 * consistency,
            0, 1
        ))

    def compute_temporal_contextual(self, posting_hours=None):
        if not posting_hours or len(posting_hours) < 3:
            return 0.5
        hours = np.array(posting_hours)
        bins = np.zeros(3)
        for h in hours:
            bins[int(h) // 8] += 1
        bins /= bins.sum()
        shift_concentration = float(np.max(bins))
        morning = float(np.sum((hours >= 8) & (hours <= 10)) / len(hours))
        evening = float(np.sum((hours >= 18) & (hours <= 22)) / len(hours))
        bot_pattern = 1.0 - float(np.clip(morning + evening, 0, 1))
        variance_signal = 1.0 / (1.0 + float(np.var(hours)) / 50.0)
        return float(np.clip(
            0.40 * shift_concentration + 0.30 * bot_pattern + 0.30 * variance_signal,
            0, 1
        ))

    def compute_cross_account_correlation(self, account_id, fingerprints):
        if len(fingerprints) < 2 or account_id not in fingerprints:
            return 0.0
        keys = [
            "type_token_ratio", "hapax_ratio", "top10_concentration",
            "avg_sentence_length", "sentence_length_variance",
            "hedge_rate", "transition_rate", "contraction_rate",
            "comma_rate", "paragraph_rhythm_score"
        ]
        def to_vec(fp):
            return np.array([fp.get(k, 0.0) for k in keys])

        all_ids  = list(fingerprints.keys())
        all_vecs = np.array([to_vec(fingerprints[a]) for a in all_ids])
        norms    = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        all_vecs /= norms

        idx         = all_ids.index(account_id)
        target      = all_vecs[idx:idx+1]
        others      = np.delete(all_vecs, idx, axis=0)
        similarities = cosine_similarity(target, others)[0]
        return float(np.clip(0.6 * similarities.max() + 0.4 * similarities.mean(), 0, 1))

    def compute_combined_bot_score(self, fp, fingerprints, account_id, posting_hours=None):
        vu = self.compute_vocabulary_uniformity(fp)
        sr = self.compute_structural_regularity(fp)
        hf = self.compute_hedging_signature(fp)
        tc = self.compute_temporal_contextual(posting_hours)
        ca = self.compute_cross_account_correlation(account_id, fingerprints)

        combined = 0.20*vu + 0.45*sr + 0.15*hf + 0.15*tc + 0.05*ca

        metrics = {
            "vocabulary_uniformity":     round(vu, 4),
            "structural_regularity":     round(sr, 4),
            "hedging_signature":         round(hf, 4),
            "temporal_contextual":       round(tc, 4),
            "cross_account_correlation": round(ca, 4),
            "bot_score":                 round(float(combined), 4),
        }
        return float(combined), metrics


# ============================================================
# 3. ENHANCED DBSCAN CLUSTER ANALYSIS
# ============================================================

class AccountClusterAnalyzer:
    """
    Identifies bot networks by clustering accounts in stylometric space.
    Uses StandardScaler + euclidean distance (from enhanced pipeline)
    for better separation than raw cosine distance.
    """

    FEATURE_KEYS = [
        "type_token_ratio", "hapax_ratio", "top10_concentration",
        "avg_word_length", "avg_sentence_length", "sentence_length_variance",
        "avg_post_length", "post_length_variance", "paragraph_rhythm_score",
        "hedge_rate", "transition_rate", "contraction_rate",
        "comma_rate", "semicolon_rate", "dash_rate",
        "intra_vocab_variance", "intra_structure_variance", "intra_hedge_variance"
    ]

    def __init__(self, epsilon=1.5, min_samples=2):
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.scaler = StandardScaler()

    def _build_matrix(self, fingerprints):
        ids = list(fingerprints.keys())
        mat = np.array([
            [fingerprints[a].get(k, 0.0) for k in DISCRIMINATING_FEATURES]
            for a in ids
        ])
        return mat, ids

    def compute_similarity_matrix(self, fingerprints):
        mat, ids = self._build_matrix(fingerprints)
        mat_std = self.scaler.fit_transform(mat)
        return cosine_similarity(mat_std), ids

    def cluster(self, fingerprints):
        if len(fingerprints) < 2:
            return pd.DataFrame([{"account_id": a, "cluster_id": -1, "is_noise": True}
                                  for a in fingerprints])
        mat, ids = self._build_matrix(fingerprints)
        mat_std = self.scaler.fit_transform(mat)
        labels = DBSCAN(
            eps=self.epsilon,
            min_samples=self.min_samples,
            metric="euclidean"
        ).fit_predict(mat_std)
        return pd.DataFrame([
            {"account_id": a, "cluster_id": int(l), "is_noise": l == -1}
            for a, l in zip(ids, labels)
        ])

    def detect_stealth_subclusters(self, cluster_df, fingerprints, test_account_ids):
        """
        Second-pass detection: look for stealth bot sub-clusters among
        test accounts that weren't flagged as obvious bots.

        Stealth bots evade pass 1 by mimicking human style, but they still
        share operator-level writing DNA that a tighter DBSCAN on stealth-specific
        features can catch.
        """
        # Candidates: all test accounts NOT already in known bot clusters
        known_bot_clusters = set()
        cluster_summary_temp = {}
        for cid in cluster_df[~cluster_df["is_noise"]]["cluster_id"].unique():
            members = cluster_df[cluster_df["cluster_id"] == int(cid)]["account_id"].tolist()
            fps = [fingerprints[a] for a in members if a in fingerprints]
            if fps:
                avg_llm = float(np.mean([fp.get("llm_phrase_density", 0) for fp in fps]))
                if avg_llm > 0.5:  # clearly LLM-generated clusters
                    known_bot_clusters.add(int(cid))

        candidates = set()
        for aid in test_account_ids:
            row = cluster_df[cluster_df["account_id"] == aid]
            if len(row) == 0:
                continue
            cid = int(row.iloc[0]["cluster_id"])
            if cid not in known_bot_clusters:
                candidates.add(aid)

        if len(candidates) < 3:
            return [], {}

        # Build stealth feature matrix
        candidate_list = sorted(candidates)
        stealth_keys = [k for k in STEALTH_FEATURES if k in fingerprints[candidate_list[0]]]
        mat = np.array([
            [fingerprints[a].get(k, 0.0) for k in stealth_keys]
            for a in candidate_list
        ])

        # Tighter DBSCAN on stealth features — try multiple epsilon values
        scaler2 = StandardScaler()
        mat_std = scaler2.fit_transform(mat)
        # Replace NaN from constant features
        mat_std = np.nan_to_num(mat_std, nan=0.0)

        # Try progressively tighter epsilon to find stealth sub-clusters
        best_labels = None
        best_n_clusters = 0
        for eps in [1.0, 0.8, 0.6]:
            labels = DBSCAN(
                eps=eps,
                min_samples=2,
                metric="euclidean"
            ).fit_predict(mat_std)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > best_n_clusters:
                best_n_clusters = n_clusters
                best_labels = labels
        labels = best_labels if best_labels is not None else np.full(len(candidate_list), -1)

        subclusters = {}
        for label in set(labels):
            if label == -1:
                continue
            members = [candidate_list[i] for i, l in enumerate(labels) if l == label]
            if len(members) < 2:
                continue

            # Compute stealth coordination score
            member_mat = mat_std[[i for i, l in enumerate(labels) if l == label]]
            dists = euclidean_distances(member_mat)
            avg_dist = dists[np.triu_indices_from(dists, k=1)].mean()
            coordination = max(0, 1.0 - avg_dist / 3.0)

            # Naturalness analysis
            avg_naturalness = float(np.mean([
                fingerprints[m].get("naturalness_score", 1.0) for m in members
            ]))
            avg_formality_var = float(np.mean([
                fingerprints[m].get("formality_variance", 0.1) for m in members
            ]))

            # Cross-member style similarity (pairwise on ALL features)
            all_feat_keys = list(DISCRIMINATING_FEATURES) + stealth_keys
            all_feat_keys = list(dict.fromkeys(all_feat_keys))  # dedupe
            pair_mat = np.array([
                [fingerprints[m].get(k, 0.0) for k in all_feat_keys]
                for m in members
            ])
            pair_std = StandardScaler().fit_transform(pair_mat)
            # Replace NaN with 0 (constant features produce NaN after scaling)
            pair_std = np.nan_to_num(pair_std, nan=0.0)
            pair_sim = cosine_similarity(pair_std)
            pair_sim = np.nan_to_num(pair_sim, nan=0.0)
            avg_pair_sim = pair_sim[np.triu_indices_from(pair_sim, k=1)].mean()

            # Stealth bot score: high coordination + low naturalness + high pair similarity
            stealth_score = (
                0.35 * coordination +
                0.25 * (1.0 - avg_naturalness) +
                0.25 * float(np.clip(avg_pair_sim, 0, 1)) +
                0.15 * (1.0 - float(np.clip(avg_formality_var * 20, 0, 1)))
            )

            is_stealth = stealth_score > 0.45 and len(members) >= 2

            subclusters[label] = {
                "members": members,
                "member_count": len(members),
                "stealth_score": round(stealth_score, 4),
                "coordination": round(coordination, 4),
                "avg_naturalness": round(avg_naturalness, 4),
                "avg_formality_variance": round(avg_formality_var, 6),
                "avg_pairwise_similarity": round(float(avg_pair_sim), 4),
                "is_stealth_network": is_stealth,
            }

        stealth_accounts = []
        for info in subclusters.values():
            if info["is_stealth_network"]:
                stealth_accounts.extend(info["members"])

        return stealth_accounts, subclusters

    def describe_clusters(self, cluster_df, fingerprints):
        summary = {}
        for cid in cluster_df[~cluster_df["is_noise"]]["cluster_id"].unique():
            cid = int(cid)
            members = cluster_df[cluster_df["cluster_id"] == cid]["account_id"].tolist()
            fps = [fingerprints[a] for a in members if a in fingerprints]
            if not fps:
                continue

            # Model identification per cluster
            model_votes = Counter()
            for fp in fps:
                model_probs = {
                    "gpt4": fp.get("model_prob_gpt4", 0),
                    "claude": fp.get("model_prob_claude", 0),
                    "llama": fp.get("model_prob_llama", 0),
                    "human": fp.get("model_prob_human", 0),
                }
                model_votes[max(model_probs, key=model_probs.get)] += 1

            # Coordination signal via standardized euclidean distance
            member_mat = np.array([
                [fingerprints[m].get(k, 0.0) for k in DISCRIMINATING_FEATURES]
                for m in members
            ])
            if len(members) >= 2:
                member_std = self.scaler.transform(member_mat)
                dists = euclidean_distances(member_std)
                avg_dist = dists[np.triu_indices_from(dists, k=1)].mean()
                coordination = max(0, 1.0 - avg_dist / 5.0)
            else:
                coordination = 0.5

            avg_llm = float(np.mean([fp.get("llm_phrase_density", 0) for fp in fps]))
            avg_typo = float(np.mean([fp.get("typo_rate", 0) for fp in fps]))

            # Feature uniformity analysis (from base)
            shared = {}
            for key in self.FEATURE_KEYS:
                vals = [fp.get(key, 0.0) for fp in fps]
                shared[key] = {
                    "mean":       round(float(np.mean(vals)), 4),
                    "std":        round(float(np.std(vals)), 4),
                    "uniformity": round(1.0 / (1.0 + float(np.std(vals))), 4),
                }
            binding = sorted(shared.items(), key=lambda x: x[1]["std"])[:5]

            summary[cid] = {
                "member_count":       len(members),
                "members":            members,
                "dominant_model":     model_votes.most_common(1)[0][0],
                "model_distribution": dict(model_votes),
                "binding_features":   {k: v for k, v in binding},
                "coordination_signal": round(coordination, 4),
                "avg_llm_density":    round(avg_llm, 4),
                "avg_typo_rate":      round(avg_typo, 4),
                "is_bot_network":     avg_llm > 0.05 or coordination > 0.7,
            }
        return summary


# ============================================================
# 4. ADAPTIVE CONFIDENCE CALCULATOR
# ============================================================

class AdaptiveConfidenceCalculator:
    """
    Direct port of Xenarch's _compute_confidence().
    """

    def compute(self, bot_score, metrics, cluster_id, cluster_size, total_accounts):
        clustering_active = cluster_id >= 0 and cluster_size >= 2
        if clustering_active and total_accounts >= 10:
            membership = min(cluster_size / max(total_accounts * 0.1, 1), 1.0)
            confidence = (
                0.40 * bot_score +
                0.30 * membership +
                0.20 * metrics.get("structural_regularity", 0) +
                0.10 * self._agreement(metrics)
            )
            method = "clustering"
        else:
            confidence = (
                0.50 * bot_score +
                0.30 * metrics.get("hedging_signature", 0) +
                0.20 * metrics.get("vocabulary_uniformity", 0)
            )
            method = "individual"
        return float(np.clip(confidence, 0, 1)), method

    def _agreement(self, metrics):
        keys = ["vocabulary_uniformity", "structural_regularity",
                "hedging_signature", "cross_account_correlation"]
        return sum(1 for k in keys if metrics.get(k, 0) > 0.5) / len(keys)


# ============================================================
# 5. CSV DATA LOADER
# ============================================================

class CSVLoader:
    """
    Loads account/post data from CSV files.
    Supports long format (one post per row) and wide format (one account per row).
    Column name matching is case-insensitive with common aliases.
    """

    ACCOUNT_ALIASES   = ["account_id", "account", "user_id", "user", "username", "handle", "author", "name", "id"]
    POST_ALIASES      = ["post_text", "text", "content", "body", "post", "tweet", "message", "comment", "tweet_text"]
    HOUR_ALIASES      = ["posting_hour", "hour", "post_hour", "hour_of_day"]
    TIMESTAMP_ALIASES = ["post_timestamp", "timestamp", "created_at", "date", "datetime", "time", "posted_at", "tweet_created"]

    def load(self, csv_path: str) -> Dict:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
        print(f"  Columns: {list(df.columns)}")
        col_map = self._map_columns(df.columns.tolist())
        fmt     = self._detect_format(df, col_map)
        print(f"  Detected format: {fmt}")
        if fmt == "long":
            return self._load_long(df, col_map)
        elif fmt == "wide":
            return self._load_wide(df, col_map)
        else:
            raise ValueError(
                f"Cannot parse CSV '{path.name}'.\n"
                f"Expected columns like: {self.ACCOUNT_ALIASES[:3]} and {self.POST_ALIASES[:3]}.\n"
                f"Got: {list(df.columns)}\n"
                f"Run scorer.loader.save_template('template.csv') to see expected format."
            )

    def load_multiple(self, *csv_paths: str) -> Dict:
        merged = {}
        for path in csv_paths:
            for account_id, data in self.load(path).items():
                if account_id in merged:
                    merged[account_id]["posts"].extend(data["posts"])
                    merged[account_id]["posting_hours"].extend(data["posting_hours"])
                else:
                    merged[account_id] = data
        return merged

    def save_template(self, output_path: str, fmt: str = "long"):
        if fmt == "long":
            template = pd.DataFrame([
                {"account_id": "user_001", "post_text": "Example post one.",     "posting_hour": 9},
                {"account_id": "user_001", "post_text": "Example post two.",     "posting_hour": 10},
                {"account_id": "user_002", "post_text": "Another user's post.",  "posting_hour": 21},
            ])
        else:
            template = pd.DataFrame([
                {"account_id": "user_001", "post_1": "First post",  "post_2": "Second post", "posting_hours": "9;10"},
                {"account_id": "user_002", "post_1": "Only post",   "post_2": "",            "posting_hours": "21"},
            ])
        template.to_csv(output_path, index=False)
        print(f"Template saved: {output_path}")

    def _map_columns(self, columns):
        lower = {c.lower(): c for c in columns}
        def find(aliases):
            for a in aliases:
                if a.lower() in lower:
                    return lower[a.lower()]
            return None
        return {
            "account":   find(self.ACCOUNT_ALIASES),
            "post":      find(self.POST_ALIASES),
            "hour":      find(self.HOUR_ALIASES),
            "timestamp": find(self.TIMESTAMP_ALIASES),
        }

    def _detect_format(self, df, col_map):
        if col_map["account"] and col_map["post"]:
            return "long"
        post_cols = [c for c in df.columns if re.match(r"post_?\d+", c.lower())]
        if col_map["account"] and post_cols:
            return "wide"
        if len(df.columns) == 2:
            col_map["account"] = df.columns[0]
            col_map["post"]    = df.columns[1]
            return "long"
        return "unknown"

    def _load_long(self, df, col_map):
        account_col = col_map["account"]
        post_col    = col_map["post"]
        hour_col    = col_map["hour"]
        ts_col      = col_map["timestamp"]
        if not hour_col and ts_col:
            try:
                df = df.copy()
                df["_hour"] = pd.to_datetime(df[ts_col]).dt.hour
                hour_col = "_hour"
            except Exception:
                pass
        accounts = {}
        for account_id, group in df.groupby(account_col):
            posts = [p.strip() for p in group[post_col].dropna().astype(str).tolist() if p.strip()]
            hours = []
            if hour_col and hour_col in group.columns:
                for h in group[hour_col].dropna():
                    try:
                        hours.append(int(float(h)) % 24)
                    except (ValueError, TypeError):
                        pass
            accounts[str(account_id)] = {"posts": posts, "posting_hours": hours}
        print(f"  Parsed {len(accounts)} accounts (long format)")
        return accounts

    def _load_wide(self, df, col_map):
        account_col = col_map["account"]
        post_cols   = sorted(
            [c for c in df.columns if re.match(r"post_?\d+", c.lower())],
            key=lambda c: int(re.search(r"\d+", c).group())
        )
        hours_col = next((c for c in df.columns if "posting_hours" in c.lower()), None)
        accounts = {}
        for _, row in df.iterrows():
            account_id = str(row[account_col])
            posts = [str(row[c]).strip() for c in post_cols
                     if pd.notna(row[c]) and str(row[c]).strip()]
            hours = []
            if hours_col and pd.notna(row.get(hours_col)):
                for h in re.split(r"[;,|]", str(row[hours_col])):
                    try:
                        hours.append(int(h.strip()) % 24)
                    except ValueError:
                        pass
            accounts[account_id] = {"posts": posts, "posting_hours": hours}
        print(f"  Parsed {len(accounts)} accounts (wide format)")
        return accounts


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

class StyleShieldScorer:
    """
    End-to-end bot network detection pipeline.
    Combines base scoring with enhanced feature extraction and clustering.
    """

    def __init__(self, dbscan_epsilon=1.5, dbscan_min_samples=2):
        self.extractor        = StylometricExtractor()
        self.scorer           = MultiMetricBotScorer()
        self.clusterer        = AccountClusterAnalyzer(epsilon=dbscan_epsilon,
                                                       min_samples=dbscan_min_samples)
        self.confidence_calc  = AdaptiveConfidenceCalculator()
        self.loader           = CSVLoader()

    def analyze_csv(self, train_paths: List[str], test_paths: List[str]) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Load separate training and testing CSV files."""
        print(f"\nLoading training data from {len(train_paths)} file(s)...")
        train_accounts = self.loader.load_multiple(*train_paths) if train_paths else {}

        print(f"\nLoading testing data from {len(test_paths)} file(s)...")
        test_accounts = self.loader.load_multiple(*test_paths)

        print(f"\nTotal training accounts: {len(train_accounts)}")
        print(f"Total testing accounts:  {len(test_accounts)}\n")

        return self.analyze_accounts(test_accounts, train_accounts=train_accounts)

    def analyze_accounts(self, accounts: Dict, train_accounts: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Full pipeline: extract → score → cluster → stealth detect → confidence → model ID.
        Returns (results_df, cluster_summary, stealth_subclusters).
        """
        if train_accounts is None:
            train_accounts = {}

        print("=" * 60)
        print("STYLESHIELD: ENHANCED BOT NETWORK DETECTION PIPELINE")
        print("=" * 60)

        # 1. Fingerprinting (enhanced: base + LLM signatures + typos + temporal)
        all_accounts = {**train_accounts, **accounts}
        print(f"\n[1/5] Extracting enhanced stylometric fingerprints ({len(all_accounts)} accounts total)...")
        fingerprints = {}
        for aid, data in all_accounts.items():
            fingerprints[aid] = self.extractor.extract(
                data.get("posts", []),
                posting_hours=data.get("posting_hours"),
            )

        # 2. Per-account bot scores (only for test accounts)
        print("\n[2/5] Computing multi-metric bot scores for test accounts...")
        all_scores = {}
        for account_id in accounts:
            bot_score, metrics = self.scorer.compute_combined_bot_score(
                fp=fingerprints[account_id],
                fingerprints=fingerprints,
                account_id=account_id,
                posting_hours=accounts[account_id].get("posting_hours"),
            )
            all_scores[account_id] = {"bot_score": bot_score, "metrics": metrics}

        # 3. Clustering (enhanced: StandardScaler + euclidean DBSCAN)
        print("\n[3/5] Running enhanced DBSCAN cluster analysis...")
        cluster_df      = self.clusterer.cluster(fingerprints)
        test_cluster_df = cluster_df[cluster_df["account_id"].isin(accounts.keys())]
        cluster_sizes   = cluster_df[~cluster_df["is_noise"]].groupby("cluster_id").size().to_dict()
        cluster_summary = self.clusterer.describe_clusters(cluster_df, fingerprints)

        # 4. SECOND PASS: Stealth bot detection within human clusters
        print("\n[4/5] Running stealth bot detection (second pass)...")
        stealth_accounts, stealth_subclusters = self.clusterer.detect_stealth_subclusters(
            cluster_df, fingerprints, set(accounts.keys())
        )
        stealth_set = set(stealth_accounts)
        if stealth_accounts:
            print(f"  Found {len(stealth_accounts)} suspected stealth bot accounts in {sum(1 for v in stealth_subclusters.values() if v['is_stealth_network'])} sub-clusters")
        else:
            print("  No stealth bot sub-clusters detected")

        # 5. Adaptive confidence + model identification
        print("\n[5/5] Computing adaptive confidence scores...")
        rows = []
        for _, row in test_cluster_df.iterrows():
            account_id   = row["account_id"]
            cluster_id   = int(row["cluster_id"])
            cluster_size = cluster_sizes.get(cluster_id, 1)
            score_data   = all_scores[account_id]
            fp           = fingerprints[account_id]

            confidence, method = self.confidence_calc.compute(
                bot_score=score_data["bot_score"],
                metrics=score_data["metrics"],
                cluster_id=cluster_id,
                cluster_size=cluster_size,
                total_accounts=len(all_accounts),
            )

            # Stealth bot boost: if second pass flagged this account,
            # boost confidence and update model identification
            is_stealth = account_id in stealth_set
            if is_stealth:
                # Find which subcluster this account belongs to
                stealth_score = 0.0
                for sc_info in stealth_subclusters.values():
                    if account_id in sc_info["members"]:
                        stealth_score = sc_info["stealth_score"]
                        break
                # Blend stealth score into confidence
                confidence = max(confidence, 0.5 * confidence + 0.5 * stealth_score)
                method = "stealth_detection"

            # Model identification
            model_probs = {
                "gpt4": fp.get("model_prob_gpt4", 0),
                "claude": fp.get("model_prob_claude", 0),
                "llama": fp.get("model_prob_llama", 0),
                "human": fp.get("model_prob_human", 0),
            }
            likely_model = max(model_probs, key=model_probs.get)
            if is_stealth and likely_model == "human":
                likely_model = "stealth_bot"

            rows.append({
                "account_id":        account_id,
                "bot_score":         score_data["bot_score"],
                "confidence":        confidence,
                "confidence_method": method,
                "cluster_id":        cluster_id,
                "is_noise":          row["is_noise"],
                "likely_model":      likely_model,
                "model_confidence":  model_probs.get(likely_model, 0) if likely_model != "stealth_bot" else stealth_score,
                "llm_phrase_density": round(fp.get("llm_phrase_density", 0), 4),
                "typo_rate":         round(fp.get("typo_rate", 0), 4),
                "naturalness_score": round(fp.get("naturalness_score", 1.0), 4),
                "formality_variance": round(fp.get("formality_variance", 0.1), 6),
                "is_stealth_suspect": is_stealth,
                **score_data["metrics"],
            })

        results_df = pd.DataFrame(rows).sort_values("confidence", ascending=False)

        # Summary
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)
        print(f"Core population (train): {len(train_accounts)}")
        print(f"Accounts analyzed (test): {len(accounts)}")
        bot_networks = sum(1 for v in cluster_summary.values() if v.get("is_bot_network", False))
        stealth_networks = sum(1 for v in stealth_subclusters.values() if v.get("is_stealth_network", False))
        print(f"Bot networks found:     {bot_networks}")
        print(f"Stealth networks found: {stealth_networks}")
        print(f"Total clusters:         {len(cluster_summary)}")
        print(f"High confidence >0.8:   {(results_df['confidence'] > 0.8).sum()}")
        if stealth_accounts:
            print(f"Stealth bot suspects:   {len(stealth_accounts)}")

        # Noise (organic) accounts
        noise_ids = results_df[results_df["is_noise"]]["account_id"].tolist()
        if noise_ids:
            avg_human_prob = float(np.mean([
                fingerprints[a].get("model_prob_human", 0) for a in noise_ids
            ]))
            print(f"Organic (noise):        {len(noise_ids)} (avg human prob: {avg_human_prob:.3f})")

        test_cids = test_cluster_df[~test_cluster_df["is_noise"]]["cluster_id"].unique()
        for cid in test_cids:
            if cid not in cluster_summary:
                continue
            info = cluster_summary[cid]
            test_members = [m for m in info['members'] if m in accounts]
            bot_tag = "BOT NETWORK" if info.get("is_bot_network", False) else "CLUSTER"
            print(f"\n[{bot_tag}] Cluster {cid}: {info['member_count']} total accounts ({len(test_members)} in test)")
            print(f"  Dominant model:   {info.get('dominant_model', 'unknown')}")
            print(f"  Coordination:     {info['coordination_signal']:.3f}")
            print(f"  Avg LLM density:  {info.get('avg_llm_density', 0):.4f}")
            print(f"  Binding: {', '.join(list(info.get('binding_features', {}).keys())[:3])}")
            print(f"  Test Members: {', '.join(test_members[:10])}{'...' if len(test_members) > 10 else ''}")

        # Stealth sub-cluster details
        for sc_id, sc_info in stealth_subclusters.items():
            if not sc_info["is_stealth_network"]:
                continue
            print(f"\n[STEALTH NETWORK] Sub-cluster {sc_id}: {sc_info['member_count']} accounts")
            print(f"  Stealth score:      {sc_info['stealth_score']:.3f}")
            print(f"  Coordination:       {sc_info['coordination']:.3f}")
            print(f"  Avg naturalness:    {sc_info['avg_naturalness']:.3f}")
            print(f"  Formality variance: {sc_info['avg_formality_variance']:.6f}")
            print(f"  Pairwise similarity:{sc_info['avg_pairwise_similarity']:.3f}")
            print(f"  Members: {', '.join(sc_info['members'][:10])}{'...' if sc_info['member_count'] > 10 else ''}")

        return results_df, cluster_summary, stealth_subclusters

    def similarity_matrix(self, accounts: Dict) -> Tuple[np.ndarray, List[str]]:
        fps = {aid: self.extractor.extract(data.get("posts", []), data.get("posting_hours"))
               for aid, data in accounts.items()}
        return self.clusterer.compute_similarity_matrix(fps)


# ============================================================
# 7. JSON SERIALIZATION HELPER
# ============================================================

def _convert_numpy(obj):
    if isinstance(obj, dict):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ============================================================
# 8. DEMO / TEST HARNESS
# ============================================================

def _make_synthetic_dataset():
    gpt4_posts = [
        "Certainly! This product offers exceptional value. Furthermore, it demonstrates notable quality in every aspect.",
        "I think this is absolutely worth considering. It's important to note that the results speak for themselves.",
        "Certainly, the benefits are clear. Moreover, the features are comprehensive and well-designed.",
        "Absolutely, this is a great option. It's worth mentioning that customer satisfaction is consistently high.",
        "Certainly! Furthermore, the specifications are impressive. It's important to note the build quality.",
    ]
    claude_posts = [
        "I'd be happy to share my thoughts on this. It seems like a reasonable choice, though there are trade-offs.",
        "That's a great question! I think the answer depends on your specific needs and priorities.",
        "Interestingly, this product has some notable strengths. However, it's worth considering alternatives.",
        "I believe this is worth exploring. That said, it's important to note some limitations exist.",
        "To be honest, I'm impressed by the design. It's worth mentioning the value proposition is strong.",
    ]
    human_sets = [
        ["omg just got this and its amazing!! been using it for like 3 days now, totally worth it lol",
         "eh its ok I guess. kinda pricey but whatever. does what it says on the box",
         "bought this for my mom's birthday and she loves it!! shipping was super fast too",
         "not sure if i'd buy again tbh. worked fine for a month then had issues",
         "this thing is legit. my buddy recommended it and he was right for once haha"],
        ["tbh kinda disappointed. expected more for the price. whatever i'll deal",
         "LOVE IT. 10/10 would buy again. already got one for my sister too",
         "does the job. nothing fancy but reliable. had mine for 6 months no complaints",
         "idk it's alright. my friend has a different one and likes it better but this works",
         "can't complain for the price honestly. pretty solid"],
        ["got this as a gift and honestly wasn't sure but i like it now",
         "three stars. works sometimes doesn't work other times. frustrating",
         "my whole family uses this now lol. we're obsessed",
         "would not recommend. broke after a week. very disappointed",
         "solid purchase. glad i found it. does exactly what i needed"],
    ]
    accounts = {}
    for i in range(10):
        accounts[f"gpt4_account_{i:02d}"] = {"posts": gpt4_posts[:], "posting_hours": [9, 10, 9, 11, 10]}
    for i in range(10):
        accounts[f"claude_account_{i:02d}"] = {"posts": claude_posts[:], "posting_hours": [14, 15, 14, 13, 15]}
    for i in range(10):
        accounts[f"human_account_{i:02d}"] = {"posts": human_sets[i % 3], "posting_hours": [8, 19, 12, 22, 7, 21, 13]}
    return accounts


def _write_synthetic_csvs():
    accounts = _make_synthetic_dataset()
    rows = []
    for account_id, data in accounts.items():
        for i, post in enumerate(data["posts"]):
            hour = data["posting_hours"][i % len(data["posting_hours"])]
            rows.append({"account_id": account_id, "post_text": post, "posting_hour": hour})
    df = pd.DataFrame(rows)
    df.to_csv("synthetic_accounts.csv", index=False)
    print("Wrote synthetic_accounts.csv")
    return "synthetic_accounts.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StyleShield — Enhanced Bot Network Detection via Stylometric Analysis"
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="One or more CSV files to analyze. If omitted, auto-discovers data/ folders or runs synthetic."
    )
    parser.add_argument(
        "--template", action="store_true",
        help="Write a blank CSV template and exit."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.5,
        help="DBSCAN epsilon (euclidean distance threshold, default 1.5)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=2,
        help="DBSCAN min_samples (default 2)"
    )
    parser.add_argument(
        "--output", default="styleshield_results.csv",
        help="Output CSV path (default: styleshield_results.csv)"
    )
    args = parser.parse_args()

    scorer = StyleShieldScorer(dbscan_epsilon=args.epsilon, dbscan_min_samples=args.min_samples)

    if args.template:
        scorer.loader.save_template("template_long.csv", fmt="long")
        scorer.loader.save_template("template_wide.csv", fmt="wide")
        sys.exit(0)

    if args.csv_files:
        results, clusters, stealth = scorer.analyze_csv([], args.csv_files)
    else:
        # Auto-discover CSVs in training and test folders
        train_dir = Path("data/training")
        test_dir  = Path("data/test")

        train_csvs = sorted(train_dir.glob("*.csv")) if train_dir.exists() else []
        test_csvs  = sorted(test_dir.glob("*.csv")) if test_dir.exists() else []

        if train_csvs or test_csvs:
            print(f"Auto-discovery: found {len(train_csvs)} training and {len(test_csvs)} testing CSVs.")
            results, clusters, stealth = scorer.analyze_csv(
                [str(p) for p in train_csvs],
                [str(p) for p in test_csvs]
            )
        else:
            print("No CSVs found in data/training or data/test.")
            print("Running on built-in synthetic dataset.\n")
            csv_path = _write_synthetic_csvs()
            results, clusters, stealth = scorer.analyze_csv([], [csv_path])

    # Print top results
    print("\n\nTOP 15 ACCOUNTS BY CONFIDENCE:")
    print("-" * 90)
    cols = ["account_id", "confidence", "bot_score", "cluster_id", "likely_model",
            "is_stealth_suspect", "naturalness_score", "llm_phrase_density", "typo_rate",
            "structural_regularity"]
    available_cols = [c for c in cols if c in results.columns]
    print(results[available_cols].head(15).to_string(index=False))

    # Export
    results.to_csv(args.output, index=False)
    clusters_path = args.output.replace(".csv", "_clusters.json")
    with open(clusters_path, "w") as f:
        json.dump(_convert_numpy(clusters), f, indent=2)
    stealth_path = args.output.replace(".csv", "_stealth.json")
    with open(stealth_path, "w") as f:
        json.dump(_convert_numpy(stealth), f, indent=2)

    print(f"\nExported: {args.output}, {clusters_path}, {stealth_path}")
    print("StyleShield complete.")
