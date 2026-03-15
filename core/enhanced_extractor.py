"""
StyleShield Enhanced Feature Extractor
======================================
Adds missing extraction targets to Caleb's StylometricExtractor.

New features:
  - LLM model signatures (GPT-4 vs Claude vs Llama vs human)
  - Sentence opener analysis
  - Typo/device fingerprinting
  - Temporal shift detection
  - Jargon density & syllable analysis
  - Enhanced intra-account variance

Drop-in: import and call enhance_extractor(existing_extractor)
or use EnhancedStylometricExtractor directly.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Optional
import numpy as np


# ============================================================
# KEYBOARD LAYOUT for adjacent-key typo detection
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

# ============================================================
# LLM-SPECIFIC PHRASE SIGNATURES
# ============================================================
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

# Common English words (non-jargon baseline)
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

# Approximate syllable counter
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


class EnhancedStylometricExtractor:
    """
    Extended feature extraction covering all spec targets.
    Can be used standalone or to augment Caleb's base extractor.
    """

    def extract_enhanced(self, posts: List[str], posting_hours: Optional[List[int]] = None) -> Dict:
        """
        Extract all enhanced features from a list of posts.
        Returns dict of new features to merge with base fingerprint.
        """
        if not posts:
            return self._empty_enhanced()

        full_text = " ".join(posts)
        full_lower = full_text.lower()
        tokens = re.findall(r"\b[a-zA-Z']+\b", full_text)
        tokens_lower = [t.lower() for t in tokens]
        sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if len(s.strip()) > 3]
        per_post = [self._extract_post_features(p) for p in posts if p.strip()]

        features = {}

        # ---- LLM MODEL SIGNATURES ----
        features["gpt4_marker_rate"] = self._phrase_rate(full_lower, GPT4_MARKERS, len(tokens))
        features["claude_marker_rate"] = self._phrase_rate(full_lower, CLAUDE_MARKERS, len(tokens))
        features["llama_marker_rate"] = self._phrase_rate(full_lower, LLAMA_MARKERS, len(tokens))
        features["generic_llm_rate"] = self._phrase_rate(full_lower, GENERIC_LLM_MARKERS, len(tokens))
        features["epistemic_qualifier_rate"] = self._phrase_rate(full_lower, EPISTEMIC_QUALIFIERS, len(tokens))

        # Individual high-signal markers
        features["certainly_rate"] = full_lower.count("certainly") / max(len(sentences), 1)
        features["furthermore_rate"] = full_lower.count("furthermore") / max(len(sentences), 1)
        features["delve_rate"] = full_lower.count("delve") / max(len(tokens), 1)
        features["moreover_rate"] = full_lower.count("moreover") / max(len(sentences), 1)

        # Combined LLM density (any model)
        features["llm_phrase_density"] = (
            features["gpt4_marker_rate"] +
            features["claude_marker_rate"] +
            features["llama_marker_rate"] +
            features["generic_llm_rate"]
        )

        # Model probability (which LLM is most likely?)
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

        # ---- SENTENCE OPENER ANALYSIS ----
        openers = []
        for s in sentences:
            words = s.strip().split()
            if words:
                openers.append(words[0].lower())

        if openers:
            opener_freq = Counter(openers)
            features["opener_diversity"] = len(set(openers)) / max(len(openers), 1)
            features["opener_repetition_rate"] = max(opener_freq.values()) / len(openers) if openers else 0
            features["question_opener_rate"] = sum(1 for o in openers if o in ("what", "why", "how", "when", "where", "who", "is", "are", "do", "does", "can", "could", "would", "should")) / max(len(openers), 1)
            # "I" opener rate (humans start with "I" more in casual writing)
            features["i_opener_rate"] = sum(1 for o in openers if o == "i") / max(len(openers), 1)
            # "The" opener rate (LLMs use "The" to start more often)
            features["the_opener_rate"] = sum(1 for o in openers if o == "the") / max(len(openers), 1)
        else:
            features["opener_diversity"] = 0
            features["opener_repetition_rate"] = 0
            features["question_opener_rate"] = 0
            features["i_opener_rate"] = 0
            features["the_opener_rate"] = 0

        # List/bullet markers (bots love lists)
        features["list_marker_rate"] = len(re.findall(r'(?:^|\n)\s*[\-\*\d]+[\.\)]\s', full_text)) / max(len(sentences), 1)

        # ---- VOCABULARY: JARGON & SYLLABLES ----
        if tokens_lower:
            uncommon = [t for t in tokens_lower if t not in COMMON_WORDS_1000 and len(t) > 3]
            features["jargon_density"] = len(uncommon) / max(len(tokens_lower), 1)
            features["avg_syllables"] = float(np.mean([count_syllables(t) for t in tokens_lower]))

            # Word frequency distribution shape (Zipf's law deviation)
            freq = Counter(tokens_lower)
            sorted_freqs = sorted(freq.values(), reverse=True)
            if len(sorted_freqs) > 1:
                # Human text follows Zipf more closely; AI is flatter
                top_ratio = sorted_freqs[0] / max(sorted_freqs[-1], 1)
                features["zipf_ratio"] = min(top_ratio / 50.0, 1.0)  # normalized
            else:
                features["zipf_ratio"] = 0
        else:
            features["jargon_density"] = 0
            features["avg_syllables"] = 0
            features["zipf_ratio"] = 0

        # ---- PUNCTUATION: EXTENDED ----
        features["parenthetical_rate"] = (full_text.count("(") + full_text.count(")")) / max(len(tokens), 1)
        features["colon_rate"] = full_text.count(":") / max(len(sentences), 1)
        features["all_caps_rate"] = sum(1 for t in tokens if t.isupper() and len(t) > 1) / max(len(tokens), 1)

        # ---- TYPO / DEVICE FINGERPRINTING ----
        typo_data = self._analyze_typos(full_text, tokens)
        features["typo_rate"] = typo_data["typo_rate"]
        features["adjacent_key_error_rate"] = typo_data["adjacent_key_rate"]
        features["dropped_char_rate"] = typo_data["dropped_char_rate"]
        features["double_char_rate"] = typo_data["double_char_rate"]
        # Device inference: mobile users have more adjacent-key errors
        # Desktop users have more transposition errors
        mobile_signal = typo_data["adjacent_key_rate"] * 2 + typo_data["dropped_char_rate"]
        features["mobile_device_signal"] = float(np.clip(mobile_signal, 0, 1))

        # ---- TEMPORAL / SHIFT DETECTION ----
        if posting_hours and len(posting_hours) >= 2:
            hours = np.array(posting_hours)
            features["posting_hour_mean"] = float(np.mean(hours))
            features["posting_hour_variance"] = float(np.var(hours))

            # Shift concentration: divide day into 3 shifts (0-7, 8-15, 16-23)
            bins = np.zeros(3)
            for h in hours:
                bins[int(h) // 8] += 1
            bins /= max(bins.sum(), 1)
            features["shift_concentration"] = float(np.max(bins))
            features["dominant_shift"] = int(np.argmax(bins))  # 0=night, 1=day, 2=evening

            # Business hours ratio (9-17 weekday pattern)
            biz = sum(1 for h in hours if 9 <= h <= 17) / len(hours)
            features["business_hours_ratio"] = biz

            # Posting regularity (low variance = bot signal)
            features["posting_regularity"] = 1.0 / (1.0 + float(np.var(hours)) / 50.0)

            # Timezone inference (rough: mode of posting hours maps to activity peak)
            mode_hour = int(Counter(posting_hours).most_common(1)[0][0])
            # Assume peak activity = local evening (18-20h)
            # So timezone offset ≈ mode_hour - 19 (mod 24)
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

        # ---- ENHANCED INTRA-ACCOUNT VARIANCE ----
        if len(per_post) >= 2:
            features["intra_punctuation_variance"] = float(np.var([p["comma_rate"] for p in per_post]))
            features["intra_opener_variance"] = float(np.var([p["opener_diversity"] for p in per_post]))
            features["intra_jargon_variance"] = float(np.var([p["jargon_density"] for p in per_post]))
            features["intra_syllable_variance"] = float(np.var([p["avg_syllables"] for p in per_post]))
            features["intra_typo_variance"] = float(np.var([p["typo_rate"] for p in per_post]))
        else:
            features["intra_punctuation_variance"] = 1.0
            features["intra_opener_variance"] = 1.0
            features["intra_jargon_variance"] = 1.0
            features["intra_syllable_variance"] = 1.0
            features["intra_typo_variance"] = 1.0

        return features

    # ---- HELPERS ----

    def _phrase_rate(self, text, phrases, token_count):
        if token_count == 0:
            return 0.0
        hits = sum(text.count(p) for p in phrases)
        return hits / max(token_count / 10, 1)

    def _analyze_typos(self, text, tokens):
        """
        Detect typo patterns that fingerprint the author/device.
        """
        words = text.split()
        if not words:
            return {"typo_rate": 0, "adjacent_key_rate": 0, "dropped_char_rate": 0, "double_char_rate": 0}

        # Simple heuristic: words not in a basic vocabulary that look like misspellings
        # Real implementation would use a spell checker, but this works for hackathon
        misspelling_patterns = 0
        adjacent_key_errors = 0
        dropped_chars = 0
        double_chars = 0

        for word in words:
            w = word.lower().strip(".,!?;:'\"()[]{}@#$%^&*")
            if len(w) < 2:
                continue

            # Double character detection (e.g., "teh" patterns)
            for i in range(len(w) - 1):
                if w[i] == w[i+1] and w[i].isalpha():
                    # Check if the double is unusual (not common doubles like "ll", "ss", "ee")
                    common_doubles = set("lseptofn")
                    if w[i] not in common_doubles:
                        double_chars += 1

            # Adjacent key detection: look for character pairs that are keyboard neighbors
            for i in range(len(w) - 1):
                c1, c2 = w[i], w[i+1]
                if c1 in QWERTY_NEIGHBORS and c2 in QWERTY_NEIGHBORS.get(c1, set()):
                    # This alone isn't a typo — check if reversing makes a more common pattern
                    adjacent_key_errors += 0.1  # low weight, accumulates

        total_words = len(words)
        # Use informal spelling markers as typo proxies
        informal_markers = len(re.findall(r'\b(u|ur|r|n|thru|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|lol|omg|tbh|idk|imo|smh|btw)\b', text.lower()))

        return {
            "typo_rate": informal_markers / max(total_words, 1),
            "adjacent_key_rate": adjacent_key_errors / max(total_words, 1),
            "dropped_char_rate": dropped_chars / max(total_words, 1),
            "double_char_rate": double_chars / max(total_words, 1),
        }

    def _extract_post_features(self, post):
        """Per-post features for intra-account variance."""
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
            "typo_rate": len(re.findall(r'\b(u|ur|r|n|thru|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|lol|omg|tbh|idk|imo|smh|btw)\b', post.lower())) / max(len(tokens), 1),
        }

    def _empty_enhanced(self):
        """Return zeroed feature dict."""
        keys = [
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
        return {k: 0.0 for k in keys}


# ============================================================
# INTEGRATION: Patch into Caleb's pipeline
# ============================================================

def enhance_fingerprint(base_fingerprint: Dict, posts: List[str],
                        posting_hours: Optional[List[int]] = None) -> Dict:
    """
    Take a base fingerprint from Caleb's extractor and add enhanced features.

    Usage:
        extractor = StylometricExtractor()  # Caleb's
        enhancer = EnhancedStylometricExtractor()

        base_fp = extractor.extract(posts)
        enhanced = enhance_fingerprint(base_fp, posts, posting_hours)
        # enhanced now has all base + new features
    """
    enhancer = EnhancedStylometricExtractor()
    new_features = enhancer.extract_enhanced(posts, posting_hours)
    return {**base_fingerprint, **new_features}


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    enhancer = EnhancedStylometricExtractor()

    # Test with GPT-4 style text
    gpt_posts = [
        "Certainly! This product offers exceptional value. Furthermore, it demonstrates notable quality in every aspect.",
        "I think this is absolutely worth considering. It's important to note that the results speak for themselves.",
        "Certainly, the benefits are clear. Moreover, the features are comprehensive and well-designed.",
    ]

    # Test with human style text
    human_posts = [
        "omg just got this and its amazing!! been using it for like 3 days now, totally worth it lol",
        "eh its ok I guess. kinda pricey but whatever. does what it says on the box",
        "bought this for my mom's birthday and she loves it!! shipping was super fast too tbh",
    ]

    # Test with Haiku bot text
    haiku_posts = [
        "Just got the new Sony WH-1000XM5 headphones and wow, the noise cancellation is absolutely incredible. Best purchase I've made all year!",
        "Been using this vitamin D supplement for 2 weeks now and I already feel so much more energized. Highly recommend to everyone!",
        "Finally upgraded to the Samsung Galaxy S24 and I'm blown away by the camera quality. Every photo looks professional.",
    ]

    print("=" * 70)
    print("ENHANCED STYLOMETRIC EXTRACTION TEST")
    print("=" * 70)

    for label, posts in [("GPT-4 BOT", gpt_posts), ("HUMAN", human_posts), ("HAIKU BOT", haiku_posts)]:
        features = enhancer.extract_enhanced(posts, posting_hours=[9, 10, 11])
        print(f"\n--- {label} ---")
        print(f"  LLM density:        {features['llm_phrase_density']:.4f}")
        print(f"  GPT-4 markers:      {features['gpt4_marker_rate']:.4f}")
        print(f"  Claude markers:     {features['claude_marker_rate']:.4f}")
        print(f"  Model prob (GPT4):  {features['model_prob_gpt4']:.3f}")
        print(f"  Model prob (Claude):{features['model_prob_claude']:.3f}")
        print(f"  Model prob (Human): {features['model_prob_human']:.3f}")
        print(f"  Epistemic qual:     {features['epistemic_qualifier_rate']:.4f}")
        print(f"  Opener diversity:   {features['opener_diversity']:.4f}")
        print(f"  Opener repetition:  {features['opener_repetition_rate']:.4f}")
        print(f"  Jargon density:     {features['jargon_density']:.4f}")
        print(f"  Avg syllables:      {features['avg_syllables']:.2f}")
        print(f"  Typo rate:          {features['typo_rate']:.4f}")
        print(f"  Mobile signal:      {features['mobile_device_signal']:.4f}")
        print(f"  All caps rate:      {features['all_caps_rate']:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON MATRIX")
    print("=" * 70)

    key_features = [
        "llm_phrase_density", "gpt4_marker_rate", "claude_marker_rate",
        "model_prob_human", "jargon_density", "typo_rate",
        "opener_diversity", "avg_syllables", "all_caps_rate",
    ]

    results = {}
    for label, posts in [("GPT-4", gpt_posts), ("HUMAN", human_posts), ("HAIKU", haiku_posts)]:
        results[label] = enhancer.extract_enhanced(posts, posting_hours=[9, 10, 11])

    print(f"\n{'Feature':<25} {'GPT-4':>8} {'HUMAN':>8} {'HAIKU':>8}")
    print("-" * 55)
    for feat in key_features:
        vals = [results[l].get(feat, 0) for l in ["GPT-4", "HUMAN", "HAIKU"]]
        print(f"{feat:<25} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f}")
