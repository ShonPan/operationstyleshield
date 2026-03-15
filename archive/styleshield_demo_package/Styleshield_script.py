"""
StyleShield: Bot Network Detection via Stylometric Anomaly Detection
=====================================================================
Adapted from Xenarch Mk14 (planetary technosignature detection).
 
Core principle: Train on natural human writing diversity. Flag artificial
uniformity. A bot farm's output is the textual equivalent of a lunar lander
sitting in a field of craters — it doesn't belong, and a properly tuned
anomaly detector will find it.
 
Scoring formula:
    B = 0.25·VU + 0.25·SR + 0.20·HF + 0.20·TC + 0.10·CA
 
Xenarch → StyleShield metric mapping:
    MSE (recon error)   → Vocabulary Uniformity (VU)
    Edge Regularity     → Structural Regularity  (SR)
    Latent Density      → Hedging & Filler Sig.  (HF)
    Contextual Metric   → Temporal-Contextual    (TC)
    Gradient Anomaly    → Cross-Account Corr.    (CA)
 
Usage:
    from styleshield_scorer import StyleShieldScorer
 
    # From dict
    scorer = StyleShieldScorer()
    results, clusters = scorer.analyze_accounts(accounts_dict)
 
    # From CSV (auto-detects long or wide format)
    results, clusters = scorer.analyze_csv("accounts.csv")
    results, clusters = scorer.analyze_csv("training.csv", "test.csv")
 
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
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
 
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
 
 
# ============================================================
# 1. STYLOMETRIC FEATURE EXTRACTOR
# ============================================================
 
class StylometricExtractor:
    """
    Extracts writing fingerprint features from a corpus of posts.
    Parallel to Xenarch's ChipExtractor — turns raw imagery/text
    into normalized feature vectors ready for anomaly scoring.
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
 
    def extract(self, posts: List[str]) -> Dict:
        if not posts:
            return self._empty_fingerprint()
 
        full_text = " ".join(posts)
        tokens = self._tokenize(full_text)
        sentences = self._split_sentences(full_text)
        per_post = [self._extract_single(p) for p in posts if p.strip()]
 
        return {
            # Vocabulary (→ VU)
            "type_token_ratio":      self._type_token_ratio(tokens),
            "vocabulary_size":       len(set(t.lower() for t in tokens)),
            "hapax_ratio":           self._hapax_ratio(tokens),
            "top10_concentration":   self._top_n_concentration(tokens, 10),
            "avg_word_length":       float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
            # Structure (→ SR)
            "avg_sentence_length":   float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0,
            "sentence_length_variance": float(np.var([len(s.split()) for s in sentences])) if sentences else 0.0,
            "avg_post_length":       float(np.mean([len(p.split()) for p in posts])),
            "post_length_variance":  float(np.var([len(p.split()) for p in posts])),
            "paragraph_rhythm_score": self._paragraph_rhythm(posts),
            # Hedging (→ HF)
            "hedge_rate":            self._phrase_rate(full_text.lower(), self.HEDGE_PHRASES),
            "transition_rate":       self._phrase_rate(full_text.lower(), self.TRANSITION_WORDS),
            "exclamation_rate":      full_text.count("!") / max(len(sentences), 1),
            "question_rate":         full_text.count("?") / max(len(sentences), 1),
            "contraction_rate":      self._contraction_rate(full_text),
            # Punctuation DNA
            "comma_rate":            full_text.count(",") / max(len(tokens), 1),
            "semicolon_rate":        full_text.count(";") / max(len(tokens), 1),
            "dash_rate":             (full_text.count("—") + full_text.count(" - ")) / max(len(sentences), 1),
            "ellipsis_rate":         full_text.count("...") / max(len(sentences), 1),
            "emoji_rate":            self._emoji_rate(full_text, len(tokens)),
            # Intra-account variance (bots are unnaturally consistent)
            "intra_vocab_variance":     self._intra_var(per_post, "type_token_ratio"),
            "intra_structure_variance": self._intra_var(per_post, "avg_sentence_length"),
            "intra_hedge_variance":     self._intra_var(per_post, "hedge_rate"),
            # Meta
            "post_count":    len(posts),
            "total_tokens":  len(tokens),
        }
 
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
 
    def _intra_var(self, per_post, key):
        vals = [p[key] for p in per_post if key in p]
        return float(np.var(vals)) if len(vals) >= 2 else 1.0
 
    def _empty_fingerprint(self):
        keys = [
            "type_token_ratio", "vocabulary_size", "hapax_ratio", "top10_concentration",
            "avg_word_length", "avg_sentence_length", "sentence_length_variance",
            "avg_post_length", "post_length_variance", "paragraph_rhythm_score",
            "hedge_rate", "transition_rate", "exclamation_rate", "question_rate",
            "contraction_rate", "comma_rate", "semicolon_rate", "dash_rate",
            "ellipsis_rate", "emoji_rate", "intra_vocab_variance",
            "intra_structure_variance", "intra_hedge_variance", "post_count", "total_tokens"
        ]
        return {k: 0.0 for k in keys}
 
 
# ============================================================
# 2. MULTI-METRIC BOT SCORER
# ============================================================
 
class MultiMetricBotScorer:
    """
    Five normalized metrics combined into a single bot score.
 
    Xenarch parallel:
        compute_reconstruction_error → compute_vocabulary_uniformity
        compute_edge_regularity      → compute_structural_regularity
        compute_latent_density       → compute_hedging_signature
        compute_contextual_anomaly   → compute_temporal_contextual
        compute_gradient_anomaly     → compute_cross_account_correlation
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
 
        combined = 0.25*vu + 0.25*sr + 0.20*hf + 0.20*tc + 0.10*ca
 
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
# 3. DBSCAN CLUSTER ANALYSIS
# ============================================================
 
class AccountClusterAnalyzer:
    """
    Identifies bot networks by clustering accounts in stylometric space.
    Direct port of Xenarch's spatial DBSCAN clustering.
    """
 
    FEATURE_KEYS = [
        "type_token_ratio", "hapax_ratio", "top10_concentration",
        "avg_word_length", "avg_sentence_length", "sentence_length_variance",
        "avg_post_length", "post_length_variance", "paragraph_rhythm_score",
        "hedge_rate", "transition_rate", "contraction_rate",
        "comma_rate", "semicolon_rate", "dash_rate",
        "intra_vocab_variance", "intra_structure_variance", "intra_hedge_variance"
    ]
 
    def __init__(self, epsilon=0.25, min_samples=2):
        self.epsilon = epsilon
        self.min_samples = min_samples
 
    def _build_matrix(self, fingerprints):
        ids = list(fingerprints.keys())
        mat = np.array([[fingerprints[a].get(k, 0.0) for k in self.FEATURE_KEYS] for a in ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms, ids
 
    def compute_similarity_matrix(self, fingerprints):
        mat, ids = self._build_matrix(fingerprints)
        return cosine_similarity(mat), ids
 
    def cluster(self, fingerprints):
        if len(fingerprints) < 2:
            return pd.DataFrame([{"account_id": a, "cluster_id": -1, "is_noise": True}
                                  for a in fingerprints])
        mat, ids = self._build_matrix(fingerprints)
        dist = np.clip(1 - cosine_similarity(mat), 0, 2)
        labels = DBSCAN(eps=self.epsilon, min_samples=self.min_samples,
                        metric="precomputed").fit_predict(dist)
        return pd.DataFrame([
            {"account_id": a, "cluster_id": int(l), "is_noise": l == -1}
            for a, l in zip(ids, labels)
        ])
 
    def describe_clusters(self, cluster_df, fingerprints):
        summary = {}
        for cid in cluster_df[~cluster_df["is_noise"]]["cluster_id"].unique():
            members = cluster_df[cluster_df["cluster_id"] == cid]["account_id"].tolist()
            fps = [fingerprints[a] for a in members if a in fingerprints]
            if not fps:
                continue
            shared = {}
            for key in self.FEATURE_KEYS:
                vals = [fp.get(key, 0.0) for fp in fps]
                shared[key] = {
                    "mean":        round(float(np.mean(vals)), 4),
                    "std":         round(float(np.std(vals)), 4),
                    "uniformity":  round(1.0 / (1.0 + float(np.std(vals))), 4),
                }
            binding = sorted(shared.items(), key=lambda x: x[1]["std"])[:5]
            summary[cid] = {
                "member_count":      len(members),
                "members":           members,
                "binding_features":  {k: v for k, v in binding},
                "coordination_signal": round(
                    float(np.mean([v["uniformity"] for v in shared.values()])), 4
                ),
            }
        return summary
 
 
# ============================================================
# 4. ADAPTIVE CONFIDENCE CALCULATOR
# ============================================================
 
class AdaptiveConfidenceCalculator:
    """
    Direct port of Xenarch's _compute_confidence().
 
    Large dataset with clusters:
        c = 0.40·S + 0.30·cluster_membership + 0.20·structural_regularity + 0.10·agreement
 
    Small dataset / isolated account:
        c = 0.50·S + 0.30·hedging_signature + 0.20·vocabulary_uniformity
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
 
    Supported formats
    -----------------
    Long format — one post per row (recommended):
 
        account_id | post_text                          | posting_hour
        -----------|------------------------------------|-------------
        user_001   | "Certainly! This is a great..."    | 9
        user_001   | "Furthermore, the quality..."      | 10
        user_002   | "omg just got this lol"             | 21
 
    Wide format — one account per row:
 
        account_id | post_1           | post_2           | posting_hours
        -----------|------------------|------------------|---------------
        user_001   | "First post..."   | "Second post..." | 9;10;14
        user_002   | "Only post here"  |                  |
 
    Column name matching is case-insensitive.
    Common aliases are accepted (see ALIASES below).
    Run `scorer.loader.save_template("template.csv")` for a blank template.
    """
 
    ACCOUNT_ALIASES   = ["account_id", "account", "user_id", "user", "username", "handle", "author"]
    POST_ALIASES      = ["post_text", "text", "content", "body", "post", "tweet", "message", "comment"]
    HOUR_ALIASES      = ["posting_hour", "hour", "post_hour", "hour_of_day"]
    TIMESTAMP_ALIASES = ["post_timestamp", "timestamp", "created_at", "date", "datetime", "time"]
 
    def load(self, csv_path: str) -> Dict:
        """Load a single CSV. Returns accounts dict."""
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
        """Merge multiple CSVs into one accounts dict."""
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
        """Write a blank template CSV showing the expected format."""
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
 
    # ------------------------------------------------------------------
 
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
 
        # Derive hour from timestamp if no explicit hour column
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
 
    From dict:
        scorer = StyleShieldScorer()
        results, clusters = scorer.analyze_accounts(accounts_dict)
 
    From CSV (auto-detects long or wide format):
        results, clusters = scorer.analyze_csv("accounts.csv")
        results, clusters = scorer.analyze_csv("training.csv", "test.csv")
 
    Generate a blank CSV template:
        scorer.loader.save_template("template.csv")
    """
 
    def __init__(self, dbscan_epsilon=0.25, dbscan_min_samples=2):
        self.extractor        = StylometricExtractor()
        self.scorer           = MultiMetricBotScorer()
        self.clusterer        = AccountClusterAnalyzer(epsilon=dbscan_epsilon,
                                                       min_samples=dbscan_min_samples)
        self.confidence_calc  = AdaptiveConfidenceCalculator()
        self.loader           = CSVLoader()
 
    # ------------------------------------------------------------------
    # CSV entry points
    # ------------------------------------------------------------------
 
    def analyze_csv(self, *csv_paths: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load one or more CSV files and run the full detection pipeline.
 
        Accepts any mix of long-format or wide-format CSVs (auto-detected).
        Multiple files are merged before analysis.
 
        Examples
        --------
        results, clusters = scorer.analyze_csv("accounts.csv")
        results, clusters = scorer.analyze_csv("training.csv", "test.csv")
        """
        print(f"\nLoading {len(csv_paths)} CSV file(s)...")
        accounts = (self.loader.load(csv_paths[0]) if len(csv_paths) == 1
                    else self.loader.load_multiple(*csv_paths))
        print(f"  Total accounts loaded: {len(accounts)}\n")
        return self.analyze_accounts(accounts)
 
    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
 
    def analyze_accounts(self, accounts: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Full pipeline: extract → score → cluster → confidence.
        Returns (results_df, cluster_summary).
        """
        print("=" * 60)
        print("STYLESHIELD: BOT NETWORK DETECTION PIPELINE")
        print("=" * 60)
 
        # 1. Fingerprinting
        print(f"\n[1/4] Extracting stylometric fingerprints ({len(accounts)} accounts)...")
        fingerprints = {aid: self.extractor.extract(data.get("posts", []))
                        for aid, data in accounts.items()}
 
        # 2. Per-account bot scores
        print("\n[2/4] Computing multi-metric bot scores...")
        all_scores = {}
        for account_id in accounts:
            bot_score, metrics = self.scorer.compute_combined_bot_score(
                fp=fingerprints[account_id],
                fingerprints=fingerprints,
                account_id=account_id,
                posting_hours=accounts[account_id].get("posting_hours"),
            )
            all_scores[account_id] = {"bot_score": bot_score, "metrics": metrics}
 
        # 3. Clustering
        print("\n[3/4] Running DBSCAN cluster analysis...")
        cluster_df      = self.clusterer.cluster(fingerprints)
        cluster_sizes   = cluster_df[~cluster_df["is_noise"]].groupby("cluster_id").size().to_dict()
        cluster_summary = self.clusterer.describe_clusters(cluster_df, fingerprints)
 
        # 4. Adaptive confidence
        print("\n[4/4] Computing adaptive confidence scores...")
        rows = []
        for _, row in cluster_df.iterrows():
            account_id   = row["account_id"]
            cluster_id   = int(row["cluster_id"])
            cluster_size = cluster_sizes.get(cluster_id, 1)
            score_data   = all_scores[account_id]
 
            confidence, method = self.confidence_calc.compute(
                bot_score=score_data["bot_score"],
                metrics=score_data["metrics"],
                cluster_id=cluster_id,
                cluster_size=cluster_size,
                total_accounts=len(accounts),
            )
            rows.append({
                "account_id":        account_id,
                "bot_score":         score_data["bot_score"],
                "confidence":        confidence,
                "confidence_method": method,
                "cluster_id":        cluster_id,
                "is_noise":          row["is_noise"],
                **score_data["metrics"],
            })
 
        results_df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
 
        # Summary
        print("\n" + "=" * 60)
        print("DETECTION SUMMARY")
        print("=" * 60)
        print(f"Accounts analyzed:    {len(accounts)}")
        print(f"Bot networks found:   {len(cluster_summary)}")
        print(f"Accounts in networks: {sum(v['member_count'] for v in cluster_summary.values())}")
        print(f"High confidence >0.8: {(results_df['confidence'] > 0.8).sum()}")
 
        for cid, info in cluster_summary.items():
            print(f"\nCluster {cid}: {info['member_count']} accounts "
                  f"(coordination: {info['coordination_signal']:.3f})")
            print(f"  Members: {', '.join(info['members'])}")
            print(f"  Binding: {', '.join(list(info['binding_features'].keys())[:3])}")
 
        return results_df, cluster_summary
 
    def similarity_matrix(self, accounts: Dict) -> Tuple[np.ndarray, List[str]]:
        fps = {aid: self.extractor.extract(data.get("posts", []))
               for aid, data in accounts.items()}
        return self.clusterer.compute_similarity_matrix(fps)
 
 
# ============================================================
# 7. DEMO / TEST HARNESS
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
    """Write the synthetic dataset to CSV files for testing."""
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
        description="StyleShield — Bot Network Detection via Stylometric Analysis"
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="One or more CSV files to analyze. If omitted, runs on built-in synthetic data."
    )
    parser.add_argument(
        "--template", action="store_true",
        help="Write a blank CSV template and exit."
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.30,
        help="DBSCAN epsilon (cosine distance threshold, default 0.30)"
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
        results, clusters = scorer.analyze_csv(*args.csv_files)
    else:
        print("No CSV files provided — running on built-in synthetic dataset.\n")
        csv_path = _write_synthetic_csvs()
        results, clusters = scorer.analyze_csv(csv_path)
 
    # Print top results
    print("\n\nTOP 15 ACCOUNTS BY CONFIDENCE:")
    print("-" * 80)
    cols = ["account_id", "confidence", "bot_score", "cluster_id",
            "vocabulary_uniformity", "structural_regularity", "hedging_signature"]
    print(results[cols].head(15).to_string(index=False))
 
    # Export
    results.to_csv(args.output, index=False)
    clusters_path = args.output.replace(".csv", "_clusters.json")

    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(clusters_path, "w") as f:
        json.dump(_convert(clusters), f, indent=2)
 
    print(f"\nExported: {args.output}, {clusters_path}")
    print("StyleShield complete.")