"""
StyleShield Enhanced Pipeline
=============================
Integrates the enhanced feature extractor with Caleb's base scorer.
Uses standardized euclidean distance for clustering (better separation).

Usage:
    python3 enhanced_pipeline.py full_combined.csv
    python3 enhanced_pipeline.py haiku_formatted.csv human_airline_Tweets.csv
"""

import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE

# Import base modules (adjust paths as needed)
from Styleshield_script import StylometricExtractor, CSVLoader, MultiMetricBotScorer
from enhanced_extractor import EnhancedStylometricExtractor, enhance_fingerprint


# Features that best separate bots from humans
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


def run_enhanced_pipeline(csv_paths, epsilon=1.5, min_samples=2):
    """Full enhanced pipeline: extract → score → cluster → report."""

    # 1. Load data
    loader = CSVLoader()
    if len(csv_paths) == 1:
        accounts = loader.load(csv_paths[0])
    else:
        accounts = loader.load_multiple(*csv_paths)

    print(f"\nTotal accounts loaded: {len(accounts)}")

    # 2. Extract base + enhanced fingerprints
    print("\n[1/4] Extracting enhanced stylometric fingerprints...")
    base_extractor = StylometricExtractor()
    enhancer = EnhancedStylometricExtractor()

    fingerprints = {}
    for aid, data in accounts.items():
        base_fp = base_extractor.extract(data.get("posts", []))
        enhanced_fp = enhancer.extract_enhanced(
            data.get("posts", []),
            data.get("posting_hours")
        )
        fingerprints[aid] = {**base_fp, **enhanced_fp}

    # 3. Build feature matrix with standardization
    print("[2/4] Building standardized feature matrix...")
    ids = list(fingerprints.keys())
    mat = np.array([
        [fingerprints[a].get(k, 0.0) for k in DISCRIMINATING_FEATURES]
        for a in ids
    ])

    scaler = StandardScaler()
    mat_std = scaler.fit_transform(mat)

    # 2b. t-SNE projection for network graph layout
    perplexity = min(15, len(ids) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(mat_std)
    for dim in range(2):
        mn, mx = coords[:, dim].min(), coords[:, dim].max()
        rng = mx - mn if mx - mn > 0 else 1
        coords[:, dim] = (coords[:, dim] - mn) / rng * 0.8 + 0.1
    graph_positions = {
        aid: {"x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4)}
        for i, aid in enumerate(ids)
    }

    # 4. DBSCAN clustering (euclidean on standardized features)
    print(f"[3/4] Running DBSCAN clustering (eps={epsilon}, min_samples={min_samples})...")
    labels = DBSCAN(
        eps=epsilon,
        min_samples=min_samples,
        metric="euclidean"
    ).fit_predict(mat_std)

    # 5. Build results
    print("[4/4] Building results...")
    rows = []
    cluster_members = {}

    for aid, label in zip(ids, labels):
        fp = fingerprints[aid]
        is_noise = label == -1

        # Model identification
        model_probs = {
            "gpt4": fp.get("model_prob_gpt4", 0),
            "claude": fp.get("model_prob_claude", 0),
            "llama": fp.get("model_prob_llama", 0),
            "human": fp.get("model_prob_human", 0),
        }
        likely_model = max(model_probs, key=model_probs.get)

        row = {
            "account_id": aid,
            "cluster_id": int(label),
            "is_noise": is_noise,
            "likely_model": likely_model,
            "model_confidence": model_probs[likely_model],
            "llm_phrase_density": round(fp.get("llm_phrase_density", 0), 4),
            "typo_rate": round(fp.get("typo_rate", 0), 4),
            "structural_regularity": round(fp.get("paragraph_rhythm_score", 0), 4),
            "avg_syllables": round(fp.get("avg_syllables", 0), 2),
            "jargon_density": round(fp.get("jargon_density", 0), 4),
            "contraction_rate": round(fp.get("contraction_rate", 0), 4),
            "intra_vocab_var": round(fp.get("intra_vocab_variance", 0), 6),
            "intra_struct_var": round(fp.get("intra_structure_variance", 0), 4),
            "post_count": fp.get("post_count", 0),
        }
        rows.append(row)

        if not is_noise:
            cluster_members.setdefault(label, []).append(aid)

    results_df = pd.DataFrame(rows)

    # Cluster summary
    cluster_summary = {}
    for cid, members in cluster_members.items():
        member_fps = [fingerprints[m] for m in members]

        # What model does this cluster use?
        model_votes = Counter(
            max(
                {"gpt4": fp.get("model_prob_gpt4", 0),
                 "claude": fp.get("model_prob_claude", 0),
                 "human": fp.get("model_prob_human", 0)},
                key=lambda k: {"gpt4": fp.get("model_prob_gpt4", 0),
                               "claude": fp.get("model_prob_claude", 0),
                               "human": fp.get("model_prob_human", 0)}[k]
            )
            for fp in member_fps
        )

        # Coordination signal: how similar are the members?
        member_mat = np.array([
            [fingerprints[m].get(k, 0.0) for k in DISCRIMINATING_FEATURES]
            for m in members
        ])
        if len(members) >= 2:
            member_std = scaler.transform(member_mat)
            dists = euclidean_distances(member_std)
            avg_dist = dists[np.triu_indices_from(dists, k=1)].mean()
            coordination = max(0, 1.0 - avg_dist / 5.0)
        else:
            coordination = 0.5

        # Average LLM markers
        avg_llm = np.mean([fp.get("llm_phrase_density", 0) for fp in member_fps])
        avg_typo = np.mean([fp.get("typo_rate", 0) for fp in member_fps])

        cluster_summary[str(cid)] = {
            "member_count": len(members),
            "members": members,
            "dominant_model": model_votes.most_common(1)[0][0],
            "model_distribution": dict(model_votes),
            "coordination_signal": round(coordination, 4),
            "avg_llm_density": round(float(avg_llm), 4),
            "avg_typo_rate": round(float(avg_typo), 4),
            "is_bot_network": avg_llm > 0.05 or coordination > 0.7,
        }

    # Noise summary
    noise_ids = [aid for aid, label in zip(ids, labels) if label == -1]
    noise_summary = {
        "count": len(noise_ids),
        "accounts": noise_ids,
        "assessment": "organic" if len(noise_ids) > 0 else "none",
        "avg_model_prob_human": round(float(np.mean([
            fingerprints[a].get("model_prob_human", 0) for a in noise_ids
        ])), 4) if noise_ids else 0,
    }

    # Print report
    print("\n" + "=" * 70)
    print("STYLESHIELD ENHANCED DETECTION REPORT")
    print("=" * 70)
    print(f"  Accounts analyzed:  {len(accounts)}")
    print(f"  Bot networks found: {sum(1 for v in cluster_summary.values() if v['is_bot_network'])}")
    print(f"  Total clusters:     {len(cluster_summary)}")
    print(f"  Noise (organic):    {noise_summary['count']}")

    for cid, info in cluster_summary.items():
        bot_tag = "BOT NETWORK" if info["is_bot_network"] else "CLUSTER"
        print(f"\n  [{bot_tag}] Cluster {cid}: {info['member_count']} accounts")
        print(f"    Dominant model:    {info['dominant_model']}")
        print(f"    Coordination:      {info['coordination_signal']:.3f}")
        print(f"    Avg LLM density:   {info['avg_llm_density']:.4f}")
        print(f"    Avg typo rate:     {info['avg_typo_rate']:.4f}")
        print(f"    Members: {', '.join(info['members'][:5])}{'...' if len(info['members']) > 5 else ''}")

    if noise_ids:
        print(f"\n  [ORGANIC] Noise: {noise_summary['count']} accounts")
        print(f"    Avg human prob:    {noise_summary['avg_model_prob_human']:.3f}")
        print(f"    Accounts: {', '.join(noise_ids[:5])}{'...' if len(noise_ids) > 5 else ''}")

    # Export
    output_csv = "enhanced_results.csv"
    output_json = "enhanced_clusters.json"
    results_df.to_csv(output_csv, index=False)
    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(output_json, "w") as f:
        json.dump(_convert({
            "clusters": cluster_summary,
            "noise": noise_summary,
            "features_used": DISCRIMINATING_FEATURES,
            "parameters": {"epsilon": epsilon, "min_samples": min_samples},
        }), f, indent=2)
    print(f"\n  Exported: {output_csv}, {output_json}")

    return results_df, cluster_summary, noise_summary, graph_positions


if __name__ == "__main__":
    csv_files = sys.argv[1:] if len(sys.argv) > 1 else ["full_combined.csv"]
    run_enhanced_pipeline(csv_files, epsilon=1.5, min_samples=2)
