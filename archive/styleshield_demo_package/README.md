# StyleShield

**Exposing Bot Networks Through Stylometric Anomaly Detection**

*Frontier Tower Hackathon — March 14–15, 2026*

---

## What It Does

StyleShield detects coordinated bot networks by analyzing writing fingerprints. Instead of looking at behavioral signals that bots learn to fake (follower ratios, posting frequency, account age), StyleShield examines the **stylometric layer** — the writing patterns that LLMs can't hide.

Feed it a set of social media accounts. It tells you how many real authors are behind them.

**500 "independent" accounts. 4 writing signatures. That's not grassroots. That's a 4-person operation.**

## How It Works

StyleShield applies anomaly detection principles from planetary science ([Xenarch](https://github.com/your-link-here) — a system that finds artificial structures in lunar imagery) to text analysis.

### Three Detection Layers

1. **Individual Authenticity** — Is this account's writing AI-generated or human? Measures vocabulary uniformity, hedging patterns, structural regularity.
2. **Cross-Account Clustering** — Are these "independent" accounts actually the same author/model? DBSCAN clustering on stylometric fingerprints reveals coordination.
3. **Infrastructure Fingerprinting** — Temporal analysis reveals shift patterns, time zones, and device types behind bot farm operations.

### Five-Metric Scoring (Xenarch-Adapted)

| Metric | Weight | Bot Signal | Xenarch Parallel |
|--------|--------|-----------|-----------------|
| Vocabulary Uniformity | 25% | AI models draw from narrower distributions than humans | MSE (Reconstruction Error) |
| Structural Regularity | 25% | Bots are unnaturally consistent in sentence structure | Edge Regularity |
| Hedging & Filler Signature | 20% | Each LLM family has characteristic hedging patterns | Latent Density |
| Temporal-Contextual | 20% | Bot farms post in shifts, not human rhythms | Contextual Metric |
| Cross-Account Correlation | 10% | "Independent" accounts with high similarity = same operator | Gradient Anomaly |

**Combined bot score:** `B = 0.25·VU + 0.25·SR + 0.20·HF + 0.20·TC + 0.10·CA`

## Quick Start

### Requirements

```bash
pip install numpy pandas scikit-learn scipy anthropic
```

### Run on built-in synthetic data

```bash
python Styleshield_script.py
```

### Run on your own CSV

```bash
python Styleshield_script.py your_accounts.csv
```

### Generate a blank CSV template

```bash
python Styleshield_script.py --template
```

### Tune DBSCAN parameters

```bash
python Styleshield_script.py --epsilon 0.30 --min-samples 2 your_data.csv
```

## CSV Input Formats

### Long format (recommended) — one post per row:

```csv
account_id,post_text,posting_hour
user_001,"Certainly! This product is exceptional.",9
user_001,"Furthermore, the quality is notable.",10
user_002,"omg just got this and its amazing!!",21
```

### Wide format — one account per row:

```csv
account_id,post_1,post_2,posting_hours
user_001,"First post here","Second post here","9;10;14"
user_002,"Only post","","21"
```

Column names are matched case-insensitively. Common aliases accepted: `user_id`, `username`, `handle`, `text`, `content`, `tweet`, `timestamp`, `created_at`, etc.

## Output

- **`styleshield_results.csv`** — Per-account scores: bot probability, confidence, cluster assignment, per-metric breakdown
- **`styleshield_results_clusters.json`** — Cluster details: member accounts, binding features, coordination signal strength

### Example Output

```
============================================================
DETECTION SUMMARY
============================================================
Accounts analyzed:    30
Bot networks found:   2
Accounts in networks: 20
High confidence >0.8: 18

Cluster 0: 10 accounts (coordination: 0.952)
  Members: gpt4_account_00, gpt4_account_01, ...
  Binding: intra_vocab_variance, intra_structure_variance, intra_hedge_variance

Cluster 1: 10 accounts (coordination: 0.948)
  Members: claude_account_00, claude_account_01, ...
  Binding: intra_vocab_variance, intra_hedge_variance, intra_structure_variance
```

## Architecture

```
Raw Posts ──► Stylometric Extractor ──► Multi-Metric Bot Scorer ──► DBSCAN Clustering ──► Adaptive Confidence
                  │                           │                          │                       │
             25 features              5 normalized metrics         Cosine similarity       Xenarch-style
             per account              weighted & combined          + cluster labels        context-aware
```

### Components

| Module | Class | Role |
|--------|-------|------|
| Feature Extraction | `StylometricExtractor` | Extracts 25 writing fingerprint features from post corpus |
| Bot Scoring | `MultiMetricBotScorer` | Five normalized metrics → combined bot score |
| Clustering | `AccountClusterAnalyzer` | DBSCAN on cosine distance in stylometric space |
| Confidence | `AdaptiveConfidenceCalculator` | Context-aware confidence (ported from Xenarch) |
| Data Loading | `CSVLoader` | Auto-detects CSV format, handles aliases |
| Pipeline | `StyleShieldScorer` | End-to-end orchestration |

## The Xenarch Connection

StyleShield is adapted from [Xenarch](link), a system that detects artificial structures (like the Apollo 11 lunar lander) in planetary imagery using a VAE trained exclusively on natural geology. Both systems share the same core insight:

> **Train on what's natural. Flag what doesn't belong.**

Xenarch trains on craters and flags landers. StyleShield trains on human writing diversity and flags artificial uniformity. The multi-metric scoring, DBSCAN clustering, and adaptive confidence calculations are direct ports from Xenarch's planetary detection pipeline.

| Xenarch (Planetary) | StyleShield (Text) |
|--------------------|--------------------|
| Train on natural geology | Train on human writing diversity |
| Flag artificial structures | Flag artificial coordination |
| Multi-metric anomaly score | Multi-metric bot score |
| Spatial DBSCAN clustering | Stylometric DBSCAN clustering |
| Adaptive confidence (Eq. 4–5) | Adaptive confidence (ported) |
| 99.58% confidence on Apollo 11 lander | Detects bot networks from writing alone |

## Track

**AI Safety & Evaluation** — Protocol Labs challenge

> "Build infrastructure to close the gap between AI deployment and safety verification. Projects should surface concrete vulnerabilities — evaluations that find real failures, runtime monitoring systems, and auditing tools for real agents."

StyleShield surfaces a concrete AI safety vulnerability: LLM-powered influence operations that evade current detection. It audits the output of AI agents deployed at scale on social media.

## Team

- **Shon Pan** — Product vision, Claude API prompts, presentation
- **Caleb Strom** — Anomaly detection architecture, scoring engine, Xenarch adaptation
- **Abdul** — Frontend, UI/UX, integration

## License

MIT
