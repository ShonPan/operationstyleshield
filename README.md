# StyleShield
### Botnet detection via stylometric anomaly analysis

**Live demo:** [operationstyleshield.vercel.app](https://operationstyleshield.vercel.app/)
**Backend API:** [operationstyleshield-production.up.railway.app](https://operationstyleshield-production.up.railway.app)
**GitHub:** [github.com/ShonPan/operationstyleshield](https://github.com/ShonPan/operationstyleshield)

---

## What is StyleShield?

StyleShield detects coordinated inauthentic behavior by analyzing the natural topology of language; this tends to capture botnets. Human writing will be pretty varied and messy, with a kind of predictable texture with varied vocabulary, inconsistent rhythm, personal quirks and lots of natural entropy. If multiple accounts produce writing that deviates from this natural gradient in the same way, it forms a kind of "artificial structure" in the language space, which allows us to identify what may be a botnet. We believe that this is a particularly concerning issue for cognitive security at this time, see Doublespeed(https://futurism.com/artificial-intelligence/doublespeed-ai-phone-farm).

This is a form of "transferred leanring" from cstr's  **Xenarch**. This is a planetary technosignature detection system that distinguishes natural lunar features from artificial ones and finds. the Apollo 11 lunar lander at 99.58% confidence. StyleShield is a theory that applies the same multi-metric anomaly detection to social media.

---

## Try it now

1. Go to **[operationstyleshield.vercel.app](https://operationstyleshield.vercel.app/)**
2. Click **"Load demo"** to analyze the bundled 53-account dataset
3. Watch the console stream real analysis output
4. Explore coordinated networks — click clusters to see evidence
5. Click **"Reveal ground truth"** to see detection accuracy
6. Upload your own CSV to analyze any dataset

---

## Run locally

**Prerequisites:** Python 3.10+, Node.js 18+

**Backend:**
```bash
git clone https://github.com/ShonPan/operationstyleshield.git
cd operationstyleshield
pip3 install flask flask-cors numpy pandas scikit-learn scipy
python3 api.py
```
API starts on `http://localhost:5000`

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Opens at `http://localhost:5173`

---

## CSV format

Upload any CSV with three columns:

```
account_id,post_text,posting_hour
jane42,"honestly this coffee shop is amazing lol",9
user_99,"Certainly the benefits are clear. Moreover the quality is high.",14
```

Each account should have 3+ posts for reliable analysis.

---

## How it works

StyleShield extracts 40+ stylometric features per account, clusters accounts by writing fingerprint similarity using DBSCAN, and identifies coordinated networks that no single-post classifier can detect.

**Detection is purely structural.** No labels, no metadata, no content moderation. If accounts write too similarly across enough dimensions, they cluster — and that coordination is the signal.

**Results include:**
- Coordinated network identification with coordination scores
- Model inference (GPT-4 detected, suspected LLM, unknown origin)
- Campaign narrative analysis (what topics the network is pushing)
- Per-account evidence explaining why each account was flagged or cleared

---

## Key results

In testing against stealth bots — AI-generated accounts prompted to mimic human writing:

| | StyleShield | Naive LLM judge |
|---|---|---|
| F1 Score | **0.758** | 0.408 |
| Stealth bots caught | **65%** | 0% |
| Overall threats caught | **75%** | 30% |

Perhaps most interestingly, these 'stealth accounts' tend to evade naive LLM as a judge but is found in this method. One regression is that this mildly increases the odds of false positives of humans classified as bots.

---

## Project structure

```
operationstyleshield/
├── api.py                  # Flask backend API
├── core/                   # Analysis engine
│   ├── Styleshield_script.py    # Base multi-metric scorer
│   ├── enhanced_extractor.py    # 40+ stylometric features
│   └── enhanced_pipeline.py     # Full pipeline with DBSCAN + t-SNE
├── frontend/               # React + Vite dashboard
├── demo/                   # Demo datasets + ground truth
└── docs/                   # Documentation + specs
```

---

## Built at

Frontier Tower Hackathon | March 14–15, 2026 | AI Safety & Evaluation Track (Protocol Labs)
