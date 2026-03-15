"""Compare all detection methods: StyleShield vs LLM Judge vs Human Judge."""
import pandas as pd
import os


def compute_metrics(y_true, y_pred):
    """Compute precision, recall, F1 for bot detection."""
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision, "Recall": recall, "F1": f1}


# Load ground truth
truth = pd.read_csv("demo_ground_truth.csv")
truth["actual_bot"] = truth["ground_truth"] != "human"

# --- StyleShield ---
results = pd.read_csv("demo_results.csv")
results["styleshield_bot"] = (results["confidence"] > 0.5) | (results["is_stealth_suspect"] == True)
ss = truth.merge(results[["account_id", "styleshield_bot"]], on="account_id", how="left")

# --- LLM Judge ---
llm_file = "eval_llm_judge_results.csv"
has_llm = os.path.exists(llm_file)
if has_llm:
    llm = pd.read_csv(llm_file)
    llm["llm_bot"] = llm["llm_prediction"] == "BOT"
    ss = ss.merge(llm[["account_id", "llm_bot"]], on="account_id", how="left")

# --- Human Judge ---
human_file = "eval_human_judge_results.csv"
has_human = os.path.exists(human_file)
if has_human:
    hj = pd.read_csv(human_file)
    hj["human_bot"] = hj["human_prediction"] == "BOT"
    ss = ss.merge(hj[["account_id", "human_bot"]], on="account_id", how="left")

merged = ss

# Build methods list
methods = [("StyleShield", "styleshield_bot")]
if has_llm:
    methods.append(("LLM Judge", "llm_bot"))
if has_human:
    methods.append(("Human Judge", "human_bot"))

# === Overall Comparison ===
print("=" * 70)
print("SIDE-BY-SIDE COMPARISON: Bot Detection Methods")
print("=" * 70)

header = f"{'Metric':<12}"
for name, _ in methods:
    header += f" {name:>14}"
print(header)
print("-" * 70)

all_metrics = {}
for name, col in methods:
    valid = merged.dropna(subset=[col])
    m = compute_metrics(valid["actual_bot"].tolist(), valid[col].tolist())
    m["N"] = len(valid)
    all_metrics[name] = m

for metric in ["Precision", "Recall", "F1"]:
    row = f"{metric:<12}"
    for name, _ in methods:
        row += f" {all_metrics[name][metric]:>14.3f}"
    print(row)

row = f"{'TP/FP/FN':<12}"
for name, _ in methods:
    m = all_metrics[name]
    row += f" {m['TP']:>3}/{m['FP']:>2}/{m['FN']:>2}     "
print(row)

row = f"{'Accounts':<12}"
for name, _ in methods:
    row += f" {all_metrics[name]['N']:>14}"
print(row)

# === Breakdown by Category ===
print()
print("=" * 70)
print("DETECTION RATE BY CATEGORY")
print("=" * 70)

for category in ["human", "bot", "stealth_bot"]:
    subset = merged[merged["ground_truth"] == category]
    total = len(subset)
    print(f"\n  {category.upper()} ({total} accounts):")
    for name, col in methods:
        valid = subset.dropna(subset=[col])
        flagged = valid[col].sum()
        n = len(valid)
        pct = flagged / max(n, 1) * 100
        label = "correctly flagged" if category != "human" else "wrongly flagged"
        print(f"    {name:<14}: {int(flagged):>2}/{n} {label} ({pct:.0f}%)")

# === Per-account detail for stealth bots ===
print()
print("=" * 70)
print("STEALTH BOT DETAIL")
print("=" * 70)

stealth = merged[merged["ground_truth"] == "stealth_bot"].sort_values("account_id")
for _, r in stealth.iterrows():
    line = f"  {r['account_id']:<28}"
    line += f" SS:{'Y' if r.get('styleshield_bot') else 'N'}"
    if has_llm and pd.notna(r.get("llm_bot")):
        line += f"  LLM:{'Y' if r['llm_bot'] else 'N'}"
    if has_human and pd.notna(r.get("human_bot")):
        line += f"  HUM:{'Y' if r['human_bot'] else 'N'}"
    print(line)
