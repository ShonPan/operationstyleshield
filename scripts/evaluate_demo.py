"""Evaluate pipeline results against ground truth."""
import pandas as pd

results = pd.read_csv("demo_results.csv")
truth = pd.read_csv("demo_ground_truth.csv")

merged = results.merge(truth, on="account_id", how="inner")

# Classification: bot if confidence > 0.5 OR is_stealth_suspect
merged["predicted_bot"] = (merged["confidence"] > 0.5) | (merged["is_stealth_suspect"] == True)
merged["actual_bot"] = merged["ground_truth"] != "human"

tp = ((merged["predicted_bot"]) & (merged["actual_bot"])).sum()
fp = ((merged["predicted_bot"]) & (~merged["actual_bot"])).sum()
fn = ((~merged["predicted_bot"]) & (merged["actual_bot"])).sum()
tn = ((~merged["predicted_bot"]) & (~merged["actual_bot"])).sum()

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 0.001)

print("=" * 60)
print("EVALUATION vs GROUND TRUTH")
print("=" * 60)
print(f"  True Positives:  {tp} (bots correctly flagged)")
print(f"  False Positives: {fp} (humans wrongly flagged)")
print(f"  False Negatives: {fn} (bots missed)")
print(f"  True Negatives:  {tn} (humans correctly cleared)")
print()
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1 Score:  {f1:.3f}")

print()
print("DETECTION BY CATEGORY:")
print("-" * 60)
for gt in ["human", "bot", "stealth_bot"]:
    subset = merged[merged["ground_truth"] == gt]
    flagged = subset["predicted_bot"].sum()
    total = len(subset)
    print(f"  {gt:15s}: {flagged}/{total} flagged as bot ({flagged/max(total,1)*100:.0f}%)")

print()
print("STEALTH BOT DETAIL:")
print("-" * 60)
stealth = merged[merged["ground_truth"] == "stealth_bot"].sort_values("confidence", ascending=False)
for _, r in stealth.iterrows():
    flag = "CAUGHT" if r["predicted_bot"] else "MISSED"
    stealth_tag = " [stealth]" if r.get("is_stealth_suspect") else ""
    print(f"  {r['account_id']:25s} conf={r['confidence']:.3f} model={str(r['likely_model']):12s} {flag}{stealth_tag}")
