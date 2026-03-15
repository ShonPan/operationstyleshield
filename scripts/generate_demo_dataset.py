"""
Generate a clean demo dataset with known ground truth labels.
Blends human, bot, and stealth bot accounts for evaluation.

Outputs:
  demo_environment.csv  - account_id, post_text, posting_hour (pipeline input)
  demo_ground_truth.csv - account_id, ground_truth, source (for evaluation)
"""

import pandas as pd
import random
import os

random.seed(42)

OUTPUT_DIR = "."
DEMO_ENV = os.path.join(OUTPUT_DIR, "demo_environment.csv")
DEMO_TRUTH = os.path.join(OUTPUT_DIR, "demo_ground_truth.csv")

rows = []
truth = []

# ── 1. HUMAN ACCOUNTS (sample from airline tweets) ──────────────────────
print("Sampling human accounts from airline tweets...")
airline = pd.read_csv("data/training/human_airline_Tweets.csv")
# Find accounts with 3+ tweets for meaningful analysis
name_col = "name"
text_col = "text"
counts = airline.groupby(name_col).size()
good_accounts = counts[counts >= 3].index.tolist()
random.shuffle(good_accounts)
selected_humans = good_accounts[:15]

for acct in selected_humans:
    acct_rows = airline[airline[name_col] == acct]
    for _, r in acct_rows.iterrows():
        hour = random.randint(6, 23)  # simulate posting hours
        rows.append({
            "account_id": f"human_{acct}",
            "post_text": str(r[text_col]),
            "posting_hour": hour,
        })
    truth.append({
        "account_id": f"human_{acct}",
        "ground_truth": "human",
        "source": "airline_tweets",
        "post_count": len(acct_rows),
    })

print(f"  Added {len(selected_humans)} human accounts")

# ── 2. GPT-4 BOT ACCOUNTS (from chatgpt_bot_tweets) ─────────────────────
print("Adding GPT-4 bot accounts...")
bots = pd.read_csv("data/test/chatgpt_bot_tweets.csv")
bot_accounts = bots["account_id"].unique().tolist()
random.shuffle(bot_accounts)
selected_bots = bot_accounts[:10]

for acct in selected_bots:
    acct_rows = bots[bots["account_id"] == acct]
    for _, r in acct_rows.iterrows():
        rows.append({
            "account_id": str(r["account_id"]),
            "post_text": str(r["post_text"]),
            "posting_hour": int(r["posting_hour"]),
        })
    truth.append({
        "account_id": str(acct),
        "ground_truth": "bot",
        "source": "chatgpt_bot_tweets",
        "post_count": len(acct_rows),
    })

print(f"  Added {len(selected_bots)} GPT-4 bot accounts")

# ── 3. HAIKU-STYLE BOT ACCOUNTS ─────────────────────────────────────────
print("Adding haiku-style bot accounts...")
haiku = pd.read_csv("data/test/haikustyle_bot_tweets.csv")
username_col = "username"
tweet_col = "tweet_text"
haiku_accounts = haiku[username_col].unique().tolist()
random.shuffle(haiku_accounts)
selected_haiku = haiku_accounts[:8]

for acct in selected_haiku:
    acct_rows = haiku[haiku[username_col] == acct]
    for _, r in acct_rows.iterrows():
        hour = random.choice([9, 10, 11, 14, 15, 16])  # bot-like hours
        rows.append({
            "account_id": str(r[username_col]),
            "post_text": str(r[tweet_col]),
            "posting_hour": hour,
        })
    truth.append({
        "account_id": str(acct),
        "ground_truth": "bot",
        "source": "haikustyle_bot_tweets",
        "post_count": len(acct_rows),
    })

print(f"  Added {len(selected_haiku)} haiku-style bot accounts")

# ── 4. STEALTH BOT ACCOUNTS ─────────────────────────────────────────────
print("Adding stealth bot accounts...")
stealth = pd.read_csv("data/test/stealth_bots.csv")
stealth_accounts = stealth["account_id"].unique().tolist()
# Take all 20 stealth accounts
for acct in stealth_accounts:
    acct_rows = stealth[stealth["account_id"] == acct]
    for _, r in acct_rows.iterrows():
        rows.append({
            "account_id": str(r["account_id"]),
            "post_text": str(r["post_text"]),
            "posting_hour": int(r["posting_hour"]),
        })
    # Extract operator ID for ground truth
    operator = "_".join(acct.split("_")[:3])  # e.g. stealth_op_a
    truth.append({
        "account_id": str(acct),
        "ground_truth": "stealth_bot",
        "source": f"stealth_bots/{operator}",
        "post_count": len(acct_rows),
    })

print(f"  Added {len(stealth_accounts)} stealth bot accounts")

# ── SHUFFLE AND SAVE ─────────────────────────────────────────────────────
random.shuffle(rows)
df_env = pd.DataFrame(rows)
df_truth = pd.DataFrame(truth)

df_env.to_csv(DEMO_ENV, index=False)
df_truth.to_csv(DEMO_TRUTH, index=False)

# Summary
print(f"\n{'='*50}")
print(f"DEMO DATASET SUMMARY")
print(f"{'='*50}")
counts = df_truth["ground_truth"].value_counts()
for label, count in counts.items():
    print(f"  {label:15s}: {count} accounts")
print(f"  {'TOTAL':15s}: {len(df_truth)} accounts")
print(f"  Total posts:     {len(df_env)}")
print(f"\nSaved: {DEMO_ENV}")
print(f"       {DEMO_TRUTH}")
