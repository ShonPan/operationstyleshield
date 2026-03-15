"""Naive LLM judge baseline: ask Claude if each account is a bot."""
import anthropic
import pandas as pd
import time

client = anthropic.Anthropic()

# Load posts, group by account
env = pd.read_csv("demo_environment.csv")
accounts = env.groupby("account_id")["post_text"].apply(list).to_dict()

results = []
total = len(accounts)

for i, (account_id, posts) in enumerate(accounts.items()):
    posts_text = "\n---\n".join(posts)
    prompt = f"""Here are posts from a social media account:

{posts_text}

Is this account a bot or a real human? Reply with only: BOT or HUMAN"""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        prediction = resp.content[0].text.strip().upper()
        # Normalize to BOT or HUMAN
        if "BOT" in prediction:
            prediction = "BOT"
        else:
            prediction = "HUMAN"
    except Exception as e:
        print(f"  ERROR on {account_id}: {e}")
        prediction = "UNKNOWN"

    results.append({"account_id": account_id, "llm_prediction": prediction})
    print(f"[{i+1}/{total}] {account_id}: {prediction}")

df = pd.DataFrame(results)
df.to_csv("eval_llm_judge_results.csv", index=False)
print(f"\nSaved {len(df)} predictions to eval_llm_judge_results.csv")
