"""Human judge baseline: present accounts for manual labeling."""
import pandas as pd
import random
import os

# Load posts, group by account
env = pd.read_csv("demo_environment.csv")
accounts = list(env.groupby("account_id")["post_text"].apply(list).items())

# Shuffle so the human can't guess from ordering
random.seed(42)
random.shuffle(accounts)

# Load existing progress if any
output_file = "eval_human_judge_results.csv"
done = set()
if os.path.exists(output_file):
    existing = pd.read_csv(output_file)
    done = set(existing["account_id"].tolist())
    print(f"Resuming — {len(done)} accounts already labeled.\n")

remaining = [(aid, posts) for aid, posts in accounts if aid not in done]
total = len(accounts)
completed = len(done)

print("=" * 60)
print("HUMAN JUDGE — Bot Detection")
print("=" * 60)
print(f"Accounts: {total} total, {len(remaining)} remaining")
print("For each account, type:  b = bot,  h = human,  s = skip,  q = quit")
print("=" * 60)

for account_id, posts in remaining:
    completed += 1
    print(f"\n{'=' * 60}")
    print(f"ACCOUNT [{completed}/{total}]: {account_id}")
    print(f"{'=' * 60}")
    for j, post in enumerate(posts):
        print(f"\n  Post {j+1}:")
        # Indent each line of the post
        for line in post.split("\n"):
            print(f"    {line}")
    print()

    while True:
        choice = input("  Verdict (b=bot, h=human, s=skip, q=quit): ").strip().lower()
        if choice in ("b", "h", "s", "q"):
            break
        print("  Invalid input. Use b, h, s, or q.")

    if choice == "q":
        print("Quitting. Progress saved.")
        break
    if choice == "s":
        print("  Skipped.")
        continue

    prediction = "BOT" if choice == "b" else "HUMAN"

    # Append to CSV
    row = pd.DataFrame([{"account_id": account_id, "human_prediction": prediction}])
    row.to_csv(output_file, mode="a", header=not os.path.exists(output_file) or len(done) == 0 and completed == 1, index=False)
    done.add(account_id)
    print(f"  Recorded: {prediction}")

# Rewrite clean CSV at the end
if os.path.exists(output_file):
    df = pd.read_csv(output_file)
    df.drop_duplicates(subset="account_id", keep="last").to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} predictions to {output_file}")
