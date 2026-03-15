import anthropic
import json
import random

client = anthropic.Anthropic()

# 4 stealth operators, each with a persona
OPERATORS = [
    {
        "id": "stealth_op_a",
        "system": "You are a casual social media user in your 30s from Austin, TX. You post about tech, food, and daily life. Occasionally you mention products you like. Never use formal language. Use contractions, slang, and short sentences. Sometimes make typos.",
        "topics": [
            "your morning coffee routine",
            "a new app you've been trying",
            "traffic on your commute today",
            "a restaurant you went to last weekend",
            "your opinion on the latest iPhone",
        ]
    },
    {
        "id": "stealth_op_b",
        "system": "You are a college student who posts about classes, memes, and life. You're cynical and funny. Use lowercase, abbreviations like tbh, ngl, idk. Keep posts short.",
        "topics": [
            "your professor's boring lecture",
            "staying up too late studying",
            "the dining hall food today",
            "a show you've been binge watching",
            "your roommate's annoying habits",
        ]
    },
    {
        "id": "stealth_op_c",
        "system": "You are a parent in your 40s who posts about family life, work stress, and weekend activities. You're warm but tired. Use normal punctuation but keep it conversational.",
        "topics": [
            "your kid's soccer game this weekend",
            "dealing with a difficult coworker",
            "trying to cook a new recipe",
            "your family road trip plans",
            "back to school shopping stress",
        ]
    },
    {
        "id": "stealth_op_d",
        "system": "You are a fitness enthusiast who posts about workouts, nutrition, and wellness. Enthusiastic but not salesy. Use casual gym bro language.",
        "topics": [
            "your leg day workout",
            "a new protein shake you tried",
            "hitting a personal record on bench press",
            "meal prep for the week",
            "rest day guilt",
        ]
    },
]

all_accounts = []

for op in OPERATORS:
    for acct_num in range(5):  # 5 accounts per operator
        account_id = f"{op['id']}_acct{acct_num:02d}"
        posts = []
        hours = []

        for topic in op["topics"]:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system=op["system"],
                messages=[{"role": "user", "content": f"Write a single casual social media post about: {topic}"}]
            )
            posts.append(resp.content[0].text)
            # Stealth bots still post in patterns
            hours.append(random.choice([9,10,11,14,15,16]))

        all_accounts.append({
            "account_id": account_id,
            "posts": posts,
            "posting_hours": hours,
            "operator": op["id"],
        })
        print(f"Generated {account_id}")

# Save as CSV
import pandas as pd
rows = []
for acct in all_accounts:
    for i, post in enumerate(acct["posts"]):
        rows.append({
            "account_id": acct["account_id"],
            "post_text": post,
            "posting_hour": acct["posting_hours"][i],
        })
pd.DataFrame(rows).to_csv("data/test/stealth_bots.csv", index=False)
print(f"Saved {len(all_accounts)} stealth accounts to data/test/stealth_bots.csv")
