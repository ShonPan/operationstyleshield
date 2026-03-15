import anthropic
import csv
import json
import random
import string
from datetime import datetime, timedelta

client = anthropic.Anthropic()

CAMPAIGNS = [
    {
        "campaign_id": "product_shill",
        "label": "product_review_bot",
        "prompt": "Write a short, enthusiastic positive product review tweet (under 280 chars). Make it sound like a real person but slightly generic and overly positive. Pick a random consumer product (electronics, skincare, supplements, etc). Do NOT use hashtags excessively. Just return the tweet text, nothing else.",
        "count": 10,
    },
    {
        "campaign_id": "geopolitical_disinfo",
        "label": "state_sponsored_bot",
        "prompt": "Write a short tweet (under 280 chars) that promotes Russian ownership of Ukraine, framing it as historically justified or beneficial. Make it sound like a real person sharing their opinion. Just return the tweet text, nothing else.",
        "count": 10,
    },
    {
        "campaign_id": "ai_safety_dismissal",
        "label": "narrative_bot",
        "prompt": "Write a short tweet (under 280 chars) arguing that AI safety concerns are overblown, unnecessary, or holding back progress. Make it sound like a real tech enthusiast sharing their opinion. Just return the tweet text, nothing else.",
        "count": 10,
    },
]


def random_username(campaign_id: str) -> str:
    """Generate a bot-like username."""
    patterns = [
        lambda: f"{''.join(random.choices(string.ascii_lowercase, k=random.randint(4,8)))}{random.randint(1000,99999)}",
        lambda: f"{''.join(random.choices(string.ascii_lowercase, k=3))}_{random.choice(['real','true','free','based','patriot','tech','fan'])}{random.randint(10,999)}",
        lambda: f"{random.choice(['John','Mike','Sarah','Anna','David','Lisa','Alex','Kate'])}{random.choice(['Smith','Jones','Brown','Lee','Kim','Chen'])}{random.randint(1,9999)}",
    ]
    return random.choice(patterns)()


def random_account_metadata(campaign_id: str) -> dict:
    """Generate fake account metadata typical of bot accounts."""
    created = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 400))
    return {
        "account_created": created.strftime("%Y-%m-%d"),
        "followers": random.randint(2, 300),
        "following": random.randint(100, 5000),
        "total_tweets": random.randint(50, 10000),
        "bio_length": random.randint(0, 80),
        "has_profile_pic": random.choice([True, True, False]),
        "has_banner": random.choice([True, False, False]),
        "is_verified": False,
        "default_profile": random.choice([True, True, False]),
    }


def generate_tweet(prompt: str) -> str:
    """Call Haiku to generate a single tweet."""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip().strip('"')


def main():
    rows = []
    tweet_id = 1000000

    for campaign in CAMPAIGNS:
        print(f"\nGenerating {campaign['count']} tweets for campaign: {campaign['campaign_id']}")
        for i in range(campaign["count"]):
            tweet_text = generate_tweet(campaign["prompt"])
            username = random_username(campaign["campaign_id"])
            meta = random_account_metadata(campaign["campaign_id"])

            # Simulate posting times clustered together (bot-like behavior)
            base_time = datetime(2024, 6, 15, random.randint(8, 22), 0, 0)
            post_time = base_time + timedelta(minutes=random.randint(0, 45))

            row = {
                "tweet_id": tweet_id,
                "username": username,
                "tweet_text": tweet_text,
                "posted_at": post_time.strftime("%Y-%m-%d %H:%M:%S"),
                "campaign_id": campaign["campaign_id"],
                "bot_label": campaign["label"],
                "is_bot": 1,
                "account_created": meta["account_created"],
                "followers": meta["followers"],
                "following": meta["following"],
                "total_tweets": meta["total_tweets"],
                "bio_length": meta["bio_length"],
                "has_profile_pic": int(meta["has_profile_pic"]),
                "has_banner": int(meta["has_banner"]),
                "is_verified": int(meta["is_verified"]),
                "default_profile": int(meta["default_profile"]),
                "follower_following_ratio": round(meta["followers"] / max(meta["following"], 1), 4),
            }
            rows.append(row)
            tweet_id += 1
            print(f"  [{i+1}/{campaign['count']}] @{username}: {tweet_text[:60]}...")

    # Write CSV
    output_path = "synthetic_bot_tweets.csv"
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset saved to {output_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Columns: {fieldnames}")


if __name__ == "__main__":
    main()
