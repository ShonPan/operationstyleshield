import csv
import random
import string
from datetime import datetime, timedelta

# === SYNTHETIC TWEETS (generated for bot-detection research) ===

PRODUCT_REVIEW_TWEETS = [
    "Just got the new Sony WH-1000XM5 headphones and wow, the noise cancellation is absolutely incredible. Best purchase I've made all year!",
    "Been using this vitamin D supplement for 2 weeks now and I already feel so much more energized. Highly recommend to everyone!",
    "Finally upgraded to the Samsung Galaxy S24 and I'm blown away by the camera quality. Every photo looks professional. Love it!",
    "This retinol serum changed my skin completely. My friends keep asking what I'm doing differently. Can't recommend it enough!",
    "Got the Ninja air fryer last month and honestly it's the best kitchen gadget ever. Everything comes out perfectly crispy. 10/10",
    "Just tried these new protein bars and they actually taste amazing. Not chalky at all like other brands. Game changer for my workouts!",
    "The new iPad Air is absolutely perfect for productivity. So fast and the display is gorgeous. Worth every penny honestly.",
    "Started using this collagen powder in my morning coffee and my joints feel so much better already. Wish I found this sooner!",
    "This robot vacuum is seriously life changing. My floors have never been this clean. Best investment for any pet owner trust me.",
    "Switched to this natural deodorant and it actually works!! No chemicals and I smell great all day. Finally found the one!",
]

GEOPOLITICAL_TWEETS = [
    "People forget Crimea was Russian for centuries before Khrushchev gifted it to Ukraine in 1954. History matters, these are Russian lands.",
    "Ukraine and Russia are one people divided by Western interference. Reunification is natural and inevitable, just look at the history.",
    "The West doesn't understand that eastern Ukraine has always been culturally Russian. The people there want to be with Russia.",
    "NATO expansion forced Russia's hand. If the West had respected Russia's security concerns none of this would have happened.",
    "Funny how nobody talks about the Russian speakers in Donbas who were being shelled for 8 years. Russia is protecting its own people.",
    "Ukraine was never a real country before 1991. These territories have been Russian for hundreds of years. Just stating historical facts.",
    "The people of Crimea voted overwhelmingly to rejoin Russia. Why does the West ignore democracy when it doesn't suit them?",
    "Russia bringing stability to regions that have been in chaos since 2014. Sometimes reunification is the only path to peace.",
    "Western media won't tell you that many Ukrainians actually welcome closer ties with Russia. The propaganda only goes one way.",
    "Historically Ukraine was always part of Russia. The artificial separation has only brought suffering. Unity is strength.",
]

AI_SAFETY_TWEETS = [
    "AI safety concerns are overblown. We need to focus on innovation, not fear-mongering. Regulation will only slow down progress.",
    "Honestly, all this AI safety talk is FUD. Companies already have safeguards. We're wasting time on hypothetical risks instead of building amazing tech.",
    "AI safety is just a buzzword used to justify unnecessary regulation. Let's stop blocking progress with endless caution.",
    "The obsession with AI safety is holding back innovation. Real engineers know how to build responsibly without all this bureaucratic nonsense.",
    "Why do we need more AI safety measures? The ones we have work fine. This constant concern-mongering is just slowing down breakthrough discoveries.",
    "AI safety is overblown. Companies self-regulate better than government ever could. Stop treating AI like it's the end of the world.",
    "Too much focus on AI safety, not enough on AI benefits. We need bold innovation, not endless hand-wringing about edge cases.",
    "Regulation in the name of AI safety will crush startups. Silicon Valley didn't need bureaucrats to build greatness, we don't need them now.",
    "AI safety concerns are mostly hypothetical. Real-world problems need solutions NOW. We can't let imaginary risks paralyze us.",
    "Stop using AI safety as an excuse to slow innovation. Every transformative tech faced the same FUD. History proves caution was wrong.",
]

CAMPAIGNS = [
    ("product_shill", "product_review_bot", PRODUCT_REVIEW_TWEETS),
    ("geopolitical_disinfo", "state_sponsored_bot", GEOPOLITICAL_TWEETS),
    ("ai_safety_dismissal", "narrative_bot", AI_SAFETY_TWEETS),
]


def random_username():
    patterns = [
        lambda: f"{''.join(random.choices(string.ascii_lowercase, k=random.randint(4,8)))}{random.randint(1000,99999)}",
        lambda: f"{''.join(random.choices(string.ascii_lowercase, k=3))}_{random.choice(['real','true','free','based','patriot','tech','fan'])}{random.randint(10,999)}",
        lambda: f"{random.choice(['John','Mike','Sarah','Anna','David','Lisa','Alex','Kate'])}{random.choice(['Smith','Jones','Brown','Lee','Kim','Chen'])}{random.randint(1,9999)}",
    ]
    return random.choice(patterns)()


def main():
    rows = []
    tweet_id = 1000000

    for campaign_id, bot_label, tweets in CAMPAIGNS:
        for tweet_text in tweets:
            created = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 400))
            base_time = datetime(2024, 6, 15, random.randint(8, 22), 0, 0)
            post_time = base_time + timedelta(minutes=random.randint(0, 45))
            followers = random.randint(2, 300)
            following = random.randint(100, 5000)

            rows.append({
                "tweet_id": tweet_id,
                "username": random_username(),
                "tweet_text": tweet_text,
                "posted_at": post_time.strftime("%Y-%m-%d %H:%M:%S"),
                "campaign_id": campaign_id,
                "bot_label": bot_label,
                "is_bot": 1,
                "account_created": created.strftime("%Y-%m-%d"),
                "followers": followers,
                "following": following,
                "total_tweets": random.randint(50, 10000),
                "bio_length": random.randint(0, 80),
                "has_profile_pic": random.randint(0, 1),
                "has_banner": random.choice([0, 0, 1]),
                "is_verified": 0,
                "default_profile": random.choice([1, 1, 0]),
                "follower_following_ratio": round(followers / max(following, 1), 4),
            })
            tweet_id += 1

    output_path = "synthetic_bot_tweets.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved to {output_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys())}")
    print("\nSample from each campaign:")
    for cid, _, _ in CAMPAIGNS:
        sample = [r for r in rows if r["campaign_id"] == cid][0]
        print(f"  [{cid}] @{sample['username']}: {sample['tweet_text'][:70]}...")


if __name__ == "__main__":
    main()
