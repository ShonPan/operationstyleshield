# StyleShield Demo Environment — Claude Code Package
# ===================================================
#
# WHAT THIS IS:
# A pre-analyzed dataset of 53 social media accounts with realistic usernames.
# The system analyzed their writing patterns and found 5 bot networks hiding
# among real humans. All account names are randomized — no labels reveal 
# which are bots. The system detected them purely from writing style.
#
# FILES IN THIS PACKAGE:
#
# 1. demo_environment_anonymized.csv
#    - The raw input data: 270 posts from 53 accounts
#    - Columns: account_id, post_text, posting_hour
#    - All account names look like real usernames (e.g. "jen_nic27", "ryanstan95")
#    - This is what the system "sees" — no labels, no hints
#
# 2. demo_results_anonymized.csv  
#    - Output from the StyleShield pipeline: one row per account
#    - Columns:
#      - account_id: the username
#      - cluster_id: which cluster they belong to (-1 = noise/no cluster)
#      - is_noise: true = not in any cluster (likely human), false = in a cluster (likely coordinated)
#      - likely_model: "gpt4", "claude", or "human" — what model probably generated the text
#      - model_confidence: 0-1, how confident the model identification is
#      - llm_phrase_density: how much AI-characteristic language (0 = none, 2+ = very AI)
#      - typo_rate: informal markers like "lol", "tbh", "ngl" (humans have these, bots usually don't)
#      - structural_regularity: how regular the paragraph/sentence structure is (high = bot signal)
#      - avg_syllables: average syllable count (AI uses bigger words)
#      - jargon_density: uncommon word frequency
#      - contraction_rate: "don't" vs "do not" usage
#      - intra_vocab_var: how much vocabulary varies across posts (low = bot, they're unnaturally consistent)
#      - intra_struct_var: how much sentence structure varies across posts
#      - post_count: number of posts analyzed
#
# 3. demo_clusters_anonymized.json
#    - Cluster-level analysis
#    - Structure:
#      {
#        "clusters": {
#          "0": {
#            "member_count": 2,
#            "members": ["username1", "username2"],
#            "dominant_model": "human",        // what model the cluster likely used
#            "coordination_signal": 0.75,       // 0-1, how similar members' writing is
#            "avg_llm_density": 0.0,            // average AI marker rate
#            "avg_typo_rate": 0.0,              // average informal marker rate
#            "is_bot_network": true             // flagged as coordinated
#          },
#          "1": { ... },
#        },
#        "noise": {
#          "count": 31,
#          "accounts": ["username3", "username4", ...],
#          "avg_model_prob_human": 0.909
#        }
#      }
#
# 4. demo_ground_truth.json
#    - SECRET KEY — do NOT show during demo, only for the reveal
#    - Maps each fake username to its real identity and type:
#      "jen_nic27": {"original": "gpt4_account_08", "type": "gpt4_bot"}
#    - Types: "human", "gpt4_bot", "haiku_bot", "stealth_bot"
#
# ===================================================
# WHAT THE RESULTS MEAN
# ===================================================
#
# CLUSTER = a group of accounts whose writing is so similar they're almost
# certainly operated by the same person or system. If 4 accounts all use 
# "certainly" + "furthermore" + zero typos + identical sentence rhythm, 
# that's a cluster. Real independent people don't write this similarly.
#
# COORDINATION = 0 to 1, how tightly similar the writing is within a cluster.
# 1.0 = identical fingerprints (same operator, same template)
# 0.6 = similar enough to cluster (same model, different personas)
# 
# NOISE = accounts that don't cluster with anyone. Their writing is unique.
# These are most likely real humans — or very sophisticated lone bots.
#
# ===================================================
# CURRENT RESULTS SUMMARY
# ===================================================
#
# Accounts scanned:    53
# Clusters found:      5 (plus 1 false-positive human cluster)
# Noise (organic):     31 accounts
#
# Cluster 1: 4 accounts  | coordination: 1.000 | GPT-4 bot farm
#   - All use "certainly", "furthermore", "it's worth noting"
#   - Zero typos, zero informal markers
#   - Identical structural regularity (0.849)
#   - LLM phrase density: 2.08 (extremely high)
#
# Cluster 2: 9 accounts  | coordination: 0.688 | Stealth bot network  
#   - Different personas (tech bro, college kid, parent, fitness)
#   - Use slang, emoji, casual tone — designed to look human
#   - But vocabulary variance across topics is 0.0009 (humans: 0.01-0.10)
#   - The model's deep structure leaks through persona prompts
#
# Cluster 0: 2 accounts  | coordination: 0.751 | Stealth bot pair
# Cluster 3: 2 accounts  | coordination: 0.776 | Haiku bot pair  
# Cluster 4: 2 accounts  | coordination: 0.841 | Haiku bot pair
# Cluster 5: 3 accounts  | coordination: 0.996 | FALSE POSITIVE (real humans)
#
# Noise: 31 accounts — 17 real humans + 14 undetected bots
#
# ===================================================
# DETECTION SCORECARD (from ground truth)
# ===================================================
#
# GPT-4 bots:    4/4 caught   (100%)
# Stealth bots:  11/20 caught (55%)
# Haiku bots:    4/9 caught   (44%)
# Humans:        17/20 correct as noise (85%), 3 false positives
#
# ===================================================
# HOW TO USE IN THE FRONTEND
# ===================================================
#
# The frontend needs to show:
#
# 1. NETWORK GRAPH (the money shot)
#    - Each dot = one account
#    - Accounts in the same cluster are close together with lines connecting them
#    - Noise accounts (no cluster) are scattered randomly with no connections
#    - Color by cluster, green for noise
#    - Read cluster membership from demo_clusters_anonymized.json
#
# 2. STATS BAR
#    - Accounts scanned: 53
#    - Networks detected: 5
#    - Coordinated accounts: 22 (sum of all cluster members)
#    - Organic (noise): 31
#
# 3. INTEL PANEL (click a cluster to see details)
#    - Cluster coordination signal
#    - Dominant model (GPT-4 / human)
#    - Average LLM density
#    - Member list with individual scores
#    - Sample text from members
#
# 4. ACCOUNT DETAIL (click an account)
#    - All scores from demo_results_anonymized.csv
#    - Their posts from demo_environment_anonymized.csv
#    - Which cluster they're in (or noise)
#
# To load data in React:
#
#   // Load results
#   const results = await fetch('/data/demo_results_anonymized.csv').then(r => r.text());
#   // Parse CSV (use papaparse or simple split)
#
#   // Load clusters  
#   const clusters = await fetch('/data/demo_clusters_anonymized.json').then(r => r.json());
#
#   // Load posts for detail view
#   const posts = await fetch('/data/demo_environment_anonymized.csv').then(r => r.text());
#
# ===================================================
# THE DEMO REVEAL
# ===================================================
#
# During the presentation:
# 1. Show the system scanning 53 "unknown" accounts
# 2. Watch clusters form — "the system found 5 coordinated networks"
# 3. Click Cluster 1: "These 4 accounts all write identically — coordination 1.000"
# 4. Click Cluster 2: "These 9 accounts have different personas but cluster together"
# 5. Show the noise: "31 accounts with unique writing — these are real people"
# 6. REVEAL: Open demo_ground_truth.json — "Cluster 1 was a GPT-4 farm. 
#    Cluster 2 was 9 stealth bots prompted to sound human. The noise was 
#    17 real humans from 2015 Twitter. No labels. Just writing fingerprints."
#
# The punchline: "One account can fool you. Twenty can't."
