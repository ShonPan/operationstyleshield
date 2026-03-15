# StyleShield: Stealth Bot Detection Improvements
# Instructions for Claude Code implementation
# ==============================================
#
# CONTEXT: We have a working bot detection pipeline (Styleshield_script.py + 
# enhanced_extractor.py + enhanced_pipeline.py). It catches obvious bots 
# (GPT-4 and Haiku farms) perfectly but stealth bots that are prompted to 
# "sound human" evade detection by blending into the human cluster.
#
# GOAL: Add features that catch stealth bots by detecting signals that LLMs
# cannot fake even when prompted to be casual/human. Three categories:
#   1. Typo authenticity analysis (physical vs performed typos)
#   2. Infrastructure fingerprinting (metadata signals)  
#   3. Temporal cross-correlation (posting pattern similarity)
#
# FILES TO MODIFY:
#   - enhanced_extractor.py (add new features to EnhancedStylometricExtractor)
#   - enhanced_pipeline.py (add infrastructure features, update DISCRIMINATING_FEATURES)
#
# FILES TO READ FOR CONTEXT:
#   - Styleshield_script.py (base scorer by Caleb — don't modify, just import from)
#   - demo_account_metadata.csv (has location, timezone, coords, follower_ratio, 
#     default_profile, has_profile_pic, account_created per account)
#   - demo_environment.csv (the main text dataset — account_id, post_text, posting_hour)

# =====================================================
# PART 1: TYPO AUTHENTICITY ANALYSIS
# =====================================================
#
# KEY INSIGHT: LLMs can be prompted to include typos, but the typos don't 
# follow real physical keyboard mechanics. Human typos have specific causes:
#
# A. Adjacent-key errors: finger hits neighboring key on QWERTY layout
#    - "teh" for "the" (e and h are adjacent)  
#    - "adn" for "and" (d and n are close)
#    - "jsut" for "just" (j and u are adjacent)
#    These follow the QWERTY_NEIGHBORS map already in enhanced_extractor.py
#
# B. Transposition errors: fingers fire in wrong order
#    - "teh" for "the", "abt" for "bat"
#    - These are always adjacent character swaps, never distant ones
#    - Humans transpose at consistent positions (often positions 2-3 in a word)
#
# C. Dropped characters: finger didn't press hard enough, especially on mobile
#    - "goin" for "going", "thnk" for "think"
#    - More common at end of words (rushed typing)
#    - More common on mobile (smaller keys)
#
# D. Double-tap errors: key registers twice
#    - "thhe" for "the"
#    - Follow key-bounce physics — more common on certain keys
#
# E. Autocorrect artifacts: phone autocorrect creates WRONG words, not misspellings
#    - "duck" for the obvious one
#    - Creates real-but-wrong words, not gibberish
#
# LLM FAKE TYPOS are different:
#    - Too evenly distributed across words (humans have personal hotspot patterns)
#    - Don't follow keyboard adjacency — LLMs might write "tha" for "the" 
#      (a is not adjacent to e on QWERTY)
#    - Too varied — real humans make the SAME typos repeatedly
#    - Missing informal abbreviations: humans naturally write "u" "ur" "bc" "w/"
#      but LLMs prompted to be casual rarely produce these consistently
#    - Perfect punctuation despite "casual" tone — humans who typo also mess up
#      punctuation; LLMs prompted for typos still punctuate perfectly
#
# Add these features to EnhancedStylometricExtractor.extract_enhanced():

"""
# ---- TYPO AUTHENTICITY FEATURES ----

# 1. Zero typo flag (binary, very strong signal)
# If an account has 20+ words and zero informal markers or typos, 
# that's extremely unlikely for a real human on social media
features["zero_typo_flag"] = 1.0 if features["typo_rate"] == 0 and len(tokens) > 20 else 0.0

# 2. Typo-punctuation coherence
# Real humans who make typos ALSO have messy punctuation
# LLMs prompted for typos still capitalize perfectly and use proper commas
sentences = [s.strip() for s in re.split(r'[.!?]+', full_text) if len(s.strip()) > 3]
properly_capitalized = sum(1 for s in sentences if s and s[0].isupper()) / max(len(sentences), 1)
features["perfect_capitalization_rate"] = properly_capitalized

# Incoherence score: high typo rate + high capitalization = FAKE typos
# Real humans: low typo → high capitalization OR high typo → low capitalization
# Fake casual: high typo → still high capitalization (the tell)
features["typo_punctuation_incoherence"] = features.get("typo_rate", 0) * properly_capitalized

# 3. Adjacent-key typo ratio
# What fraction of suspected typos follow QWERTY adjacency?
# Real typos: high ratio (physics-based errors)
# Fake typos: low ratio (random character substitutions)
# Use QWERTY_NEIGHBORS already defined in enhanced_extractor.py
adjacent_errors = 0
total_unusual_bigrams = 0
for word in full_text.lower().split():
    w = word.strip(".,!?;:'\"()[]{}@#$%^&*")
    if len(w) < 3:
        continue
    for i in range(len(w) - 1):
        c1, c2 = w[i], w[i+1]
        if c1.isalpha() and c2.isalpha():
            # Check if this bigram is common in English
            # Unusual bigrams suggest typos
            if c1 in QWERTY_NEIGHBORS and c2 in QWERTY_NEIGHBORS:
                if c2 in QWERTY_NEIGHBORS.get(c1, set()):
                    adjacent_errors += 1
                total_unusual_bigrams += 1

if total_unusual_bigrams > 0:
    features["adjacent_key_typo_ratio"] = adjacent_errors / total_unusual_bigrams
else:
    features["adjacent_key_typo_ratio"] = 0

# 4. Typo consistency across posts (humans repeat the same typos)
# Real humans have a "typo fingerprint" — they consistently misspell 
# the same words the same way. LLM-generated typos are more random.
if len(per_post) >= 2:
    post_typo_patterns = []
    for post in posts:
        # Extract informal markers per post
        markers = set(re.findall(r'\b(u|ur|r|n|thru|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|lol|omg|tbh|idk|imo|smh|btw|ngl|bc|w/)\b', post.lower()))
        post_typo_patterns.append(markers)
    
    # Measure overlap between posts (high overlap = consistent human, low = random LLM)
    if len(post_typo_patterns) >= 2:
        overlaps = []
        for i in range(len(post_typo_patterns)):
            for j in range(i+1, len(post_typo_patterns)):
                a, b = post_typo_patterns[i], post_typo_patterns[j]
                if a or b:
                    overlap = len(a & b) / max(len(a | b), 1)
                    overlaps.append(overlap)
        features["typo_consistency"] = float(np.mean(overlaps)) if overlaps else 0
    else:
        features["typo_consistency"] = 0
else:
    features["typo_consistency"] = 0

# 5. Informal marker rate (expanded)
# Humans use these naturally and frequently on social media
# LLMs almost never produce them even when prompted to be casual
informal_patterns = r'\\b(lol|lmao|rofl|omg|tbh|idk|imo|smh|btw|ngl|fr|rn|af|nvm|irl|fwiw|gonna|wanna|gotta|kinda|sorta|ya|yep|nope|haha|hehe|lmfao|bruh|dude|yo|ooh|aww|hmm|meh|ugh|wtf|stfu|fml|tfw|mfw|ikr|istg|lowkey|highkey|deadass|srsly|pls|plz|thx|ty|np|jk|jfc|smol|boi|fam|goat|sus|vibe|vibes|slay|bet|cap|bussin|no cap)\\b'
informal_count = len(re.findall(informal_patterns, full_text.lower()))
features["informal_marker_rate"] = informal_count / max(len(tokens), 1)

# 6. Contraction naturalness
# Humans mix contractions inconsistently: "I don't" then "I do not" in same text
# LLMs tend to be consistent (all contractions or all formal)
contractions = len(re.findall(r"\\b\\w+'\\w+\\b", full_text))
formal_negations = len(re.findall(r'\\b(do not|does not|did not|will not|can not|cannot|would not|should not|could not|is not|are not|was not|were not|have not|has not|had not)\\b', full_text.lower()))
if contractions + formal_negations > 0:
    # 0.5 = mixed (human), 0 or 1 = consistent (bot)
    mix_ratio = contractions / (contractions + formal_negations)
    features["contraction_consistency"] = 1.0 - abs(mix_ratio - 0.5) * 2  # high = mixed = human
else:
    features["contraction_consistency"] = 0.5
"""

# =====================================================
# PART 2: INFRASTRUCTURE FINGERPRINTING
# =====================================================
#
# Add a new function to enhanced_pipeline.py that merges account metadata
# into the fingerprint vectors before clustering.
#
# Input: demo_account_metadata.csv with columns:
#   account_id, account_type, location, timezone, coords, 
#   follower_ratio, default_profile, has_profile_pic, account_created, post_count
#
# The metadata CSV is joined to fingerprints on account_id.

"""
# Add this function to enhanced_pipeline.py

def add_infrastructure_features(fingerprints, metadata_csv_path):
    '''
    Merge infrastructure metadata into fingerprint vectors.
    Accounts without metadata get neutral values (0.5).
    '''
    import pandas as pd
    from collections import Counter
    
    meta_df = pd.read_csv(metadata_csv_path)
    meta_map = meta_df.set_index('account_id').to_dict('index')
    
    # First pass: collect creation dates for batch detection
    creation_dates = {}
    for aid in fingerprints:
        meta = meta_map.get(aid, {})
        d = meta.get('account_created', '')
        if d:
            creation_dates.setdefault(str(d), []).append(aid)
    
    # Second pass: add features
    for aid in fingerprints:
        meta = meta_map.get(aid, {})
        
        # Location signals (bots don't set these)
        has_loc = 1.0 if str(meta.get('location', '')).strip() else 0.0
        has_tz = 1.0 if str(meta.get('timezone', '')).strip() else 0.0
        has_coords = 1.0 if str(meta.get('coords', '')).strip() else 0.0
        
        fingerprints[aid]["has_location"] = has_loc
        fingerprints[aid]["has_timezone"] = has_tz
        fingerprints[aid]["has_coords"] = has_coords
        
        # Profile completeness score (0 = bot-like, 1 = human-like)
        pic = float(meta.get('has_profile_pic', 1))
        default = float(meta.get('default_profile', 0))
        fingerprints[aid]["profile_completeness"] = (
            pic * 0.3 +
            (1.0 - default) * 0.3 +
            has_loc * 0.2 +
            has_tz * 0.2
        )
        
        # Follower ratio signal
        # Real humans: 0.5-2.0, Bot farms: < 0.2
        ratio = float(meta.get('follower_ratio', 0.5))
        fingerprints[aid]["follower_ratio_signal"] = min(ratio, 2.0) / 2.0
        
        # Account creation batch size
        # If 8 accounts were created on the same day, that's a bot farm signal
        d = str(meta.get('account_created', ''))
        batch_size = len(creation_dates.get(d, []))
        fingerprints[aid]["creation_batch_signal"] = min(batch_size / 5.0, 1.0)
        # Normalize: 1 account on that date = 0.2, 5+ = 1.0
    
    return fingerprints
"""

# Call this in enhanced_pipeline.py after extracting fingerprints:
#
#   fingerprints = add_infrastructure_features(fingerprints, 'demo_account_metadata.csv')
#
# Make the metadata path an optional argument to run_enhanced_pipeline()


# =====================================================
# PART 3: TEMPORAL CROSS-CORRELATION
# =====================================================
#
# Stealth bots from the same operator post at similar times even when
# they try to vary content. This is because:
#   - Same operator = same shift schedule
#   - Same data center = same timezone
#   - Batch processing = posts go out in waves
#
# Add this as a cross-account feature (computed after all individual
# fingerprints are extracted, similar to cross_account_correlation in
# Caleb's MultiMetricBotScorer).

"""
# Add to enhanced_pipeline.py or enhanced_extractor.py

def compute_temporal_cross_correlation(accounts):
    '''
    Compare posting time distributions across all accounts.
    Accounts with suspiciously similar posting schedules get flagged.
    
    Input: dict of {account_id: {"posting_hours": [9, 10, 11, ...]}}
    Output: dict of {account_id: {"temporal_cross_sim": float}}
    '''
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Build posting hour histogram per account (24 bins)
    aids = list(accounts.keys())
    histograms = {}
    
    for aid in aids:
        hours = accounts[aid].get("posting_hours", [])
        hist = np.zeros(24)
        for h in hours:
            hist[int(h) % 24] += 1
        total = hist.sum()
        if total > 0:
            hist = hist / total  # normalize to distribution
        histograms[aid] = hist
    
    # Compute pairwise similarity of posting time distributions
    mat = np.array([histograms[a] for a in aids])
    
    # Handle zero vectors
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_norm = mat / norms
    
    sim_matrix = cosine_similarity(mat_norm)
    
    features = {}
    for i, aid in enumerate(aids):
        # Max similarity to any other account (exclude self)
        sims = sim_matrix[i].copy()
        sims[i] = 0  # exclude self
        
        features[aid] = {
            "temporal_max_similarity": float(np.max(sims)),
            "temporal_mean_similarity": float(np.mean(sims)),
            # How many accounts have >0.9 temporal similarity? 
            # If 5 accounts match your schedule, that's an operator signal
            "temporal_cohort_size": int(np.sum(sims > 0.85)),
        }
    
    return features
"""

# Merge into fingerprints:
#   temporal_features = compute_temporal_cross_correlation(accounts)
#   for aid in fingerprints:
#       fingerprints[aid].update(temporal_features.get(aid, {}))


# =====================================================
# PART 4: UPDATE DISCRIMINATING_FEATURES
# =====================================================
#
# Replace the DISCRIMINATING_FEATURES list in enhanced_pipeline.py:

"""
DISCRIMINATING_FEATURES = [
    # === Text analysis (catches obvious bots) ===
    "llm_phrase_density",
    "gpt4_marker_rate",
    "claude_marker_rate",
    "model_prob_human",
    "avg_syllables",
    "jargon_density",
    "contraction_rate",
    "paragraph_rhythm_score",
    "sentence_length_variance",
    "opener_diversity",
    
    # === Intra-account consistency (bots = unnaturally uniform) ===
    "intra_vocab_variance",
    "intra_structure_variance",
    "intra_hedge_variance",
    "intra_punctuation_variance",
    "intra_jargon_variance",
    
    # === NEW: Typo authenticity (catches stealth bots) ===
    "zero_typo_flag",
    "informal_marker_rate",
    "perfect_capitalization_rate",
    "typo_punctuation_incoherence",
    "typo_consistency",
    "contraction_consistency",
    
    # === NEW: Infrastructure (catches data center operations) ===
    "has_location",
    "profile_completeness",
    "follower_ratio_signal",
    "creation_batch_signal",
    
    # === NEW: Temporal cross-correlation (catches same-operator accounts) ===
    "temporal_max_similarity",
    "temporal_cohort_size",
]
"""


# =====================================================
# PART 5: TESTING
# =====================================================
#
# After implementing all changes, run:
#
#   python3 enhanced_pipeline.py demo_environment.csv --metadata demo_account_metadata.csv
#
# Expected results:
#   - GPT-4 farm: caught (LLM markers + zero typos + no location)
#   - Haiku farm: caught (coordination + no location + batch creation)
#   - Stealth bots: NOW caught (zero typos + no informal markers + 
#     perfect capitalization + no location + temporal cohort matching)
#   - Real humans: noise (typos present, informal markers, locations set,
#     varied timezones, scattered posting times)
#
# If stealth bots still blend in, try:
#   1. Increase weight of zero_typo_flag and informal_marker_rate
#   2. Lower epsilon (try 1.0 or 0.8 with the new features)
#   3. Check that infrastructure features are actually being loaded
#      (print fingerprints for one stealth bot and one human to compare)


# =====================================================
# NOTES ON "CAN LLMS BE PROMPTED TO MAKE TYPOS?"
# =====================================================
#
# Yes, but the typos are detectably fake. Here's why:
#
# 1. DISTRIBUTION: Human typo hotspots are personal and consistent.
#    A person who types "teh" always types "teh" — it's muscle memory.
#    LLM typos are randomly placed across different words each time.
#    Measure: typo_consistency feature (overlap of typo patterns across posts)
#
# 2. KEYBOARD PHYSICS: Real typos follow QWERTY adjacency.
#    "r" and "t" are adjacent, so "thr" → "tht" is a real typo.
#    "r" and "p" are far apart, so "thr" → "thp" is suspicious.
#    Measure: adjacent_key_typo_ratio feature
#
# 3. CORRELATION WITH OTHER SIGNALS: Humans who typo also:
#    - Use inconsistent capitalization
#    - Miss punctuation
#    - Use informal abbreviations (lol, tbh, idk)
#    - Have shorter posts
#    LLMs prompted for typos still maintain perfect grammar elsewhere.
#    Measure: typo_punctuation_incoherence feature
#
# 4. MOBILE vs DESKTOP PATTERNS: Mobile typos cluster at specific
#    screen positions (thumbs hit wrong keys). Desktop typos are
#    transposition errors (fingers fire in wrong order). LLMs don't
#    know which device they're "using" so the pattern is incoherent.
#    Measure: mobile_device_signal (already in enhanced_extractor.py)
#
# 5. VOLUME: Real humans on social media produce informal markers at
#    ~5-10% rate. LLMs almost never produce "lol", "tbh", "ngl", 
#    "bruh", "deadass" etc. even when prompted to be casual.
#    Measure: informal_marker_rate feature
#
# The punchline for the demo:
# "You can prompt an LLM to misspell words. But you can't make it 
# misspell them the way a human thumb does on a phone at 11 PM — 
# in the same spots, every time, while also forgetting to capitalize
# and throwing in a 'lol' and a 'tbh'."
