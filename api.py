"""
Ground Truth API — runs the actual analysis pipeline on uploaded CSVs.
Streams progress updates via Server-Sent Events so the frontend can show a live console.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import sys
import os
import tempfile
import time
import threading
import queue

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'core'))

app = Flask(__name__)
CORS(app)

# Store results in memory for download
latest_results = {}


class ProgressCapture:
    """Captures print output from a specific thread and sends it to a queue for SSE streaming."""
    def __init__(self, q, thread_id):
        self.q = q
        self.thread_id = thread_id
        self.original = sys.stdout

    def write(self, text):
        # Only capture output from the pipeline thread
        if threading.current_thread().ident == self.thread_id:
            if text.strip():
                self.q.put(text.strip())
        self.original.write(text)

    def flush(self):
        self.original.flush()


def assess_cluster_threat(info, member_data):
    """
    Once we have a cluster, use the COMBINED evidence from all members
    to assess whether this is likely a bot operation. Individual accounts
    might pass as human. But the cluster-level statistics often reveal
    what individual analysis can't.
    """
    coord = info.get('coordination_signal', 0)
    avg_llm = info.get('avg_llm_density', 0)
    member_count = info['member_count']

    # Collect cluster-wide stats from per-account results
    vocab_vars = [m.get('intra_vocab_var', 0) for m in member_data]
    struct_regs = [m.get('structural_regularity', 0) for m in member_data]
    typo_rates = [m.get('typo_rate', 0) for m in member_data]

    avg_vocab_var = sum(vocab_vars) / len(vocab_vars) if vocab_vars else 0
    avg_struct_reg = sum(struct_regs) / len(struct_regs) if struct_regs else 0
    zero_typo_count = sum(1 for t in typo_rates if t == 0)
    zero_typo_pct = zero_typo_count / len(typo_rates) if typo_rates else 0

    # Build evidence list
    evidence = []
    bot_score = 0

    # LLM markers (strong signal)
    if avg_llm > 0.5:
        evidence.append(f'High LLM phrase density across cluster ({avg_llm:.2f})')
        bot_score += 3
    elif avg_llm > 0.05:
        evidence.append(f'Low-level LLM markers detected ({avg_llm:.2f}) \u2014 model partially evading detection')
        bot_score += 1

    # Zero typos across the cluster (strong signal at scale)
    if zero_typo_pct > 0.8 and member_count >= 3:
        evidence.append(
            f'{zero_typo_count}/{member_count} accounts have zero informal markers '
            f'\u2014 extremely unlikely for {member_count} independent humans on social media')
        bot_score += 2
    elif zero_typo_pct > 0.5:
        evidence.append(f'{zero_typo_count}/{member_count} accounts have zero informal markers')
        bot_score += 1

    # Vocabulary variance (the topology argument)
    if avg_vocab_var < 0.005 and member_count >= 3:
        evidence.append(
            f'Average vocabulary variance {avg_vocab_var:.4f} across {member_count} accounts '
            f'\u2014 writing is unnaturally uniform across different topics')
        bot_score += 2

    # Structural regularity
    if avg_struct_reg > 0.75:
        evidence.append(f'Average paragraph regularity {avg_struct_reg:.2f} \u2014 sentence patterns are machine-like')
        bot_score += 1

    # Coordination strength
    if coord > 0.9:
        evidence.append(f'Coordination signal {coord:.3f} \u2014 near-identical writing fingerprints')
        bot_score += 2
    elif coord > 0.7:
        evidence.append(f'Coordination signal {coord:.3f} \u2014 highly similar writing fingerprints')
        bot_score += 1

    # Scale bonus
    if member_count >= 8:
        evidence.append(
            f'{member_count} accounts clustering together '
            f'\u2014 organic coordination at this scale is extremely rare')
        bot_score += 1

    # Determine assessment
    if bot_score >= 5:
        bot_likelihood = 'very_high'
        assessment = (f'Almost certainly a bot operation. '
                      f'{len(evidence)} independent signals point to automated generation.')
    elif bot_score >= 3:
        bot_likelihood = 'high'
        assessment = ('Likely a bot operation. Cluster-level evidence is stronger '
                      'than any individual account score.')
    elif bot_score >= 2:
        bot_likelihood = 'moderate'
        assessment = ('Possibly bot-operated, possibly human-coordinated (e.g., troll farm '
                      'with shared talking points). Either way, these accounts are not independent.')
    else:
        bot_likelihood = 'low'
        assessment = ('Coordinated but bot indicators are weak. Could be organic community '
                      'with similar writing patterns, or sophisticated operation. Warrants investigation.')

    return {
        'bot_likelihood': bot_likelihood,
        'bot_score': bot_score,
        'evidence': evidence,
        'assessment': assessment,
        'cluster_stats': {
            'avg_vocab_variance': round(avg_vocab_var, 6),
            'avg_structural_regularity': round(avg_struct_reg, 4),
            'zero_typo_pct': round(zero_typo_pct, 2),
            'avg_llm_density': round(avg_llm, 4),
        }
    }


def analyze_cluster_narratives(clusters, posts_by_account):
    """
    For each cluster, analyze what narrative(s) the coordinated accounts are pushing.
    Uses keyword matching — simple, fast, no extra API calls.
    """
    NARRATIVE_KEYWORDS = {
        'product_promotion': [
            'buy', 'purchase', 'product', 'brand', 'recommend', 'amazing',
            'best', 'love it', 'worth', 'incredible', 'game changer',
            'highly recommend', 'must have', 'five stars', 'changed my life',
            'discount', 'deal', 'sale', 'promo', 'check out', 'link in bio'
        ],
        'geopolitical_disinfo': [
            'russia', 'ukraine', 'nato', 'crimea', 'war', 'military',
            'invasion', 'sanctions', 'peace', 'conflict', 'western',
            'propaganda', 'media lies', 'deep state', 'regime'
        ],
        'ai_safety_dismissal': [
            'ai safety', 'regulation', 'overblown', 'fear-mongering',
            'innovation', 'progress', 'bureaucrat', 'ai risk',
            'unnecessary', 'slow down', 'holding back'
        ],
        'political_influence': [
            'vote', 'election', 'candidate', 'democrat', 'republican',
            'trump', 'biden', 'policy', 'border', 'immigration',
            'corruption', 'rigged', 'stolen', 'rally', 'movement'
        ],
        'health_misinfo': [
            'vaccine', 'pharma', 'natural cure', 'big pharma',
            'side effects', 'toxin', 'immune', 'detox', 'supplement',
            "they don't want you to know"
        ],
        'crypto_pump': [
            'crypto', 'bitcoin', 'token', 'moon', 'hodl', 'gains',
            'invest', 'portfolio', 'bull run', 'altcoin', 'defi', 'nft'
        ],
        'general_lifestyle': [
            'coffee', 'morning', 'workout', 'gym', 'restaurant',
            'commute', 'recipe', 'weekend', 'family', 'kids',
            'app', 'phone', 'travel', 'vacation'
        ],
    }

    DANGEROUS_NARRATIVES = {'geopolitical_disinfo', 'political_influence', 'health_misinfo'}

    results = {}

    for cid, info in clusters.items():
        members = info.get('members', [])
        all_posts = []
        for member_id in members:
            posts = posts_by_account.get(member_id, [])
            all_posts.extend(posts)

        if not all_posts:
            results[cid] = {
                'dominant_narrative': 'unknown',
                'narratives': {},
                'narrative_count': 0,
                'sample_posts_by_narrative': {},
                'assessment': 'Insufficient data for narrative analysis'
            }
            continue

        full_text = ' '.join(all_posts).lower()

        scores = {}
        post_matches = {}

        for narrative, keywords in NARRATIVE_KEYWORDS.items():
            hits = sum(full_text.count(kw) for kw in keywords)
            scores[narrative] = hits

            matched = []
            for post in all_posts:
                post_lower = post.lower()
                post_hits = sum(post_lower.count(kw) for kw in keywords)
                matching_kws = [kw for kw in keywords if kw in post_lower]
                if post_hits > 0:
                    matched.append((post_hits, post[:200], matching_kws))
            matched.sort(reverse=True)
            post_matches[narrative] = [
                {'text': p, 'keywords': kws} for _, p, kws in matched[:3]
            ]

        total = sum(scores.values()) or 1
        distribution = {k: round(v / total, 3) for k, v in scores.items() if v > 0}

        sorted_narratives = sorted(distribution.items(), key=lambda x: -x[1])
        dominant = sorted_narratives[0][0] if sorted_narratives else 'unknown'

        significant = [(k, v) for k, v in sorted_narratives if v >= 0.1]
        has_dangerous = any(k in DANGEROUS_NARRATIVES for k, _ in significant)

        if len(significant) == 1:
            assessment = f'Single-narrative campaign: {dominant.replace("_", " ")} ({distribution[dominant]*100:.0f}% of content)'
        elif len(significant) == 2 and 'general_lifestyle' in dict(significant):
            other = [k for k, v in significant if k != 'general_lifestyle'][0]
            assessment = f'Mixed campaign using lifestyle content as cover for {other.replace("_", " ")} narrative'
        elif len(significant) >= 2:
            tops = ', '.join(f'{k.replace("_", " ")} ({v*100:.0f}%)' for k, v in significant[:3])
            assessment = f'Multi-narrative campaign pushing: {tops}'
        else:
            assessment = 'Low narrative signal \u2014 may be general engagement farming'

        results[cid] = {
            'dominant_narrative': dominant,
            'narratives': distribution,
            'narrative_count': len(significant),
            'is_multi_narrative': len(significant) > 1,
            'uses_lifestyle_cover': 'general_lifestyle' in dict(significant) and len(significant) > 1,
            'has_dangerous_narrative': has_dangerous,
            'sample_posts_by_narrative': {k: post_matches[k] for k in dict(significant)},
            'assessment': assessment,
        }

    return results


def _json_safe(obj):
    """Make numpy types JSON-serializable."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, 'item'):
        return obj.item()
    return obj


@app.route('/', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "Ground Truth API"})


@app.route('/api/analyze_stream', methods=['POST'])
def analyze_stream():
    """
    Upload a CSV and stream progress via Server-Sent Events.
    The frontend shows these as a live console output.
    """
    if 'csv' not in request.files:
        return jsonify({"error": "No CSV file uploaded"}), 400

    f = request.files['csv']
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, 'upload.csv')
    f.save(csv_path)

    result_holder = {}
    q = queue.Queue()

    def run_pipeline():
        import pandas as pd
        from enhanced_pipeline import run_enhanced_pipeline

        old_stdout = sys.stdout
        old_cwd = os.getcwd()
        sys.stdout = ProgressCapture(q, threading.current_thread().ident)

        try:
            # Run pipeline in tmp_dir so file exports don't pollute project root
            os.chdir(tmp_dir)
            results_df, clusters, noise, graph_positions = run_enhanced_pipeline([csv_path], epsilon=1.5)

            # Load posts for sample text
            posts_df = pd.read_csv(csv_path)
            id_col = posts_df.columns[0]
            text_col = posts_df.columns[1]
            posts_by_account = {}
            for aid, group in posts_df.groupby(id_col):
                posts_by_account[str(aid)] = group[text_col].tolist()[:5]

            # Build per-account lookup for cluster-level assessment
            results_by_account = {}
            for _, row in results_df.iterrows():
                r = {}
                for col in results_df.columns:
                    val = row[col]
                    if hasattr(val, 'item'):
                        val = val.item()
                    r[col] = val
                results_by_account[r['account_id']] = r

            # Attach sample posts + threat classification to clusters
            for cid, info in clusters.items():
                info['sample_posts'] = {}
                for member in info.get('members', [])[:5]:
                    info['sample_posts'][member] = posts_by_account.get(member, [])[:2]

                if info.get('is_bot_network'):
                    # Gather member-level data for cluster assessment
                    member_data = [results_by_account[m] for m in info['members'] if m in results_by_account]
                    threat = assess_cluster_threat(info, member_data)
                    info['bot_likelihood'] = threat['bot_likelihood']
                    info['bot_score'] = threat['bot_score']
                    info['evidence'] = threat['evidence']
                    info['assessment'] = threat['assessment']
                    info['cluster_stats'] = threat['cluster_stats']

                    # Primary label: always "Coordinated network" — this is what we proved
                    info['threat_type'] = 'coordinated'
                    info['threat_label'] = 'Coordinated network'

                    # Model inference: best guess based on evidence
                    avg_llm = info.get('avg_llm_density', 0)
                    avg_typo = info.get('avg_typo_rate', 0)
                    coord = info.get('coordination_signal', 0)
                    dominant = info.get('dominant_model', 'human')
                    model_dist = info.get('model_distribution', {})

                    if avg_llm > 0.5 and model_dist.get('gpt4', 0) > 0:
                        info['model_inference'] = 'Detected: GPT-4'
                        info['model_inference_type'] = 'detected'
                        info['model_inference_reason'] = (
                            f'High LLM phrase density ({avg_llm:.2f}) with GPT-4 specific markers '
                            f'(\"certainly,\" \"furthermore,\" \"moreover\"). '
                            f'Model identification is high confidence.')
                    elif avg_llm > 0.5 and model_dist.get('claude', 0) > 0:
                        info['model_inference'] = 'Detected: Claude'
                        info['model_inference_type'] = 'detected'
                        info['model_inference_reason'] = (
                            f'High LLM phrase density ({avg_llm:.2f}) with Claude-specific markers. '
                            f'Model identification is high confidence.')
                    elif avg_llm > 0.5:
                        info['model_inference'] = 'Detected: LLM (unknown model)'
                        info['model_inference_type'] = 'detected'
                        info['model_inference_reason'] = (
                            f'High LLM phrase density ({avg_llm:.2f}) but no model-specific markers. '
                            f'Content is likely AI-generated but the specific model could not be identified.')
                    elif avg_llm > 0.05:
                        info['model_inference'] = 'Suspected: LLM (unknown model)'
                        info['model_inference_type'] = 'suspected'
                        info['model_inference_reason'] = (
                            f'Low-level LLM markers detected ({avg_llm:.2f}) \u2014 '
                            f'possibly a model that partially evades phrase detection, '
                            f'or content that has been lightly edited after generation.')
                    elif coord > 0.6 and avg_typo == 0:
                        info['model_inference'] = 'Suspected: LLM or scripted'
                        info['model_inference_type'] = 'suspected'
                        stats = threat['cluster_stats']
                        info['model_inference_reason'] = (
                            f'Zero informal markers across all accounts, '
                            f'structural regularity {stats["avg_structural_regularity"]:.2f} '
                            f'\u2014 consistent with LLM generation or human operators using shared templates. '
                            f'Could not identify specific model.')
                    else:
                        info['model_inference'] = 'Unknown origin'
                        info['model_inference_type'] = 'unknown'
                        info['model_inference_reason'] = (
                            f'Accounts show coordination (signal: {coord:.3f}) but include informal markers '
                            f'like typos ({avg_typo:.4f}). Could be a sophisticated LLM operation mimicking '
                            f'casual writing, or human operators working from shared talking points. '
                            f'The coordination is real; the origin is unclear.')
                else:
                    info['threat_type'] = 'noise'
                    info['threat_label'] = 'No coordination'
                    info['bot_likelihood'] = 'none'
                    info['bot_score'] = 0
                    info['evidence'] = []
                    info['assessment'] = ''
                    info['cluster_stats'] = {}
                    info['model_inference'] = None
                    info['model_inference_type'] = None
                    info['model_inference_reason'] = None

            # Narrative analysis per cluster
            narratives = analyze_cluster_narratives(clusters, posts_by_account)
            for cid, narr in narratives.items():
                if cid in clusters:
                    clusters[cid]['narrative'] = narr

            # Convert results to JSON-safe list
            results_list = []
            for _, row in results_df.iterrows():
                r = {}
                for col in results_df.columns:
                    val = row[col]
                    if hasattr(val, 'item'):
                        val = val.item()
                    r[col] = val
                results_list.append(r)

            # Store for download endpoints
            latest_results['results_csv'] = results_df.to_csv(index=False)
            latest_results['clusters_json'] = json.dumps(_json_safe(clusters), indent=2)

            result_holder['data'] = _json_safe({
                "status": "complete",
                "summary": {
                    "accounts_analyzed": len(results_df),
                    "clusters_found": len(clusters),
                    "coordinated_accounts": sum(info['member_count'] for info in clusters.values()),
                    "noise_accounts": noise['count'],
                },
                "results": results_list,
                "clusters": clusters,
                "noise": noise,
                "posts": posts_by_account,
                "positions": graph_positions,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_holder['data'] = {"status": "error", "error": str(e)}
        finally:
            sys.stdout = old_stdout
            os.chdir(old_cwd)
            q.put("__DONE__")
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Padding to defeat proxy chunk-buffering (some proxies wait for >=2KB)
    padding = ":" + " " * 2048 + "\n\n"

    def stream():
        # Send initial padding to flush proxy buffers
        yield padding
        while True:
            try:
                msg = q.get(timeout=120)
                if msg == "__DONE__":
                    data = result_holder.get('data', {"status": "error", "error": "Unknown"})
                    yield f"data: {json.dumps({'type': 'result', 'data': data}, default=str)}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"
            except queue.Empty:
                # Heartbeat keeps connection alive
                yield ": heartbeat\n\n"

    return Response(stream(), mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache, no-transform',
                        'X-Accel-Buffering': 'no',
                        'Connection': 'keep-alive',
                        'Content-Type': 'text/event-stream; charset=utf-8',
                    })


@app.route('/api/download/results', methods=['GET'])
def download_results():
    csv_data = latest_results.get('results_csv', '')
    if not csv_data:
        return jsonify({"error": "No results available. Run analysis first."}), 404
    return Response(csv_data, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=styleshield_results.csv"})


@app.route('/api/download/clusters', methods=['GET'])
def download_clusters():
    json_data = latest_results.get('clusters_json', '')
    if not json_data:
        return jsonify({"error": "No results available. Run analysis first."}), 404
    return Response(json_data, mimetype='application/json',
                    headers={"Content-Disposition": "attachment;filename=styleshield_clusters.json"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"Ground Truth API starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
