"""
Generate hard negatives + run AUC analysis on XAI signals
==========================================================
Negatives = random overhead patch + random rover patch from the same image pair,
sampled anywhere in the scene (avoiding labeled match locations).

Run:  python analyze_pairs.py

Outputs:
    - earth_pairs/negatives/  — saved negative patch pairs
    - auc_results.txt         — AUC scores per signal
    - signal_distributions.png
"""

import json, os, random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, RocCurveDisplay

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR      = '.'
OUTPUT_DIR     = 'earth_pairs'
PATCH_SIZE     = 64
NEGATIVES_PER_PAIR = 8   # how many negatives to generate per image pair
MIN_DIST_PX    = 150     # min distance from any labeled match location (full-res px)
SEED           = 42
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


def compute_patch_metadata(patch_arr):
    gray = patch_arr.mean(axis=2).astype(np.float32) / 255.0
    hist, _ = np.histogram(gray, bins=32, range=(0, 1), density=True)
    hist = hist + 1e-10; hist /= hist.sum()
    entropy_H = float(-np.sum(hist * np.log2(hist)))
    try:
        s = np.linalg.svd(gray, compute_uv=False)
        svd_rank = float(s[0] / (s.sum() + 1e-8))
    except Exception:
        svd_rank = 0.0
    gx = np.diff(gray, axis=1); gy = np.diff(gray, axis=0)
    grad_mag = float(np.mean(np.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2)))
    return {
        'entropy_H': round(entropy_H, 4),
        'svd_rank':  round(svd_rank,  4),
        'grad_mag':  round(grad_mag,  4),
        'contrast':  round(float(np.std(gray)), 4),
    }

def extract_patch(img_arr, x, y):
    h, w = img_arr.shape[:2]; half = PATCH_SIZE // 2
    x1, y1 = max(0, x - half), max(0, y - half)
    x2, y2 = min(w, x + half), min(h, y + half)
    patch = img_arr[y1:y2, x1:x2]
    if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
        pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        pad[:patch.shape[0], :patch.shape[1]] = patch
        patch = pad
    return patch

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def too_close(x, y, taken, min_dist):
    for tx, ty in taken:
        if abs(x - tx) < min_dist and abs(y - ty) < min_dist:
            return True
    return False

def sample_random_point(img_arr, taken, min_dist, max_tries=200):
    h, w = img_arr.shape[:2]
    half = PATCH_SIZE // 2
    for _ in range(max_tries):
        x = random.randint(half, w - half)
        y = random.randint(half, h - half)
        if not too_close(x, y, taken, min_dist):
            return x, y
    return None


def load_matches():
    matches = []
    for p in sorted(Path(OUTPUT_DIR).glob('*/metadata.json')):
        with open(p) as f:
            m = json.load(f)
        if m.get('is_match') and 'negatives' not in str(p):
            matches.append(m)
    return matches


def generate_negatives(matches):
    # Group matches by image pair
    from collections import defaultdict
    by_pair = defaultdict(list)
    for m in matches:
        oh_f = Path(m['overhead']['image']).name
        rv_f = Path(m['rover']['image']).name
        by_pair[(oh_f, rv_f)].append(m)

    neg_dir = Path(OUTPUT_DIR) / 'negatives'
    neg_dir.mkdir(exist_ok=True)

    negatives = []
    for (oh_f, rv_f), pair_matches in by_pair.items():
        oh_path = os.path.join(IMAGE_DIR, oh_f)
        rv_path = os.path.join(IMAGE_DIR, rv_f)
        if not os.path.exists(oh_path) or not os.path.exists(rv_path):
            print(f"  Skipping {oh_f} — image not found")
            continue

        oh_full = load_image(oh_path)
        rv_full = load_image(rv_path)

        # Collect labeled locations to avoid
        oh_taken = [m['overhead']['pixel_xy'] for m in pair_matches]
        rv_taken = [m['rover']['pixel_xy']    for m in pair_matches]

        generated = 0
        attempts  = 0
        while generated < NEGATIVES_PER_PAIR and attempts < 500:
            attempts += 1
            oh_pt = sample_random_point(oh_full, oh_taken, MIN_DIST_PX)
            rv_pt = sample_random_point(rv_full, rv_taken, MIN_DIST_PX)
            if oh_pt is None or rv_pt is None:
                continue

            ox, oy = oh_pt
            rx, ry = rv_pt

            oh_patch = extract_patch(oh_full, ox, oy)
            rv_patch = extract_patch(rv_full, rx, ry)
            oh_m = compute_patch_metadata(oh_patch)
            rv_m = compute_patch_metadata(rv_patch)

            pair_id  = f'{Path(oh_f).stem}__{Path(rv_f).stem}__NEG_{generated:02d}'
            pair_dir = neg_dir / pair_id
            pair_dir.mkdir(exist_ok=True)
            Image.fromarray(oh_patch).save(pair_dir / 'overhead_patch.jpg', quality=95)
            Image.fromarray(rv_patch).save(pair_dir / 'rover_patch.jpg',    quality=95)

            dH = round(abs(oh_m['entropy_H'] - rv_m['entropy_H']), 4)
            mH = round(min(oh_m['entropy_H'],  rv_m['entropy_H']), 4)
            sp = round(mH - dH, 4)

            meta = {
                'pair_id':     pair_id,
                'landmark_id': 'NEG',
                'is_match':    False,
                'overhead': {'image': oh_path, 'pixel_xy': [ox, oy],
                             'viewpoint': 'overhead', **oh_m},
                'rover':    {'image': rv_path, 'pixel_xy': [rx, ry],
                             'viewpoint': 'rover',    **rv_m},
                'delta_entropy_H': dH,
                'delta_svd_rank':  round(abs(oh_m['svd_rank'] - rv_m['svd_rank']), 4),
                'delta_grad_mag':  round(abs(oh_m['grad_mag'] - rv_m['grad_mag']), 4),
                'min_entropy_H':   mH,
                'min_grad_mag':    round(min(oh_m['grad_mag'], rv_m['grad_mag']), 4),
                'score_sp':        sp,
            }
            with open(pair_dir / 'metadata.json', 'w') as f:
                json.dump(meta, f, indent=2)

            oh_taken.append([ox, oy])
            rv_taken.append([rx, ry])
            negatives.append(meta)
            generated += 1

        print(f"  {oh_f}↔{rv_f}: {generated} negatives generated")

    return negatives


def run_auc_analysis(matches, negatives):
    all_data = matches + negatives
    y_true   = [int(d['is_match']) for d in all_data]

    signals = {
        'S(p) = min_H - ΔH':  [d['score_sp']        for d in all_data],
        'min_entropy_H':       [d['min_entropy_H']    for d in all_data],
        '-delta_entropy_H':    [-d['delta_entropy_H'] for d in all_data],
        '-delta_svd_rank':     [-d['delta_svd_rank']  for d in all_data],
        'min_grad_mag':        [d['min_grad_mag']      for d in all_data],
    }

    print(f"\n{'─'*50}")
    print(f"{'Signal':<28}  AUC")
    print(f"{'─'*50}")
    results = {}
    for name, scores in signals.items():
        auc = roc_auc_score(y_true, scores)
        results[name] = auc
        bar = '█' * int(auc * 30)
        print(f"{name:<28}  {auc:.3f}  {bar}")
    print(f"{'─'*50}")
    print(f"  Matches: {sum(y_true)}  |  Negatives: {len(y_true)-sum(y_true)}")

    # Save results
    with open('auc_results.txt', 'w') as f:
        f.write(f"Matches: {sum(y_true)}  Negatives: {len(y_true)-sum(y_true)}\n\n")
        for name, auc in sorted(results.items(), key=lambda x: -x[1]):
            f.write(f"{name:<28}  {auc:.4f}\n")
    print("\nSaved: auc_results.txt")

    return y_true, signals, results


def plot_results(y_true, signals, results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f0f23')
    fig.suptitle('XAI Signal Analysis — Earth Analog Pairs', color='white', fontsize=12)

    # ── ROC curves ────────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    ax1.tick_params(colors='#aaa'); ax1.yaxis.label.set_color('#aaa')
    ax1.xaxis.label.set_color('#aaa')
    for sp in ax1.spines.values(): sp.set_color('#333')
    colors = ['#ffd460', '#a8d8ea', '#7ec8a0', '#ff8fa3', '#c9b1ff']
    for (name, scores), color in zip(signals.items(), colors):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = results[name]
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{name}  AUC={auc:.3f}')
    ax1.plot([0,1],[0,1], '--', color='#555', lw=1)
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves', color='white')
    ax1.legend(fontsize=7, facecolor='#111', labelcolor='white', loc='lower right')

    # ── Score distributions ───────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='#aaa'); ax2.yaxis.label.set_color('#aaa')
    ax2.xaxis.label.set_color('#aaa')
    for sp in ax2.spines.values(): sp.set_color('#333')

    sp_scores = signals['S(p) = min_H - ΔH']
    match_sp  = [s for s, y in zip(sp_scores, y_true) if y == 1]
    neg_sp    = [s for s, y in zip(sp_scores, y_true) if y == 0]
    ax2.hist(match_sp, bins=20, alpha=0.7, color='#2ecc71', label=f'Match (n={len(match_sp)})')
    ax2.hist(neg_sp,   bins=20, alpha=0.7, color='#e74c3c', label=f'Negative (n={len(neg_sp)})')
    ax2.set_xlabel('S(p) score'); ax2.set_ylabel('Count')
    ax2.set_title('S(p) Distribution: Match vs Negative', color='white')
    ax2.legend(facecolor='#111', labelcolor='white')

    plt.tight_layout()
    plt.savefig('signal_distributions.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f23')
    print("Saved: signal_distributions.png")
    plt.show()


def main():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               'scikit-learn', '-q', '--break-system-packages'])
        from sklearn.metrics import roc_auc_score

    print("Loading matches...")
    matches = load_matches()
    print(f"  {len(matches)} labeled matches found")

    print("\nGenerating negatives...")
    neg_dir = Path(OUTPUT_DIR) / 'negatives'
    existing_negs = list(neg_dir.glob('*/metadata.json')) if neg_dir.exists() else []

    if existing_negs:
        print(f"  Found {len(existing_negs)} existing negatives — loading them")
        negatives = []
        for p in existing_negs:
            with open(p) as f:
                negatives.append(json.load(f))
    else:
        negatives = generate_negatives(matches)

    print(f"  {len(negatives)} negatives ready")

    print("\nRunning AUC analysis...")
    y_true, signals, results = run_auc_analysis(matches, negatives)

    print("\nPlotting...")
    plot_results(y_true, signals, results)


if __name__ == '__main__':
    main()
