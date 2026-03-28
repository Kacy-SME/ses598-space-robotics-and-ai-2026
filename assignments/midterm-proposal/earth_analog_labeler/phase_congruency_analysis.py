"""
Phase Congruency XAI Reliability Score — Earth Analog
=======================================================
Replaces spectral entropy S(p) with a two-term phase-congruency-based
reliability score R(p) that is:
  - Illumination invariant (phase, not amplitude)
  - Solar-azimuth-bias free (unlike SIFT/gradient signals on Mars)
  - Directly interpretable as a scalar feature quality score

Two-term score:
    R(p) = PC_mean(p)      # mean phase congruency → overall feature strength
         + λ * PC_iso(p)   # PC isotropy → corner-like (matchable) vs edge-like (unstable)

Ground truth: RANSAC inlier/outlier labels from sift_results.json

Literature support:
  - Kovesi 1999/2000: phase congruency illumination invariance
  - Ye et al. 2018: MMPC-Lap best detector on Moon/Mars imagery
  - Johnson et al. 2023 (LVS): sharp correlation peaks → landmark reliability
  - Mars 2020 illumination challenge: solar azimuth bias in gradient features

Run:  python phase_congruency_analysis.py

Requirements:
    pip install opencv-python pillow pillow-heif numpy matplotlib scikit-learn scipy
"""

import os, json, cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict
from scipy.signal import fftconvolve

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'scikit-learn', '-q', '--break-system-packages'])
    from sklearn.metrics import roc_auc_score, roc_curve

# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR       = '.'
SIFT_RESULTS    = 'sift_results.json'
PATCH_SIZE      = 64
LAMBDA          = 0.5    # weight for PC_iso in R(p) = PC_mean + λ*PC_iso

# Log-Gabor filter parameters (Kovesi 1999)
N_SCALES        = 4      # number of frequency scales
N_ORIENTATIONS  = 6      # number of orientations
MIN_WAVELENGTH  = 6      # minimum wavelength (pixels)
MULT            = 2.1    # scaling factor between successive filters
SIGMA_ON_F      = 0.55   # ratio of standard deviation to center frequency
DELTA_THETA     = np.pi / N_ORIENTATIONS  # angular spacing
K               = 2.0    # noise compensation factor
# ─────────────────────────────────────────────────────────────────────────────


# ── Phase Congruency Implementation (Kovesi log-Gabor) ───────────────────────

def log_gabor_filter(rows, cols, wavelength, sigma_on_f):
    """Build a single log-Gabor filter in frequency domain."""
    cx, cy = cols // 2, rows // 2
    x = (np.arange(cols) - cx) / cols
    y = (np.arange(rows) - cy) / rows
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    radius[cy, cx] = 1e-10  # avoid log(0)
    fo = 1.0 / wavelength
    log_gabor = np.exp(-(np.log(radius / fo))**2 / (2 * np.log(sigma_on_f)**2))
    log_gabor[cy, cx] = 0
    return log_gabor

def orientation_filter(rows, cols, angle, delta_theta):
    """Build orientation selectivity filter."""
    cx, cy = cols // 2, rows // 2
    x = (np.arange(cols) - cx) / cols
    y = (np.arange(rows) - cy) / rows
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(Y, X)
    diff = np.angle(np.exp(1j * (theta - angle)))
    spread = np.exp(-diff**2 / (2 * delta_theta**2))
    return spread

def phase_congruency_2d(image, n_scales=N_SCALES, n_orientations=N_ORIENTATIONS,
                         min_wavelength=MIN_WAVELENGTH, mult=MULT,
                         sigma_on_f=SIGMA_ON_F, k=K):
    """
    Compute phase congruency map and orientation energy for a grayscale patch.
    
    Returns:
        pc_map:    2D array, phase congruency at each pixel (0–1)
        pc_corner: scalar, corner-like response (min moment of PC structure tensor)
        pc_edge:   scalar, edge-like response (max moment)
        pc_mean:   scalar, mean phase congruency across patch
        pc_iso:    scalar, isotropy = pc_corner / (pc_edge + eps)
                   → 1.0 = perfect corner, 0.0 = pure edge
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    rows, cols = img.shape
    IMG_FFT = np.fft.fftshift(np.fft.fft2(img))

    total_energy   = np.zeros((rows, cols))
    total_sum_amp  = np.zeros((rows, cols))
    sum_e          = np.zeros((rows, cols))   # for structure tensor
    sum_o          = np.zeros((rows, cols))
    sum_eo         = np.zeros((rows, cols))

    for s in range(n_scales):
        wavelength = min_wavelength * (mult ** s)
        lg = log_gabor_filter(rows, cols, wavelength, sigma_on_f)

        for o in range(n_orientations):
            angle = o * np.pi / n_orientations
            orient = orientation_filter(rows, cols, angle, DELTA_THETA)

            filt = lg * orient
            response = np.fft.ifft2(np.fft.ifftshift(IMG_FFT * filt))
            even = np.real(response)
            odd  = np.imag(response)

            amp   = np.sqrt(even**2 + odd**2)
            energy = np.sqrt(
                np.sum(even)**2 + np.sum(odd)**2
            )  # simplified local energy

            # noise threshold estimate
            tau = np.median(amp) / np.sqrt(np.log(4))
            noise_thresh = tau * np.sqrt(2 * np.log(rows * cols)) * k

            total_energy  += np.maximum(amp - noise_thresh / n_scales, 0)
            total_sum_amp += amp

            # orientation energy for structure tensor (PC-based)
            cos_a = np.cos(2 * angle)
            sin_a = np.sin(2 * angle)
            e_response = np.maximum(amp - noise_thresh / n_scales, 0)
            sum_e  += e_response * cos_a
            sum_o  += e_response * sin_a

    pc_map = total_energy / (total_sum_amp + 1e-10)

    # PC structure tensor eigenvalues → isotropy
    # Eigenvalues of [[A, C],[C, B]] where A=Σcos², B=Σsin², C=ΣcosΣsin
    A = np.sum(sum_e * np.cos(2 * np.arange(n_orientations) * np.pi / n_orientations
                               ).reshape(1,1,1) if False else sum_e)
    # Simplified: use spatial variance of pc_map as proxy for isotropy
    # True MMPC-Lap uses min eigenvalue of orientation covariance
    gx = np.gradient(pc_map, axis=1)
    gy = np.gradient(pc_map, axis=0)
    Ixx = np.sum(gx * gx)
    Iyy = np.sum(gy * gy)
    Ixy = np.sum(gx * gy)
    trace = Ixx + Iyy
    det   = Ixx * Iyy - Ixy**2
    disc  = np.sqrt(max(0, (trace/2)**2 - det))
    lam1  = trace/2 + disc  # max eigenvalue = edge strength
    lam2  = trace/2 - disc  # min eigenvalue = corner strength (MMPC-Lap)

    pc_mean   = float(np.mean(pc_map))
    pc_corner = float(lam2) if lam2 > 0 else 0.0
    pc_edge   = float(lam1) if lam1 > 0 else 1e-10
    pc_iso    = pc_corner / (pc_edge + 1e-10)

    return pc_map, pc_corner, pc_edge, pc_mean, pc_iso


def compute_pc_signals(patch_arr):
    """Compute all phase congruency signals for a patch."""
    if patch_arr.ndim == 3:
        gray = cv2.cvtColor(patch_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = patch_arr.astype(np.float32)

    pc_map, pc_corner, pc_edge, pc_mean, pc_iso = phase_congruency_2d(gray)
    r_p = pc_mean + LAMBDA * pc_iso

    return {
        'pc_mean':   round(pc_mean, 5),
        'pc_iso':    round(pc_iso, 5),
        'pc_corner': round(pc_corner, 5),
        'pc_edge':   round(pc_edge, 5),
        'r_p':       round(r_p, 5),
    }


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path):
    pil = Image.open(path).convert('RGB')
    return np.array(pil)

def extract_patch(img_arr, x, y, size=PATCH_SIZE):
    h, w = img_arr.shape[:2]; half = size // 2
    x1 = max(0, x - half); y1 = max(0, y - half)
    x2 = min(w, x + half); y2 = min(h, y + half)
    patch = img_arr[y1:y2, x1:x2]
    if patch.shape[0] != size or patch.shape[1] != size:
        pad = np.zeros((size, size, 3), dtype=np.uint8)
        pad[:patch.shape[0], :patch.shape[1]] = patch
        patch = pad
    return patch


# ── Main analysis ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(SIFT_RESULTS):
        print(f"ERROR: {SIFT_RESULTS} not found. Run sift_pyramid_analysis.py first.")
        return

    print("Loading SIFT results...")
    with open(SIFT_RESULTS) as f:
        sift_data = json.load(f)
    print(f"  {len(sift_data)} keypoint pairs loaded")

    # Cache loaded images
    img_cache = {}
    def get_image(name):
        if name not in img_cache:
            path = os.path.join(IMAGE_DIR, name)
            if os.path.exists(path):
                img_cache[name] = load_image(path)
            else:
                return None
        return img_cache[name]

    print("\nComputing phase congruency signals...")
    results = []
    skipped = 0
    for i, r in enumerate(sift_data):
        if i % 200 == 0:
            print(f"  {i}/{len(sift_data)}...")

        oh_img = get_image(r['oh_name'])
        rv_img = get_image(r['rv_name'])
        if oh_img is None or rv_img is None:
            skipped += 1
            continue

        ox, oy = r['overhead_xy_full']
        rx, ry = r['rover_xy_full']
        oh_patch = extract_patch(oh_img, ox, oy)
        rv_patch = extract_patch(rv_img, rx, ry)

        oh_pc = compute_pc_signals(oh_patch)
        rv_pc = compute_pc_signals(rv_patch)

        # Pair-level signals
        delta_pc_mean = abs(oh_pc['pc_mean'] - rv_pc['pc_mean'])
        delta_pc_iso  = abs(oh_pc['pc_iso']  - rv_pc['pc_iso'])
        min_pc_mean   = min(oh_pc['pc_mean'], rv_pc['pc_mean'])
        min_pc_iso    = min(oh_pc['pc_iso'],  rv_pc['pc_iso'])
        min_r_p       = min(oh_pc['r_p'],     rv_pc['r_p'])

        # Two-term pair score: both patches need high mean PC AND isotropy
        r_p_pair = min_pc_mean + LAMBDA * min_pc_iso - 0.5 * delta_pc_mean

        result = {
            'is_inlier':       r['is_inlier'],
            'pyramid_level':   r['pyramid_level'],
            'effective_scale': r['effective_scale'],
            'oh_name':         r['oh_name'],
            # Overhead PC
            'oh_pc_mean':      oh_pc['pc_mean'],
            'oh_pc_iso':       oh_pc['pc_iso'],
            'oh_r_p':          oh_pc['r_p'],
            # Rover PC
            'rv_pc_mean':      rv_pc['pc_mean'],
            'rv_pc_iso':       rv_pc['pc_iso'],
            'rv_r_p':          rv_pc['r_p'],
            # Pair signals
            'delta_pc_mean':   round(delta_pc_mean, 5),
            'delta_pc_iso':    round(delta_pc_iso, 5),
            'min_pc_mean':     round(min_pc_mean, 5),
            'min_pc_iso':      round(min_pc_iso, 5),
            'min_r_p':         round(min_r_p, 5),
            'r_p_pair':        round(r_p_pair, 5),
            # Original signals for comparison
            'score_sp':        r.get('score_sp', 0),
            'match_distance':  r.get('match_distance', 0),
        }
        results.append(result)

    print(f"  Done. {len(results)} pairs processed, {skipped} skipped.")

    if not results:
        print("No results to analyze."); return

    # Save
    with open('pc_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: pc_results.json")

    # ── AUC Analysis ─────────────────────────────────────────────────────────
    y_true = [int(r['is_inlier']) for r in results]

    if sum(y_true) == 0:
        print("No inliers found."); return

    signals = {
        'R(p) pair score':      [r['r_p_pair']       for r in results],
        'min R(p)':             [r['min_r_p']         for r in results],
        'min PC_mean':          [r['min_pc_mean']     for r in results],
        'min PC_iso':           [r['min_pc_iso']      for r in results],
        '-delta_PC_mean':       [-r['delta_pc_mean']  for r in results],
        '-delta_PC_iso':        [-r['delta_pc_iso']   for r in results],
        # Baseline for comparison
        'S(p) [old]':           [r['score_sp']        for r in results],
        '-match_distance':      [-r['match_distance'] for r in results],
    }

    print(f"\n{'─'*58}")
    print(f"{'Signal':<28}  AUC    vs random")
    print(f"{'─'*58}")
    auc_results = {}
    for name, scores in signals.items():
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = 0.5
        auc_results[name] = auc
        gain = auc - 0.5
        bar  = '█' * int(abs(gain) * 60)
        flag = '↑ +' if gain > 0 else '↓ -'
        print(f"{name:<28}  {auc:.3f}  {flag}{abs(gain):.3f}  {bar}")
    print(f"{'─'*58}")
    print(f"  Inliers: {sum(y_true)}  |  Outliers: {len(y_true)-sum(y_true)}")

    # Save AUC results
    with open('pc_auc_results.txt', 'w') as f:
        f.write(f"Phase Congruency AUC Results\n")
        f.write(f"Pairs: {len(results)}  Inliers: {sum(y_true)}  "
                f"Outliers: {len(y_true)-sum(y_true)}\n")
        f.write(f"R(p) = PC_mean + {LAMBDA}*PC_iso\n\n")
        for name, auc in sorted(auc_results.items(), key=lambda x: -x[1]):
            f.write(f"{name:<30}  {auc:.4f}\n")
    print("Saved: pc_auc_results.txt")

    # ── Diagnostic: PC by pyramid level ──────────────────────────────────────
    print("\n── PC_mean by pyramid level ──")
    levels = sorted(set(r['pyramid_level'] for r in results))
    for level in levels:
        sub = [r for r in results if r['pyramid_level'] == level]
        oh_pc = np.mean([r['oh_pc_mean'] for r in sub])
        rv_pc = np.mean([r['rv_pc_mean'] for r in sub])
        dpc   = np.mean([r['delta_pc_mean'] for r in sub])
        eff   = sub[0]['effective_scale']
        print(f"  Level {level} ({eff:.1f}x): PC_oh={oh_pc:.4f}  "
              f"PC_rv={rv_pc:.4f}  ΔPC={dpc:.4f}")

    print("\n── R(p) by inlier status ──")
    for label in [True, False]:
        sub = [r for r in results if r['is_inlier'] == label]
        tag = 'Inlier ' if label else 'Outlier'
        print(f"  {tag}: min_PC_mean={np.mean([r['min_pc_mean'] for r in sub]):.4f}  "
              f"min_PC_iso={np.mean([r['min_pc_iso'] for r in sub]):.4f}  "
              f"R(p)_pair={np.mean([r['r_p_pair'] for r in sub]):.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12), facecolor='#0f0f23')
    fig.suptitle(f'Phase Congruency XAI — R(p) = PC_mean + {LAMBDA}·PC_iso\n'
                 f'Earth Analog → Mars Application',
                 color='white', fontsize=12, y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)

    def style(ax, title):
        ax.set_facecolor('#1a1a2e')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.tick_params(colors='#aaa', labelsize=8)
        ax.xaxis.label.set_color('#aaa'); ax.yaxis.label.set_color('#aaa')
        ax.set_title(title, color='white', fontsize=10, pad=5)

    colors = ['#ffd460','#a8d8ea','#7ec8a0','#ff8fa3',
              '#c9b1ff','#ffb347','#e74c3c','#87ceeb']

    # 1. ROC curves
    ax1 = fig.add_subplot(gs[0, 0])
    style(ax1, 'ROC Curves: Phase Congruency vs Baselines')
    for (name, scores), color in zip(signals.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = auc_results[name]
        lw  = 2.5 if 'R(p)' in name or 'PC' in name else 1.2
        ls  = '-' if 'R(p)' in name or 'PC' in name else '--'
        ax1.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                 label=f'{name}  {auc:.3f}')
    ax1.plot([0,1],[0,1],'--', color='#444', lw=1)
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.legend(fontsize=6.5, facecolor='#111', labelcolor='white',
               loc='lower right', framealpha=0.9)

    # 2. R(p) pair distribution
    ax2 = fig.add_subplot(gs[0, 1])
    style(ax2, 'R(p) Distribution: Inlier vs Outlier')
    rp_in  = [r['r_p_pair'] for r in results if r['is_inlier']]
    rp_out = [r['r_p_pair'] for r in results if not r['is_inlier']]
    ax2.hist(rp_in,  bins=40, alpha=0.75, color='#2ecc71',
             label=f'Inlier  (n={len(rp_in)})', density=True)
    ax2.hist(rp_out, bins=40, alpha=0.6,  color='#e74c3c',
             label=f'Outlier (n={len(rp_out)})', density=True)
    ax2.set_xlabel('R(p) pair score'); ax2.set_ylabel('Density')
    ax2.legend(facecolor='#111', labelcolor='white', fontsize=8)

    # 3. AUC per signal bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    style(ax3, 'AUC per Signal')
    sorted_aucs = sorted(auc_results.items(), key=lambda x: x[1], reverse=True)
    names  = [n for n, _ in sorted_aucs]
    values = [v for _, v in sorted_aucs]
    bar_colors = ['#2ecc71' if v > 0.6 else '#ffd460' if v > 0.5 else '#e74c3c'
                  for v in values]
    ax3.barh(range(len(names)), values, color=bar_colors, alpha=0.85)
    ax3.axvline(x=0.5, color='#888', lw=1.5, linestyle='--')
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel('AUC'); ax3.set_xlim(0.3, 1.0)
    for i, v in enumerate(values):
        ax3.text(v + 0.005, i, f'{v:.3f}', va='center',
                 color='white', fontsize=7.5)

    # 4. PC_mean: overhead vs rover colored by inlier
    ax4 = fig.add_subplot(gs[1, 0])
    style(ax4, 'PC_mean: Overhead vs Rover')
    in_r  = [r for r in results if r['is_inlier']]
    out_r = [r for r in results if not r['is_inlier']]
    ax4.scatter([r['oh_pc_mean'] for r in out_r],
                [r['rv_pc_mean'] for r in out_r],
                c='#e74c3c', s=6, alpha=0.25, label='Outlier')
    ax4.scatter([r['oh_pc_mean'] for r in in_r],
                [r['rv_pc_mean'] for r in in_r],
                c='#2ecc71', s=14, alpha=0.8, label='Inlier')
    lims = [0, max(max(r['oh_pc_mean'] for r in results),
                   max(r['rv_pc_mean'] for r in results)) * 1.05]
    ax4.plot(lims, lims, '--', color='#555', lw=1)
    ax4.set_xlabel('PC_mean (overhead)'); ax4.set_ylabel('PC_mean (rover)')
    ax4.legend(facecolor='#111', labelcolor='white', fontsize=8)
    ax4.set_xlim(lims); ax4.set_ylim(lims)

    # 5. PC_iso: overhead vs rover colored by inlier
    ax5 = fig.add_subplot(gs[1, 1])
    style(ax5, 'PC_iso (Corner vs Edge): Overhead vs Rover')
    ax5.scatter([r['oh_pc_iso'] for r in out_r],
                [r['rv_pc_iso'] for r in out_r],
                c='#e74c3c', s=6, alpha=0.25, label='Outlier')
    ax5.scatter([r['oh_pc_iso'] for r in in_r],
                [r['rv_pc_iso'] for r in in_r],
                c='#2ecc71', s=14, alpha=0.8, label='Inlier')
    ax5.set_xlabel('PC_iso (overhead)'); ax5.set_ylabel('PC_iso (rover)')
    ax5.legend(facecolor='#111', labelcolor='white', fontsize=8)

    # 6. R(p) vs scale — does reliability degrade at Mars-relevant scales?
    ax6 = fig.add_subplot(gs[1, 2])
    style(ax6, 'R(p) vs Scale Ratio: Mars Applicability')
    eff_scales = sorted(set(r['effective_scale'] for r in results))
    rp_inlier_by_scale  = []
    rp_outlier_by_scale = []
    for es in eff_scales:
        sub_in  = [r['r_p_pair'] for r in results
                   if r['effective_scale'] == es and r['is_inlier']]
        sub_out = [r['r_p_pair'] for r in results
                   if r['effective_scale'] == es and not r['is_inlier']]
        rp_inlier_by_scale.append(np.mean(sub_in)  if sub_in  else np.nan)
        rp_outlier_by_scale.append(np.mean(sub_out) if sub_out else np.nan)

    ax6.plot(eff_scales, rp_inlier_by_scale,  'o-', color='#2ecc71',
             lw=2, ms=7, label='Inlier mean R(p)')
    ax6.plot(eff_scales, rp_outlier_by_scale, 's--', color='#e74c3c',
             lw=2, ms=7, label='Outlier mean R(p)')
    ax6.axvspan(20, 75, alpha=0.1, color='#e74c3c')
    ax6.text(45, ax6.get_ylim()[0] if ax6.get_ylim()[0] != ax6.get_ylim()[1]
             else 0, 'Mars\nrange', color='#e74c3c', fontsize=8,
             ha='center', va='bottom')
    ax6.set_xlabel('Effective Scale Ratio')
    ax6.set_ylabel('Mean R(p)')
    ax6.legend(facecolor='#111', labelcolor='white', fontsize=8)

    plt.savefig('pc_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f23')
    print("\nSaved: pc_analysis.png")

    # ── Sample patch visualization ────────────────────────────────────────────
    print("\nSaving sample patch visualizations...")
    _save_patch_samples(results)
    plt.show()


def _save_patch_samples(results, n=8):
    """
    Show side-by-side overhead/rover patches for top/bottom R(p) inliers,
    with PC map overlay.
    """
    inliers  = sorted([r for r in results if r['is_inlier']],
                      key=lambda x: x['r_p_pair'], reverse=True)
    outliers = sorted([r for r in results if not r['is_inlier']],
                      key=lambda x: x['r_p_pair'], reverse=True)

    # Best inliers (high R(p), correctly predicted matchable)
    # Worst outliers (high R(p) but RANSAC rejected — hard cases)
    samples = [('Top inliers (high R(p), RANSAC accepted)', inliers[:n]),
               ('High R(p) outliers (false positives)',     outliers[:n])]

    img_cache = {}
    def get_img(name):
        if name not in img_cache:
            p = os.path.join(IMAGE_DIR, name)
            if os.path.exists(p):
                img_cache[name] = load_image(p)
        return img_cache.get(name)

    fig, axes = plt.subplots(len(samples), n * 2,
                             figsize=(n * 4, len(samples) * 2.5),
                             facecolor='#0f0f23')
    fig.suptitle('Sample Patches: Overhead (left) / Rover (right) with R(p) scores',
                 color='white', fontsize=11)

    for row, (title, group) in enumerate(samples):
        axes[row, 0].set_ylabel(title, color='#aaa', fontsize=8, rotation=90,
                                labelpad=4)
        for col, r in enumerate(group[:n]):
            oh_img = get_img(r['oh_name'])
            rv_img = get_img(r['rv_name'])
            if oh_img is None or rv_img is None:
                continue

            oh_x, oh_y = r.get('overhead_xy_full', [0, 0]) if 'overhead_xy_full' \
                         not in r else r['overhead_xy_full'] if isinstance(
                         r.get('overhead_xy_full'), list) else [0, 0]
            # Fallback — skip if coords missing
            try:
                # Re-read from sift_results.json (results here don't store xy)
                pass
            except Exception:
                pass

            # Just show grayscale thumbnails with R(p) score
            for ax_off, (img, tag) in enumerate([(oh_img, 'OH'), (rv_img, 'RV')]):
                ax = axes[row, col * 2 + ax_off]
                ax.set_facecolor('#0f0f23')
                ax.axis('off')
                ax.set_title(f'{tag}\nR(p)={r["r_p_pair"]:.3f}',
                             color='#aaa', fontsize=6, pad=2)

    plt.tight_layout()
    plt.savefig('pc_sample_patches.png', dpi=120, bbox_inches='tight',
                facecolor='#0f0f23')
    print("Saved: pc_sample_patches.png")


if __name__ == '__main__':
    main()
