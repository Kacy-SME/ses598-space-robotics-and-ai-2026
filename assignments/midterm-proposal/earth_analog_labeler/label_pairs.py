"""
Earth Analog Correspondence Labeler
=====================================
Single window: overhead (left) | rover (right) | reference strip (bottom).
The reference strip shows the saved overhead patch for each landmark ID
so you can see "this is what D1 looks like" across all pairs.

Controls:
    Click LEFT panel     → mark overhead point
    Click RIGHT panel    → mark rover point
    1–9                  → select landmark ID
    M                    → Save Match
    N                    → Save Negative
    C                    → Clear pending clicks
    →  /  ←              → Next / Prev image pair
    Q                    → Quit

Requirements:
    pip install pillow pillow-heif matplotlib numpy
"""

import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURE THESE
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DIR  = '.'
OUTPUT_DIR = 'earth_pairs'

IMAGE_PAIRS = [
    ('IMG_7964.HEIC', 'IMG_7973.HEIC'),
    ('IMG_7965.HEIC', 'IMG_7974.HEIC'),
    ('IMG_7966.HEIC', 'IMG_7975.HEIC'),
    ('IMG_7967.HEIC', 'IMG_7976.HEIC'),
    ('IMG_7968.HEIC', 'IMG_7977.HEIC'),
    ('IMG_7969.HEIC', 'IMG_7978.HEIC'),
    ('IMG_7970.HEIC', 'IMG_7979.HEIC'),
    ('IMG_7971.HEIC', 'IMG_7980.HEIC'),
]

LANDMARK_IDS = [
    'D1',   # 1
    'D2',   # 2
    'R1',   # 3
    'R2',   # 4
    'G1',   # 5
    'G2',   # 6
    'S1',   # 7
    'S2',   # 8
    'X1',   # 9
]

OVERHEAD_HEIGHT_M = 2.0
ROVER_HEIGHT_M    = 0.5
ROVER_ANGLE_DEG   = 30.0
DISPLAY_SCALE     = 0.38
PATCH_SIZE        = 64
# ─────────────────────────────────────────────────────────────────────────────


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

def count_pairs_total():
    return len(list(Path(OUTPUT_DIR).glob('*/metadata.json')))

def load_existing_dots(oh_file, rv_file):
    dots = []
    for p in sorted(Path(OUTPUT_DIR).glob('*/metadata.json')):
        with open(p) as f:
            m = json.load(f)
        if (Path(m['overhead']['image']).name == oh_file and
                Path(m['rover']['image']).name == rv_file):
            dots.append(m)
    return dots

def get_last_positions(lid):
    """Return (oh_x, oh_y, rv_x, rv_y) from the most recently saved entry for this landmark ID,
    searching across all pairs. Returns None if never labeled."""
    best = None
    for p in sorted(Path(OUTPUT_DIR).glob('*/metadata.json')):
        with open(p) as f:
            m = json.load(f)
        if m.get('landmark_id') == lid:
            best = (m['overhead']['pixel_xy'][0], m['overhead']['pixel_xy'][1],
                    m['rover']['pixel_xy'][0],    m['rover']['pixel_xy'][1])
    return best

def get_reference_patches():
    """For each landmark ID, find the first saved overhead patch across all pairs."""
    refs = {}
    for p in sorted(Path(OUTPUT_DIR).glob('*/metadata.json')):
        with open(p) as f:
            m = json.load(f)
        lid = m.get('landmark_id')
        if lid and lid not in refs:
            patch_path = Path(p).parent / 'overhead_patch.jpg'
            if patch_path.exists():
                refs[lid] = np.array(Image.open(patch_path))
    return refs

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def make_thumb(arr):
    h, w = arr.shape[:2]
    return np.array(Image.fromarray(arr).resize(
        (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)), Image.LANCZOS))


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    valid_pairs = []
    for oh_f, rv_f in IMAGE_PAIRS:
        oh_p = os.path.join(IMAGE_DIR, oh_f)
        rv_p = os.path.join(IMAGE_DIR, rv_f)
        if os.path.exists(oh_p) and os.path.exists(rv_p):
            valid_pairs.append((oh_f, rv_f, oh_p, rv_p))
        else:
            missing = [x for x in [oh_p, rv_p] if not os.path.exists(x)]
            print(f"  Skipping ({oh_f}, {rv_f}) — missing: {missing}")

    if not valid_pairs:
        sys.exit("No valid image pairs found.")

    scale_ratio = round(
        OVERHEAD_HEIGHT_M / (ROVER_HEIGHT_M * np.cos(np.radians(ROVER_ANGLE_DEG))), 2)

    print(f"\n{len(valid_pairs)} image pairs ready.")
    print(f"Controls: click → 1–9 set ID → M=match  N=negative  C=clear  ←→=pair  Q=quit\n")

    st = {
        'pair_idx':    0,
        'oh':          None,
        'rv':          None,
        'oh_markers':  [],
        'rv_markers':  [],
        'landmark_id': LANDMARK_IDS[0],
        'total':       count_pairs_total(),
        'oh_full': None, 'rv_full': None,
        'oh_sx': 1, 'oh_sy': 1,
        'rv_sx': 1, 'rv_sy': 1,
    }

    # ── Layout: 2 rows. Top row = images (80%), bottom row = reference strip (20%) ──
    n_refs = len(LANDMARK_IDS)
    fig = plt.figure(figsize=(20, 11), facecolor='#0f0f23')
    fig.canvas.manager.set_window_title('TRN Labeler')

    gs = GridSpec(2, n_refs, figure=fig,
                  height_ratios=[4.5, 1],
                  hspace=0.08, wspace=0.04,
                  left=0.02, right=0.98, top=0.95, bottom=0.02)

    # Main image panels span all columns each
    ax_oh = fig.add_subplot(gs[0, :n_refs//2])
    ax_rv = fig.add_subplot(gs[0, n_refs//2:])

    # Reference strip — one axis per landmark ID
    ref_axes = [fig.add_subplot(gs[1, i]) for i in range(n_refs)]

    for ax in [ax_oh, ax_rv]:
        ax.set_facecolor('#1a1a2e')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.tick_params(colors='#555', labelsize=6)

    for ax in ref_axes:
        ax.set_facecolor('#111')
        ax.axis('off')

    # Status bar at top
    status_txt = fig.text(0.5, 0.975, '', ha='center', color='white', fontsize=9,
                           bbox=dict(boxstyle='round', fc='#2d2d44', alpha=0.9))

    def refresh_reference_strip():
        refs = get_reference_patches()
        for i, (ax_r, lid) in enumerate(zip(ref_axes, LANDMARK_IDS)):
            ax_r.cla()
            ax_r.axis('off')
            is_active = (lid == st['landmark_id'])
            border_color = '#ffe066' if is_active else '#333'
            for sp in ax_r.spines.values():
                sp.set_visible(True)
                sp.set_color(border_color)
                sp.set_linewidth(2.5 if is_active else 1)
            if lid in refs:
                ax_r.imshow(refs[lid])
                ax_r.set_title(f'[{i+1}] {lid}',
                               color='#ffe066' if is_active else '#aaa',
                               fontsize=8, fontweight='bold' if is_active else 'normal',
                               pad=2)
            else:
                ax_r.set_facecolor('#0a0a1a')
                ax_r.text(0.5, 0.5, f'[{i+1}]\n{lid}\n—',
                          ha='center', va='center', color='#555',
                          fontsize=8, transform=ax_r.transAxes)
        fig.canvas.draw_idle()

    def update_status(msg):
        oh_f, rv_f = valid_pairs[st['pair_idx']][:2]
        n_here = len(load_existing_dots(oh_f, rv_f))
        lid = st['landmark_id']
        status_txt.set_text(
            f'Pair {st["pair_idx"]+1}/{len(valid_pairs)}  ·  '
            f'{oh_f} ↔ {rv_f}  ·  '
            f'{n_here} labeled this pair, {st["total"]} total  ·  '
            f'ID: {lid}  ·  {msg}')
        refresh_reference_strip()

    def clear_active_markers():
        for m in st['oh_markers'] + st['rv_markers']:
            try: m.remove()
            except: pass
        st['oh_markers'].clear()
        st['rv_markers'].clear()

    def show_ghost(lid):
        """Pre-fill positions from last saved label for this ID and show dim ghost markers."""
        clear_active_markers()
        pos = get_last_positions(lid)
        if pos is None:
            st['oh'] = st['rv'] = None
            return
        ox, oy, rx, ry = pos
        st['oh'] = (ox, oy)
        st['rv'] = (rx, ry)
        # Draw dim ghost markers (dashed, low alpha)
        ox_t, oy_t = ox / st['oh_sx'], oy / st['oh_sy']
        rx_t, ry_t = rx / st['rv_sx'], ry / st['rv_sy']
        st['oh_markers'] += [
            ax_oh.plot(ox_t, oy_t, '+', color='#a8d8ea', ms=22, mew=2, alpha=0.4)[0],
            ax_oh.add_patch(mpatches.Circle((ox_t, oy_t), 16,
                            fill=False, ec='#a8d8ea', lw=1.5, alpha=0.35, linestyle='--')),
            ax_oh.text(ox_t + 18, oy_t - 10, 'prev', color='#a8d8ea',
                       fontsize=7, alpha=0.5),
        ]
        st['rv_markers'] += [
            ax_rv.plot(rx_t, ry_t, '+', color='#ffd460', ms=22, mew=2, alpha=0.4)[0],
            ax_rv.add_patch(mpatches.Circle((rx_t, ry_t), 16,
                            fill=False, ec='#ffd460', lw=1.5, alpha=0.35, linestyle='--')),
            ax_rv.text(rx_t + 18, ry_t - 10, 'prev', color='#ffd460',
                       fontsize=7, alpha=0.5),
        ]
        fig.canvas.draw_idle()

    def render_pair(idx):
        oh_f, rv_f, oh_p, rv_p = valid_pairs[idx]
        oh_full  = load_image(oh_p)
        rv_full  = load_image(rv_p)
        oh_thumb = make_thumb(oh_full)
        rv_thumb = make_thumb(rv_full)

        st.update({
            'oh_full': oh_full, 'rv_full': rv_full,
            'oh_sx': oh_full.shape[1] / oh_thumb.shape[1],
            'oh_sy': oh_full.shape[0] / oh_thumb.shape[0],
            'rv_sx': rv_full.shape[1] / rv_thumb.shape[1],
            'rv_sy': rv_full.shape[0] / rv_thumb.shape[0],
        })
        st['oh'] = st['rv'] = None
        clear_active_markers()

        ax_oh.cla(); ax_rv.cla()
        for ax in [ax_oh, ax_rv]:
            ax.set_facecolor('#1a1a2e')
            for sp in ax.spines.values(): sp.set_color('#333')
            ax.tick_params(colors='#555', labelsize=6)

        ax_oh.imshow(oh_thumb)
        ax_oh.set_title(
            f'OVERHEAD  ·  {oh_f}  [{idx+1}/{len(valid_pairs)}]  '
            '← → to change pair',
            color='#a8d8ea', fontsize=9, pad=5)

        ax_rv.imshow(rv_thumb)
        ax_rv.set_title(
            f'ROVER  ·  {rv_f}  [{idx+1}/{len(valid_pairs)}]  '
            'M=match  N=negative  C=clear',
            color='#ffd460', fontsize=9, pad=5)

        # Redraw saved dots
        for m in load_existing_dots(oh_f, rv_f):
            c   = '#2ecc71' if m['is_match'] else '#e74c3c'
            lid = m.get('landmark_id', '?')
            ox, oy = m['overhead']['pixel_xy']
            rx, ry = m['rover']['pixel_xy']
            ax_oh.plot(ox / st['oh_sx'], oy / st['oh_sy'],
                       'o', color=c, ms=9, alpha=0.85, zorder=5)
            ax_oh.text(ox / st['oh_sx'] + 5, oy / st['oh_sy'] - 5,
                       lid, color=c, fontsize=8, fontweight='bold', zorder=5)
            ax_rv.plot(rx / st['rv_sx'], ry / st['rv_sy'],
                       'o', color=c, ms=9, alpha=0.85, zorder=5)
            ax_rv.text(rx / st['rv_sx'] + 5, ry / st['rv_sy'] - 5,
                       lid, color=c, fontsize=8, fontweight='bold', zorder=5)

        show_ghost(st['landmark_id'])
        update_status('Ghost = last saved position. M to accept, click to adjust'
                      if get_last_positions(st['landmark_id']) else 'Click overhead panel first')

    def do_save(is_match):
        if not (st['oh'] and st['rv']):
            update_status('⚠  Click BOTH panels first'); return

        oh_f, rv_f, oh_p, rv_p = valid_pairs[st['pair_idx']]
        ox, oy = st['oh']; rx, ry = st['rv']
        oh_patch = extract_patch(st['oh_full'], ox, oy)
        rv_patch = extract_patch(st['rv_full'], rx, ry)
        oh_m = compute_patch_metadata(oh_patch)
        rv_m = compute_patch_metadata(rv_patch)

        lid      = st['landmark_id']
        pair_id  = f'{Path(oh_f).stem}__{Path(rv_f).stem}__{lid}'
        pair_dir = Path(OUTPUT_DIR) / pair_id
        pair_dir.mkdir(exist_ok=True)
        Image.fromarray(oh_patch).save(pair_dir / 'overhead_patch.jpg', quality=95)
        Image.fromarray(rv_patch).save(pair_dir / 'rover_patch.jpg',    quality=95)

        dH = round(abs(oh_m['entropy_H'] - rv_m['entropy_H']), 4)
        mH = round(min(oh_m['entropy_H'],  rv_m['entropy_H']), 4)
        sp = round(mH - dH, 4)

        meta = {
            'pair_id':     pair_id,
            'landmark_id': lid,
            'is_match':    is_match,
            'overhead': {'image': oh_p, 'pixel_xy': [ox, oy], 'viewpoint': 'overhead', **oh_m},
            'rover':    {'image': rv_p, 'pixel_xy': [rx, ry], 'viewpoint': 'rover',    **rv_m},
            'scale_ratio':     scale_ratio,
            'delta_entropy_H': dH,
            'delta_svd_rank':  round(abs(oh_m['svd_rank'] - rv_m['svd_rank']), 4),
            'delta_grad_mag':  round(abs(oh_m['grad_mag'] - rv_m['grad_mag']), 4),
            'min_entropy_H':   mH,
            'min_grad_mag':    round(min(oh_m['grad_mag'], rv_m['grad_mag']), 4),
            'score_sp':        sp,
        }
        with open(pair_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        c = '#2ecc71' if is_match else '#e74c3c'
        ax_oh.plot(ox / st['oh_sx'], oy / st['oh_sy'], 'o', color=c, ms=9, alpha=0.9, zorder=5)
        ax_oh.text(ox / st['oh_sx'] + 5, oy / st['oh_sy'] - 5,
                   lid, color=c, fontsize=8, fontweight='bold', zorder=5)
        ax_rv.plot(rx / st['rv_sx'], ry / st['rv_sy'], 'o', color=c, ms=9, alpha=0.9, zorder=5)
        ax_rv.text(rx / st['rv_sx'] + 5, ry / st['rv_sy'] - 5,
                   lid, color=c, fontsize=8, fontweight='bold', zorder=5)

        st['total'] += 1
        st['oh'] = st['rv'] = None
        clear_active_markers()

        lbl = 'MATCH   ' if is_match else 'NEGATIVE'
        update_status(f'Saved {lid} ({lbl.strip()})  ΔH={dH:.3f}  S(p)={sp:.3f}  ·  Click to continue')
        print(f'[{pair_id}] {lbl}  OH=({ox},{oy}) RV=({rx},{ry})  '
              f'ΔH={dH:.3f}  min_H={mH:.3f}  S(p)={sp:.3f}')

    def on_click(event):
        if event.xdata is None or event.button != 1: return

        if event.inaxes == ax_oh:
            for m in st['oh_markers']:
                try: m.remove()
                except: pass
            st['oh_markers'].clear()
            x, y = int(event.xdata * st['oh_sx']), int(event.ydata * st['oh_sy'])
            st['oh'] = (x, y)
            st['oh_markers'] += [
                ax_oh.plot(event.xdata, event.ydata, '+',
                           color='#a8d8ea', ms=22, mew=2.5)[0],
                ax_oh.add_patch(mpatches.Circle((event.xdata, event.ydata), 16,
                                fill=False, ec='#a8d8ea', lw=2, alpha=0.9))
            ]
            update_status('✓ Overhead  ✓ Rover — press M or N' if st['rv']
                          else '✓ Overhead — now click Rover panel')

        elif event.inaxes == ax_rv:
            for m in st['rv_markers']:
                try: m.remove()
                except: pass
            st['rv_markers'].clear()
            x, y = int(event.xdata * st['rv_sx']), int(event.ydata * st['rv_sy'])
            st['rv'] = (x, y)
            st['rv_markers'] += [
                ax_rv.plot(event.xdata, event.ydata, '+',
                           color='#ffd460', ms=22, mew=2.5)[0],
                ax_rv.add_patch(mpatches.Circle((event.xdata, event.ydata), 16,
                                fill=False, ec='#ffd460', lw=2, alpha=0.9))
            ]
            update_status('✓ Overhead  ✓ Rover — press M or N' if st['oh']
                          else '✓ Rover — now click Overhead panel')

    def on_key(event):
        k = event.key
        if k == 'm':
            do_save(True)
        elif k == 'n':
            do_save(False)
        elif k == 'c':
            st['oh'] = st['rv'] = None
            clear_active_markers()
            update_status('Cleared — click overhead first')
        elif k in ('right', 'l'):
            st['pair_idx'] = (st['pair_idx'] + 1) % len(valid_pairs)
            render_pair(st['pair_idx'])
        elif k in ('left', 'h'):
            st['pair_idx'] = (st['pair_idx'] - 1) % len(valid_pairs)
            render_pair(st['pair_idx'])
        elif k == 'q':
            plt.close('all')
        elif k in [str(i) for i in range(1, len(LANDMARK_IDS) + 1)]:
            st['landmark_id'] = LANDMARK_IDS[int(k) - 1]
            show_ghost(st['landmark_id'])
            update_status(f'ID → {st["landmark_id"]}  ' +
                          ('Ghost shown — M to accept, click to adjust'
                           if get_last_positions(st['landmark_id'])
                           else 'No prior label — click to place'))
        elif k == 'u':
            oh_f, rv_f = valid_pairs[st['pair_idx']][:2]
            lid = st['landmark_id']
            pair_id = f'{Path(oh_f).stem}__{Path(rv_f).stem}__{lid}'
            pair_dir = Path(OUTPUT_DIR) / pair_id
            if pair_dir.exists():
                import shutil
                shutil.rmtree(pair_dir)
                st['total'] = max(0, st['total'] - 1)
                render_pair(st['pair_idx'])
                print(f'[DELETED] {pair_id}')
            else:
                update_status(f'Nothing to undo for {lid} on this pair')

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event',    on_key)

    render_pair(0)
    plt.show()

    print(f'\n── Done ──  {count_pairs_total()} total pairs in {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()