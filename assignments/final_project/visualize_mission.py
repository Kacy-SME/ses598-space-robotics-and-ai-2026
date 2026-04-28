#!/usr/bin/env python3
"""
visualize_mission.py
--------------------
Run after a mission completes to generate plots from /tmp/mission_log.json

Usage:
    python3 visualize_mission.py
    python3 visualize_mission.py --log /tmp/mission_log.json --out ./plots
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Mars-inspired color palette ──────────────────────────────────────────────
MARS_RED    = '#c1440e'
MARS_DUST   = '#d4956a'
MARS_DARK   = '#1a0f0a'
MARS_MID    = '#2e1a0e'
MARS_LIGHT  = '#f0c090'
ACCENT_CYAN = '#4fc3c3'
ACCENT_GOLD = '#f5c518'

plt.rcParams.update({
    'figure.facecolor': MARS_DARK,
    'axes.facecolor':   MARS_MID,
    'axes.edgecolor':   MARS_DUST,
    'axes.labelcolor':  MARS_LIGHT,
    'xtick.color':      MARS_DUST,
    'ytick.color':      MARS_DUST,
    'text.color':       MARS_LIGHT,
    'grid.color':       '#3a2510',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})


def load_log(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"ERROR: log file not found at {path}")
        print("Make sure the mission ran with logging enabled.")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


# ── Plot 1: Boustrophedon path ────────────────────────────────────────────────
def plot_phase1_path(log: dict, out: pathlib.Path):
    wps = log.get('phase1_waypoints', [])
    if not wps:
        print("No Phase 1 waypoints to plot.")
        return

    xs = [w['x'] for w in wps]
    ys = [w['y'] for w in wps]
    n  = len(wps)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(MARS_DARK)

    # Draw path with gradient color (early→late)
    cmap = plt.cm.YlOrRd
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=cmap(0.3 + 0.7 * t), linewidth=2, alpha=0.85)

    # Waypoint markers
    sc = ax.scatter(xs, ys, c=range(n), cmap='YlOrRd',
                    s=80, zorder=5, edgecolors=MARS_DARK, linewidths=0.8)

    # Start / end markers
    ax.scatter([xs[0]],  [ys[0]],  s=180, marker='*',
               color=ACCENT_CYAN, zorder=10, label='Start')
    ax.scatter([xs[-1]], [ys[-1]], s=180, marker='X',
               color=ACCENT_GOLD, zorder=10, label='End')

    # Waypoint index labels
    for w in wps:
        ax.text(w['x'] + 0.4, w['y'] + 0.4, str(w['idx']),
                fontsize=6, color=MARS_DUST, alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Waypoint order', color=MARS_LIGHT)
    cbar.ax.yaxis.set_tick_params(color=MARS_DUST)

    ax.set_title(f'Phase 1 — Boustrophedon Survey ({n} waypoints)',
                 color=MARS_LIGHT, fontsize=13, pad=12)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.legend(facecolor=MARS_MID, edgecolor=MARS_DUST, labelcolor=MARS_LIGHT)
    ax.grid(True); ax.set_aspect('equal')

    fig.tight_layout()
    dest = out / 'phase1_boustrophedon.png'
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {dest}")


# ── Plot 2: UCB reward trajectory ─────────────────────────────────────────────
def plot_phase2_rewards(log: dict, out: pathlib.Path):
    visits = log.get('phase2_visits', [])
    if not visits:
        print("No Phase 2 visits to plot.")
        return

    idxs    = [v['idx']    for v in visits]
    rewards = [v['reward'] for v in visits]
    is_lm   = [v.get('is_landmark', False) for v in visits]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(MARS_DARK)

    ax.plot(idxs, rewards, color=MARS_DUST, linewidth=1.2,
            alpha=0.6, zorder=2, label='UCB reward')

    # Highlight landmarks
    lm_x = [idxs[i]    for i, lm in enumerate(is_lm) if lm]
    lm_y = [rewards[i] for i, lm in enumerate(is_lm) if lm]
    ax.scatter(lm_x, lm_y, color=ACCENT_GOLD, s=90, zorder=5,
               label=f'Landmark ({len(lm_x)} selected)', edgecolors=MARS_DARK)

    ax.axhline(y=np.mean(rewards), color=ACCENT_CYAN,
               linestyle='--', linewidth=1, alpha=0.7, label='Mean reward')

    ax.set_title('Phase 2 — UCB Reward per Visit', color=MARS_LIGHT,
                 fontsize=13, pad=12)
    ax.set_xlabel('Visit index'); ax.set_ylabel('Reward R')
    ax.legend(facecolor=MARS_MID, edgecolor=MARS_DUST, labelcolor=MARS_LIGHT)
    ax.grid(True)

    fig.tight_layout()
    dest = out / 'phase2_rewards.png'
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {dest}")


# ── Plot 3: Landmark map ───────────────────────────────────────────────────────
def plot_landmark_map(log: dict, out: pathlib.Path):
    visits    = log.get('phase2_visits', [])
    landmarks = log.get('landmarks', [])
    if not visits:
        print("No Phase 2 data for landmark map.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(MARS_DARK)

    # All visited positions (size ∝ reward)
    vx = [v['x'] for v in visits]
    vy = [v['y'] for v in visits]
    vr = [v['reward'] for v in visits]
    norm = Normalize(vmin=min(vr), vmax=max(vr))
    sm   = ScalarMappable(cmap='YlOrRd', norm=norm)

    ax.scatter(vx, vy, c=vr, cmap='YlOrRd', norm=norm,
               s=60, alpha=0.5, zorder=3, edgecolors='none', label='Visited')

    # Landmark positions
    if landmarks:
        lx = [l['x'] for l in landmarks]
        ly = [l['y'] for l in landmarks]
        lr = [l['reward'] for l in landmarks]
        ax.scatter(lx, ly, c=lr, cmap='YlOrRd', norm=norm,
                   s=200, zorder=6, edgecolors=ACCENT_GOLD,
                   linewidths=1.5, marker='*', label=f'{len(landmarks)} landmarks')
        for i, (x, y, r) in enumerate(zip(lx, ly, lr)):
            ax.text(x + 0.5, y + 0.5, f'L{i+1}\n{r:.1f}',
                    fontsize=6.5, color=ACCENT_GOLD, zorder=7)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('UCB Reward', color=MARS_LIGHT)
    cbar.ax.yaxis.set_tick_params(color=MARS_DUST)

    ax.set_title(f'Phase 2 — Landmark Map ({len(landmarks)} landmarks)',
                 color=MARS_LIGHT, fontsize=13, pad=12)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.legend(facecolor=MARS_MID, edgecolor=MARS_DUST, labelcolor=MARS_LIGHT)
    ax.grid(True); ax.set_aspect('equal')

    fig.tight_layout()
    dest = out / 'landmark_map.png'
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {dest}")


# ── Plot 4: SAE concept activation heatmap ────────────────────────────────────
def plot_concept_heatmap(log: dict, out: pathlib.Path):
    acts = log.get('concept_activations')
    if not acts:
        print("No concept activation data (add _train_sae logging patch).")
        return

    A = np.array(acts)   # shape: (n_patches, n_concepts)
    n_patches, n_concepts = A.shape

    fig, ax = plt.subplots(figsize=(max(6, n_concepts * 0.8),
                                     max(5, n_patches * 0.3)))
    fig.patch.set_facecolor(MARS_DARK)

    im = ax.imshow(A, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_xlabel('SAE Concept index'); ax.set_ylabel('Phase 1 patch index')
    ax.set_title(f'SAE Concept Activations ({n_patches} patches × {n_concepts} concepts)',
                 color=MARS_LIGHT, fontsize=12, pad=10)
    ax.set_xticks(range(n_concepts))
    ax.set_xticklabels([f'C{i}' for i in range(n_concepts)])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Activation', color=MARS_LIGHT)
    cbar.ax.yaxis.set_tick_params(color=MARS_DUST)

    fig.tight_layout()
    dest = out / 'sae_concept_heatmap.png'
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {dest}")


# ── Plot 5: Phase 1 vs Phase 2 coverage comparison ────────────────────────────
def plot_coverage_comparison(log: dict, out: pathlib.Path):
    wps    = log.get('phase1_waypoints', [])
    visits = log.get('phase2_visits', [])
    lms    = log.get('landmarks', [])
    if not wps or not visits:
        print("Need both Phase 1 and Phase 2 data for coverage comparison.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(MARS_DARK)

    # Phase 1 — uniform grid
    p1x = [w['x'] for w in wps]
    p1y = [w['y'] for w in wps]
    ax1.scatter(p1x, p1y, color=MARS_DUST, s=80, zorder=4)
    for i in range(len(wps) - 1):
        ax1.plot([p1x[i], p1x[i+1]], [p1y[i], p1y[i+1]],
                 color=MARS_RED, linewidth=1.5, alpha=0.7)
    ax1.set_title('Phase 1 — Uniform Survey', color=MARS_LIGHT, fontsize=12)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.grid(True); ax1.set_aspect('equal')

    # Phase 2 — concept-driven landmarks
    vx = [v['x'] for v in visits]
    vy = [v['y'] for v in visits]
    vr = [v['reward'] for v in visits]
    ax2.scatter(vx, vy, c=vr, cmap='YlOrRd', s=50, alpha=0.5,
                zorder=3, edgecolors='none')
    if lms:
        lx = [l['x'] for l in lms]
        ly = [l['y'] for l in lms]
        ax2.scatter(lx, ly, color=ACCENT_GOLD, s=200,
                    marker='*', zorder=6, edgecolors=MARS_DARK,
                    linewidths=0.8, label='Landmarks')
        ax2.legend(facecolor=MARS_MID, edgecolor=MARS_DUST, labelcolor=MARS_LIGHT)
    ax2.set_title('Phase 2 — Concept-Driven UCB', color=MARS_LIGHT, fontsize=12)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
    ax2.grid(True); ax2.set_aspect('equal')

    fig.suptitle('Coverage: Uniform Survey vs Concept-Guided Exploration',
                 color=MARS_LIGHT, fontsize=13, y=1.01)
    fig.tight_layout()
    dest = out / 'coverage_comparison.png'
    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {dest}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualize mission log')
    parser.add_argument('--log', default='/tmp/mission_log.json')
    parser.add_argument('--out', default='/tmp/mission_plots')
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log = load_log(args.log)

    print(f"\nLoaded log from {args.log}")
    print(f"  Phase 1 waypoints : {len(log.get('phase1_waypoints', []))}")
    print(f"  Phase 2 visits    : {len(log.get('phase2_visits', []))}")
    print(f"  Landmarks         : {len(log.get('landmarks', []))}")
    print(f"  SAE concepts      : {log.get('sae_concepts', 'N/A')}")
    print(f"\nGenerating plots → {out}/\n")

    plot_phase1_path(log, out)
    plot_phase2_rewards(log, out)
    plot_landmark_map(log, out)
    plot_concept_heatmap(log, out)
    plot_coverage_comparison(log, out)

    print(f"\nDone. Open plots in: {out}/")


if __name__ == '__main__':
    main()
