#!/usr/bin/env python3
"""
live_mission_dashboard.py
=========================
Standalone script — no ROS required.
Run in a separate terminal during flight:

    python3 live_mission_dashboard.py

Watches /tmp/mission_log.json every 2 seconds and renders a live
4-panel matplotlib dashboard:
  [0] Phase status + stats
  [1] SAE concept activation heatmap (Phase 1 patches × selective concepts)
  [2] Drone flight path + landmark map (colored by dominant concept)
  [3] Phase 2 reward bar chart (per visit, threshold line)
"""

import json
import pathlib
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # works in VNC; fall back to 'Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

LOG_PATH = pathlib.Path('/tmp/mission_log.json')
POLL_SEC = 2.0
REWARD_THRESHOLD = 0.5

CONCEPT_COLORS = [
    '#F24D1A',  # C0 Mars red
    '#1ABDF2',  # C1 cyan
    '#F2CC1A',  # C2 gold
    '#7DDA2D',  # C3 green
    '#CC2DCC',  # C4 magenta
    '#3D7AF2',  # C5 blue
    '#F28A1A',  # C6 orange
    '#AAAAAA',  # C7 grey
]

PHASE_LABELS = {
    'PRE_ARM':           ('PRE-ARM',         '#555555'),
    'ENGAGING_OFFBOARD': ('ENGAGING OFFBOARD','#888800'),
    'ARM_TAKEOFF':       ('ARMING / TAKEOFF', '#BB6600'),
    'PHASE1_FLY':        ('◉ PHASE 1 — SURVEY','#1ABDF2'),
    'TRAIN_SAE':         ('⚙ TRAINING SAE…',  '#F2CC1A'),
    'PHASE2_FLY':        ('★ PHASE 2 — LANDMARK','#7DDA2D'),
    'DONE':              ('✔ MISSION COMPLETE','#F24D1A'),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_log():
    if not LOG_PATH.exists():
        return None
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def infer_phase(log):
    """Guess current phase from log contents (mission node writes phase in log)."""
    if log.get('landmarks'):
        if len(log['phase2_visits']) > 0:
            last = log['phase2_visits'][-1]
            # If last visit reward < threshold many times, probably DONE
            return 'PHASE2_FLY'
    if log.get('concept_activations'):
        return 'TRAIN_SAE' if not log.get('phase2_visits') else 'PHASE2_FLY'
    if log.get('phase1_waypoints'):
        return 'PHASE1_FLY'
    return 'PRE_ARM'


# ── Dashboard ────────────────────────────────────────────────────────────────

def build_dashboard():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor='#0d0d0d')
    fig.suptitle(
        'Concept-Driven Sparse Landmark Mapping · LIVE',
        fontsize=14, color='#F2CC1A', fontweight='bold', y=0.98
    )

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)
    ax_status  = fig.add_subplot(gs[0, 0])   # phase status
    ax_heat    = fig.add_subplot(gs[0, 1:])  # concept heatmap (wide)
    ax_map     = fig.add_subplot(gs[1, 0:2]) # spatial map
    ax_rewards = fig.add_subplot(gs[1, 2])   # reward bar chart

    for ax in [ax_status, ax_heat, ax_map, ax_rewards]:
        ax.set_facecolor('#111111')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    return fig, ax_status, ax_heat, ax_map, ax_rewards


def render(fig, ax_status, ax_heat, ax_map, ax_rewards, log):
    for ax in [ax_status, ax_heat, ax_map, ax_rewards]:
        ax.cla()
        ax.set_facecolor('#111111')

    phase = infer_phase(log)
    phase_label, phase_color = PHASE_LABELS.get(phase, (phase, '#FFFFFF'))

    # ── Panel 0: Status ──────────────────────────────────────────────────────
    ax_status.axis('off')
    ax_status.text(0.5, 0.80, phase_label, ha='center', va='center',
                   fontsize=13, fontweight='bold', color=phase_color,
                   transform=ax_status.transAxes)

    p1_n  = len(log.get('phase1_waypoints', []))
    p2_n  = len(log.get('phase2_visits', []))
    lm_n  = len(log.get('landmarks', []))
    n_concepts = log.get('sae_concepts', 0)

    stats = [
        f'Phase 1 waypoints:  {p1_n}',
        f'SAE concepts:       {n_concepts}',
        f'Phase 2 visits:     {p2_n}',
        f'Landmarks found:    {lm_n}',
    ]
    for i, s in enumerate(stats):
        ax_status.text(0.08, 0.58 - i * 0.14, s, ha='left', va='center',
                       fontsize=9, color='#CCCCCC',
                       transform=ax_status.transAxes, family='monospace')

    ax_status.set_title('Mission Status', color='#888888', fontsize=9, pad=4)

    # ── Panel 1: Concept activation heatmap ─────────────────────────────────
    acts = log.get('concept_activations')
    if acts and n_concepts > 0:
        A = np.array(acts)                        # (N_patches, 512)
        active = A.max(axis=0) > 0
        A_sel  = A[:, active]                     # (N, n_active)

        # Subsample patches for readability
        max_show = 60
        step = max(1, A_sel.shape[0] // max_show)
        A_vis = A_sel[::step, :]

        im = ax_heat.imshow(A_vis.T, aspect='auto', cmap='inferno',
                            interpolation='nearest')
        ax_heat.set_xlabel('Patch index (subsampled)', color='#888888', fontsize=8)
        ax_heat.set_ylabel('SAE concept neuron', color='#888888', fontsize=8)
        ax_heat.tick_params(colors='#666666', labelsize=7)
        ax_heat.set_title(
            f'SAE Concept Activations  [{A_vis.shape[1]} active concepts × {A_vis.shape[0]} patches]',
            color='#888888', fontsize=9, pad=4
        )
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
        cbar.ax.tick_params(colors='#666666', labelsize=7)
    else:
        ax_heat.text(0.5, 0.5, 'Waiting for SAE training…',
                     ha='center', va='center', color='#555555', fontsize=11,
                     transform=ax_heat.transAxes)
        ax_heat.set_title('SAE Concept Activations', color='#888888', fontsize=9, pad=4)

    # ── Panel 2: Spatial map ─────────────────────────────────────────────────
    # Phase 1 waypoints (visited)
    p1_wps = log.get('phase1_waypoints', [])
    if p1_wps:
        xs = [w['x'] for w in p1_wps]
        ys = [w['y'] for w in p1_wps]
        ax_map.scatter(xs, ys, s=12, color='#334455', zorder=1, label='Phase 1 survey')
        # Flight path line
        ax_map.plot(xs, ys, color='#223344', linewidth=0.6, zorder=1)

    # Phase 2 visits (non-landmark)
    visits = log.get('phase2_visits', [])
    non_lm = [v for v in visits if not v.get('is_landmark')]
    if non_lm:
        ax_map.scatter([v['x'] for v in non_lm],
                       [v['y'] for v in non_lm],
                       s=18, color='#444444', zorder=2, label='Phase 2 (no landmark)')

    # Landmarks — colored by dominant concept
    landmarks = log.get('landmarks', [])
    for lm in landmarks:
        cidx = lm.get('dominant_concept', 0) % len(CONCEPT_COLORS)
        ax_map.scatter(lm['x'], lm['y'],
                       s=120, color=CONCEPT_COLORS[cidx],
                       edgecolors='white', linewidths=0.5,
                       zorder=4, marker='*')
        ax_map.annotate(
            f"C{cidx}\n{lm['reward']:.1f}",
            (lm['x'], lm['y']),
            textcoords='offset points', xytext=(5, 5),
            fontsize=6, color=CONCEPT_COLORS[cidx]
        )

    # Grid boundary
    ax_map.set_xlim(-17, 16); ax_map.set_ylim(-15, 15)
    ax_map.set_xlabel('X (m)', color='#888888', fontsize=8)
    ax_map.set_ylabel('Y (m)', color='#888888', fontsize=8)
    ax_map.tick_params(colors='#666666', labelsize=7)
    ax_map.set_title(
        f'Terrain Map  [{lm_n} landmarks]',
        color='#888888', fontsize=9, pad=4
    )
    ax_map.grid(True, color='#1a1a1a', linewidth=0.5)

    # Threshold line legend
    if landmarks:
        patches = [mpatches.Patch(color=CONCEPT_COLORS[i % len(CONCEPT_COLORS)],
                                  label=f'Concept {i}')
                   for i in sorted({lm.get('dominant_concept', 0) for lm in landmarks})]
        ax_map.legend(handles=patches, loc='upper right',
                      fontsize=6, framealpha=0.2, labelcolor='white')

    # ── Panel 3: Phase 2 reward bars ─────────────────────────────────────────
    if visits:
        idxs    = [v['idx'] for v in visits]
        rewards = [v['reward'] for v in visits]
        colors  = [('#7DDA2D' if r > REWARD_THRESHOLD else '#444444') for r in rewards]
        ax_rewards.bar(idxs, rewards, color=colors, width=0.8)
        ax_rewards.axhline(REWARD_THRESHOLD, color='#F24D1A',
                           linewidth=1.2, linestyle='--', label=f'threshold={REWARD_THRESHOLD}')
        ax_rewards.set_xlabel('Phase 2 visit index', color='#888888', fontsize=8)
        ax_rewards.set_ylabel('SAE reward', color='#888888', fontsize=8)
        ax_rewards.set_ylim(0, max(rewards) * 1.15 + 0.1)
        ax_rewards.tick_params(colors='#666666', labelsize=7)
        ax_rewards.legend(fontsize=7, framealpha=0.2, labelcolor='white')
    else:
        ax_rewards.text(0.5, 0.5, 'Waiting for Phase 2…',
                        ha='center', va='center', color='#555555', fontsize=10,
                        transform=ax_rewards.transAxes)

    ax_rewards.set_title('Phase 2 Concept Rewards', color='#888888', fontsize=9, pad=4)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f'Watching {LOG_PATH} every {POLL_SEC}s — close window to exit')
    fig, ax_status, ax_heat, ax_map, ax_rewards = build_dashboard()
    plt.ion()
    plt.show(block=False)

    last_mod = None
    while plt.fignum_exists(fig.number):
        try:
            mod = LOG_PATH.stat().st_mtime if LOG_PATH.exists() else None
        except Exception:
            mod = None

        if mod != last_mod:
            log = load_log()
            if log is not None:
                render(fig, ax_status, ax_heat, ax_map, ax_rewards, log)
                last_mod = mod

        plt.pause(POLL_SEC)

    print('Dashboard closed.')


if __name__ == '__main__':
    main()
