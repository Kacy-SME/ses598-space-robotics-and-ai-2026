#!/usr/bin/env python3
"""
live_mission_dashboard_v5.py
============================
Live dashboard for v5 sequential greedy → entropy mission.
Shows both policies side by side as they execute.

Run:
    python3 live_mission_dashboard_v5.py [--log /tmp/mission_log_v5.json]
"""

import json
import pathlib
import time
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

LOG_PATH = pathlib.Path(
    sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--log'
    else '/tmp/mission_log_v5.json')
POLL_SEC = 2.0
REWARD_THRESHOLD = 0.5

CONCEPT_COLORS = [
    '#F24D1A', '#1ABDF2', '#F2CC1A', '#7DDA2D',
    '#CC2DCC', '#3D7AF2', '#F28A1A', '#AAAAAA',
]

PHASE_LABELS = {
    'PRE_ARM':           ('PRE-ARM',              '#555555'),
    'ENGAGING_OFFBOARD': ('ENGAGING OFFBOARD',    '#888800'),
    'ARM_TAKEOFF':       ('ARMING / TAKEOFF',     '#BB6600'),
    'PHASE1_FLY':        ('◉ PHASE 1 — SURVEY',  '#1ABDF2'),
    'TRAIN_SAE':         ('⚙ TRAINING SAE…',     '#F2CC1A'),
    'PHASE2A_GREEDY':    ('▶ PHASE 2a — GREEDY', '#F24D1A'),
    'PHASE2B_ENTROPY':   ('★ PHASE 2b — ENTROPY','#7DDA2D'),
    'DONE':              ('✔ MISSION COMPLETE',   '#F2CC1A'),
}

GRID_X_MIN, GRID_X_MAX = -12.0, 11.0
GRID_Y_MIN, GRID_Y_MAX = -10.0, 10.0
GRID_STEP = 5.0


def load_log():
    if not LOG_PATH.exists():
        return None
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def infer_phase(log):
    if log.get('phase2b_visits'):
        if len(log['phase2b_visits']) >= log.get('greedy_n', 15):
            return 'DONE'
        return 'PHASE2B_ENTROPY'
    if log.get('phase2a_visits'):
        return 'PHASE2A_GREEDY'
    if log.get('concept_activations'):
        return 'TRAIN_SAE'
    if log.get('phase1_waypoints'):
        return 'PHASE1_FLY'
    return 'PRE_ARM'


def build_dashboard():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d0d')
    fig.suptitle(
        'SES 598 · SAE Concept Terrain Mission v5 · Greedy → Entropy · LIVE',
        fontsize=13, color='#F2CC1A', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(
        2, 4, hspace=0.45, wspace=0.35,
        left=0.05, right=0.97, top=0.92, bottom=0.06)

    ax_status  = fig.add_subplot(gs[0, 0])
    ax_heat    = fig.add_subplot(gs[0, 1:3])
    ax_stats   = fig.add_subplot(gs[0, 3])
    ax_map     = fig.add_subplot(gs[1, 0:2])
    ax_greedy  = fig.add_subplot(gs[1, 2])
    ax_entropy = fig.add_subplot(gs[1, 3])

    for ax in [ax_status, ax_heat, ax_stats,
               ax_map, ax_greedy, ax_entropy]:
        ax.set_facecolor('#111111')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    return fig, ax_status, ax_heat, ax_stats, ax_map, ax_greedy, ax_entropy


def render(fig, ax_status, ax_heat, ax_stats, ax_map, ax_greedy, ax_entropy, log):
    for ax in [ax_status, ax_heat, ax_stats, ax_map, ax_greedy, ax_entropy]:
        ax.cla()
        ax.set_facecolor('#111111')

    phase = infer_phase(log)
    phase_label, phase_color = PHASE_LABELS.get(phase, (phase, '#FFFFFF'))

    p1_n  = len(log.get('phase1_waypoints', []))
    p2a_n = len(log.get('phase2a_visits', []))
    p2b_n = len(log.get('phase2b_visits', []))
    lm_g  = len(log.get('greedy_landmarks', []))
    lm_e  = len(log.get('entropy_landmarks', []))
    n_con = log.get('sae_concepts', 0)
    greedy_n = log.get('greedy_n', 15)

    # ── Status panel ─────────────────────────────────────────────────────────
    ax_status.axis('off')
    ax_status.text(0.5, 0.82, phase_label, ha='center', va='center',
                   fontsize=11, fontweight='bold', color=phase_color,
                   transform=ax_status.transAxes)
    stats = [
        f'Phase 1 waypoints:  {p1_n}',
        f'SAE concepts:       {n_con}',
        f'Phase 2a (greedy):  {p2a_n}/{greedy_n}',
        f'Phase 2b (entropy): {p2b_n}',
        f'Greedy landmarks:   {lm_g}',
        f'Entropy landmarks:  {lm_e}',
    ]
    for i, s in enumerate(stats):
        color = '#F24D1A' if 'greedy' in s.lower() else \
                '#7DDA2D' if 'entropy' in s.lower() else '#CCCCCC'
        ax_status.text(0.05, 0.65 - i * 0.10, s, ha='left', va='center',
                       fontsize=8, color=color,
                       transform=ax_status.transAxes, family='monospace')
    ax_status.set_title('Mission Status', color='#888888', fontsize=9, pad=4)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    acts = log.get('concept_activations')
    if acts and n_con > 0:
        A = np.array(acts)
        A_sel = A[:, A.max(axis=0) > 0]
        step  = max(1, A_sel.shape[0] // 60)
        A_vis = A_sel[::step, :]
        im = ax_heat.imshow(A_vis.T, aspect='auto', cmap='inferno',
                            interpolation='nearest')
        ax_heat.set_xlabel('Patch index', color='#888888', fontsize=8)
        ax_heat.set_ylabel('SAE concept neuron', color='#888888', fontsize=8)
        ax_heat.tick_params(colors='#666666', labelsize=7)
        ax_heat.set_title(
            f'SAE Concept Activations  '
            f'[{A_vis.shape[1]} active × {A_vis.shape[0]} patches]',
            color='#888888', fontsize=9, pad=4)
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
        cbar.ax.tick_params(colors='#666666', labelsize=7)
    else:
        ax_heat.text(0.5, 0.5, 'Waiting for SAE training…',
                     ha='center', va='center', color='#555555', fontsize=11,
                     transform=ax_heat.transAxes)
        ax_heat.set_title('SAE Concept Activations',
                          color='#888888', fontsize=9, pad=4)

    # ── Stats comparison ─────────────────────────────────────────────────────
    ax_stats.axis('off')
    ax_stats.set_title('Policy Comparison', color='#888888', fontsize=9, pad=4)
    rows = [
        ('', 'Greedy', 'Entropy'),
        ('Visits', str(p2a_n), str(p2b_n)),
        ('Landmarks', str(lm_g), str(lm_e)),
        ('Efficiency',
         f'{lm_g/p2a_n:.2f}' if p2a_n > 0 else '—',
         f'{lm_e/p2b_n:.2f}' if p2b_n > 0 else '—'),
    ]
    for i, (label, v3, v4) in enumerate(rows):
        y = 0.88 - i * 0.20
        ax_stats.text(0.05, y, label, color='#888888', fontsize=9,
                      transform=ax_stats.transAxes, family='monospace')
        ax_stats.text(0.45, y, v3, color='#F24D1A', fontsize=9,
                      fontweight='bold' if i == 0 else 'normal',
                      transform=ax_stats.transAxes, family='monospace')
        ax_stats.text(0.75, y, v4, color='#7DDA2D', fontsize=9,
                      fontweight='bold' if i == 0 else 'normal',
                      transform=ax_stats.transAxes, family='monospace')

    # ── Spatial map ───────────────────────────────────────────────────────────
    xs_g = list(np.arange(GRID_X_MIN, GRID_X_MAX + GRID_STEP, GRID_STEP))
    ys_g = list(np.arange(GRID_Y_MIN, GRID_Y_MAX + GRID_STEP, GRID_STEP))
    for x in xs_g:
        for y in ys_g:
            ax_map.scatter(x, y, s=8, color='#222222', zorder=1)

    # Phase 2a greedy path
    p2a = log.get('phase2a_visits', [])
    if p2a:
        ax_map.plot([v['x'] for v in p2a], [v['y'] for v in p2a],
                    color='#F24D1A', linewidth=0.8, alpha=0.5, zorder=2)

    # Phase 2b entropy path
    p2b = log.get('phase2b_visits', [])
    if p2b:
        ax_map.plot([v['x'] for v in p2b], [v['y'] for v in p2b],
                    color='#7DDA2D', linewidth=0.8, alpha=0.5, zorder=2,
                    linestyle='--')

    # Greedy landmarks
    for lm in log.get('greedy_landmarks', []):
        ax_map.scatter(lm['x'] - 0.5, lm['y'], s=120, color='#F24D1A',
                       edgecolors='white', linewidths=0.5,
                       zorder=4, marker='*')

    # Entropy landmarks
    for lm in log.get('entropy_landmarks', []):
        ax_map.scatter(lm['x'] + 0.5, lm['y'], s=120, color='#7DDA2D',
                       edgecolors='white', linewidths=0.5,
                       zorder=4, marker='D')

    ax_map.set_xlim(GRID_X_MIN - 3, GRID_X_MAX + 3)
    ax_map.set_ylim(GRID_Y_MIN - 3, GRID_Y_MAX + 3)
    ax_map.set_xlabel('X (m)', color='#888888', fontsize=8)
    ax_map.set_ylabel('Y (m)', color='#888888', fontsize=8)
    ax_map.set_title(
        f'Terrain Map  '
        f'[★ greedy={lm_g}  ◆ entropy={lm_e}]',
        color='#888888', fontsize=9, pad=4)
    ax_map.tick_params(colors='#666666', labelsize=7)
    ax_map.grid(True, color='#1a1a1a', linewidth=0.5)
    patches = [
        mpatches.Patch(color='#F24D1A', label='Greedy path/landmarks'),
        mpatches.Patch(color='#7DDA2D', label='Entropy path/landmarks'),
    ]
    ax_map.legend(handles=patches, fontsize=6,
                  framealpha=0.2, labelcolor='white')

    # ── Greedy reward bars ────────────────────────────────────────────────────
    if p2a:
        idxs    = list(range(len(p2a)))
        rewards = [v['reward'] for v in p2a]
        colors  = [('#F24D1A' if r > REWARD_THRESHOLD else '#444444')
                   for r in rewards]
        ax_greedy.bar(idxs, rewards, color=colors, width=0.8)
        ax_greedy.axhline(REWARD_THRESHOLD, color='#FFFFFF',
                          linewidth=1.0, linestyle='--')
        ax_greedy.set_ylim(0, max(rewards) * 1.15 + 0.1)
    else:
        ax_greedy.text(0.5, 0.5, 'Waiting for Phase 2a…',
                       ha='center', va='center', color='#555555', fontsize=9,
                       transform=ax_greedy.transAxes)
    ax_greedy.set_title('Greedy Rewards (top 15)',
                        color='#F24D1A', fontsize=9, pad=4)
    ax_greedy.set_xlabel('Visit', color='#888888', fontsize=8)
    ax_greedy.set_ylabel('SAE reward', color='#888888', fontsize=8)
    ax_greedy.tick_params(colors='#666666', labelsize=7)

    # ── Entropy reward bars ───────────────────────────────────────────────────
    if p2b:
        idxs    = list(range(len(p2b)))
        rewards = [v['reward'] for v in p2b]
        colors  = [('#7DDA2D' if r > REWARD_THRESHOLD else '#444444')
                   for r in rewards]
        ax_entropy.bar(idxs, rewards, color=colors, width=0.8)
        ax_entropy.axhline(REWARD_THRESHOLD, color='#FFFFFF',
                           linewidth=1.0, linestyle='--')
        ax_entropy.set_ylim(0, max(rewards) * 1.15 + 0.1)
    else:
        ax_entropy.text(0.5, 0.5, 'Waiting for Phase 2b…',
                        ha='center', va='center', color='#555555', fontsize=9,
                        transform=ax_entropy.transAxes)
    ax_entropy.set_title('Entropy Rewards (bottom 15)',
                         color='#7DDA2D', fontsize=9, pad=4)
    ax_entropy.set_xlabel('Visit', color='#888888', fontsize=8)
    ax_entropy.set_ylabel('SAE reward', color='#888888', fontsize=8)
    ax_entropy.tick_params(colors='#666666', labelsize=7)

    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    print(f'Watching {LOG_PATH} — close window to exit')
    fig, ax_status, ax_heat, ax_stats, ax_map, ax_greedy, ax_entropy = \
        build_dashboard()
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
                render(fig, ax_status, ax_heat, ax_stats,
                       ax_map, ax_greedy, ax_entropy, log)
                last_mod = mod

        plt.pause(POLL_SEC)

    print('Dashboard closed.')


if __name__ == '__main__':
    main()
