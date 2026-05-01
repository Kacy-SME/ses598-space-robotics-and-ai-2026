#!/usr/bin/env python3
"""
compare_policies.py
===================
Loads mission logs from both v3 (greedy) and v4 (entropy-seeking) runs
and produces a 3-panel comparison figure:

  [0] Cumulative landmarks found vs visits
  [1] Spatial landmark map — greedy vs entropy side by side
  [2] Concept coverage — unique concepts activated per visit

Run after completing both a v3 and v4 mission:
    python3 compare_policies.py

Output: /tmp/policy_comparison.png
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_V3   = pathlib.Path('/tmp/mission_log_v3_clean.json')
LOG_V4   = pathlib.Path('/tmp/mission_log_v4.json')
OUT_PATH = pathlib.Path('/tmp/policy_comparison.png')

REWARD_THRESHOLD = 0.5
GRID_X_MIN, GRID_X_MAX = -12.0, 11.0
GRID_Y_MIN, GRID_Y_MAX = -10.0, 10.0
GRID_STEP = 5.0


def load(path):
    if not path.exists():
        print(f'Missing: {path}')
        return None
    with open(path) as f:
        return json.load(f)


def cumulative_landmarks(visits):
    cum = []
    count = 0
    for v in visits:
        if v.get('is_landmark'):
            count += 1
        cum.append(count)
    return cum


def concept_coverage(visits, concept_activations):
    acts = np.array(concept_activations) if concept_activations else None
    seen = set()
    coverage = []
    for i, v in enumerate(visits):
        wp_idx = v.get('wp_idx', i)
        if acts is not None and wp_idx < len(acts):
            active = np.where(acts[wp_idx] > 0)[0]
            seen.update(active.tolist())
        coverage.append(len(seen))
    return coverage


def main():
    log_v3 = load(LOG_V3)
    log_v4 = load(LOG_V4)

    if log_v3 is None and log_v4 is None:
        print('No logs found. Run both v3 and v4 missions first.')
        return

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 6), facecolor='#0d0d0d')
    fig.suptitle(
        'Policy Comparison: Greedy (v3) vs Entropy-Seeking (v4)',
        fontsize=13, color='#F2CC1A', fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.35,
                           left=0.05, right=0.97, top=0.88, bottom=0.12)

    ax_cum = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[0, 2])

    for ax in [ax_cum, ax_map, ax_cov]:
        ax.set_facecolor('#111111')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    colors = {'v3': '#F24D1A', 'v4': '#1ABDF2'}
    labels = {'v3': 'Greedy (v3)', 'v4': 'Entropy-Seeking (v4)'}

    # Panel 0: Cumulative landmarks
    for key, log in [('v3', log_v3), ('v4', log_v4)]:
        if log is None:
            continue
        visits = log.get('phase2_visits', [])
        if not visits:
            continue
        cum = cumulative_landmarks(visits)
        ax_cum.plot(range(1, len(cum)+1), cum,
                    color=colors[key], linewidth=2.0,
                    marker='o', markersize=3, label=labels[key])

    ax_cum.set_xlabel('Phase 2 visits', color='#888888', fontsize=9)
    ax_cum.set_ylabel('Cumulative landmarks', color='#888888', fontsize=9)
    ax_cum.set_title('Landmark Discovery Rate', color='#888888', fontsize=10, pad=6)
    ax_cum.legend(fontsize=8, framealpha=0.2, labelcolor='white')
    ax_cum.tick_params(colors='#666666', labelsize=8)
    ax_cum.grid(True, color='#1a1a1a', linewidth=0.5)

    # Panel 1: Spatial map
    xs_grid = list(np.arange(GRID_X_MIN, GRID_X_MAX + GRID_STEP, GRID_STEP))
    ys_grid = list(np.arange(GRID_Y_MIN, GRID_Y_MAX + GRID_STEP, GRID_STEP))
    for x in xs_grid:
        for y in ys_grid:
            ax_map.scatter(x, y, s=8, color='#222222', zorder=1)

    for key, log in [('v3', log_v3), ('v4', log_v4)]:
        if log is None:
            continue
        landmarks = log.get('landmarks', [])
        visits    = log.get('phase2_visits', [])
        if visits:
            vxs = [v['x'] for v in visits]
            vys = [v['y'] for v in visits]
            ax_map.plot(vxs, vys, color=colors[key],
                        linewidth=0.8, alpha=0.4, zorder=2)
        offset = -0.8 if key == 'v3' else 0.8
        first  = True
        for lm in landmarks:
            ax_map.scatter(lm['x'] + offset, lm['y'],
                           s=120, color=colors[key],
                           edgecolors='white', linewidths=0.5,
                           zorder=4, marker='*',
                           label=labels[key] if first else '')
            first = False

    ax_map.set_xlim(GRID_X_MIN - 3, GRID_X_MAX + 3)
    ax_map.set_ylim(GRID_Y_MIN - 3, GRID_Y_MAX + 3)
    ax_map.set_xlabel('X (m)', color='#888888', fontsize=9)
    ax_map.set_ylabel('Y (m)', color='#888888', fontsize=9)
    ax_map.set_title('Landmark Spatial Distribution', color='#888888', fontsize=10, pad=6)
    ax_map.tick_params(colors='#666666', labelsize=8)
    ax_map.grid(True, color='#1a1a1a', linewidth=0.5)
    handles, lbls = ax_map.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = h
    ax_map.legend(seen.values(), seen.keys(),
                  fontsize=8, framealpha=0.2, labelcolor='white')

    # Panel 2: Concept coverage
    for key, log in [('v3', log_v3), ('v4', log_v4)]:
        if log is None:
            continue
        visits = log.get('phase2_visits', [])
        acts   = log.get('concept_activations', [])
        if not visits:
            continue
        cov = concept_coverage(visits, acts)
        ax_cov.plot(range(1, len(cov)+1), cov,
                    color=colors[key], linewidth=2.0,
                    marker='s', markersize=3, label=labels[key])

    ax_cov.set_xlabel('Phase 2 visits', color='#888888', fontsize=9)
    ax_cov.set_ylabel('Unique concepts activated', color='#888888', fontsize=9)
    ax_cov.set_title('Concept Space Coverage', color='#888888', fontsize=10, pad=6)
    ax_cov.legend(fontsize=8, framealpha=0.2, labelcolor='white')
    ax_cov.tick_params(colors='#666666', labelsize=8)
    ax_cov.grid(True, color='#1a1a1a', linewidth=0.5)

    print(f'Saving to {OUT_PATH}')
    plt.savefig(str(OUT_PATH), dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print('Done.')

    for key, log in [('v3', log_v3), ('v4', log_v4)]:
        if log is None:
            continue
        visits    = log.get('phase2_visits', [])
        landmarks = log.get('landmarks', [])
        print(f'\n{labels[key]}:')
        print(f'  Visits:    {len(visits)}')
        print(f'  Landmarks: {len(landmarks)}')
        if visits:
            print(f'  Efficiency: {len(landmarks)/len(visits):.3f} landmarks/visit')


if __name__ == '__main__':
    main()
