#!/usr/bin/env python3
"""
visualize_top_patches.py
========================
Loads the SAE checkpoint and Phase 1 frames, finds the top-activating
patches for each dominant landmark concept neuron, and saves a figure.

Run after a completed mission:
    python3 visualize_top_patches.py

Output: /tmp/top_concept_patches.png
"""

import json
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
SAE_CHECKPOINT = pathlib.Path('/tmp/sae_checkpoint.pth')
MISSION_LOG    = pathlib.Path('/tmp/mission_log.json')
FRAMES_DIR     = pathlib.Path('/tmp/mission_frames')
OUT_PATH       = pathlib.Path('/tmp/top_concept_patches.png')

TOP_K_PATCHES  = 5   # how many top patches to show per concept
TOP_N_CONCEPTS = 4   # how many landmark concepts to visualize

# ── SAE definition (must match training) ─────────────────────────────────────
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=384, dict_size=512, topk=32):
        super().__init__()
        self.topk = topk
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)

    def encode(self, x):
        h = F.relu(self.encoder(x))
        if self.topk < h.shape[-1]:
            topk_vals, _ = torch.topk(h, self.topk, dim=-1)
            threshold = topk_vals[:, -1:]
            h = h * (h >= threshold).float()
        return h

    def forward(self, x):
        h = self.encode(x)
        return self.decoder(h), h


# ── DINOv2 embedding ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def embed(dinov2, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = transform(rgb).unsqueeze(0)
    with torch.no_grad():
        return dinov2(t).squeeze(0).numpy().astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('Loading mission log...')
    with open(MISSION_LOG) as f:
        log = json.load(f)

    landmarks   = log.get('landmarks', [])
    acts_list   = log.get('concept_activations', [])  # (N, 512)

    if not landmarks:
        print('No landmarks in log — run a complete mission first.')
        return
    if not acts_list:
        print('No concept_activations in log.')
        return

    acts = np.array(acts_list)  # (N_patches, 512)
    N    = acts.shape[0]

    # Find dominant concept for each landmark (by visit log)
    visits = log.get('phase2_visits', [])
    landmark_concepts = [v['dominant_concept'] for v in visits if v.get('is_landmark')]
    if not landmark_concepts:
        print('No landmark concept indices found in phase2_visits.')
        return

    # Pick top unique concepts by how often they appear as dominant
    from collections import Counter
    concept_counts = Counter(landmark_concepts)
    top_concepts   = [c for c, _ in concept_counts.most_common(TOP_N_CONCEPTS)]
    print(f'Top landmark concepts: {top_concepts}')

    # Load frames
    frame_paths = sorted(FRAMES_DIR.glob('phase1_*.jpg'))
    if len(frame_paths) == 0:
        print(f'No Phase 1 frames found in {FRAMES_DIR}')
        return
    if len(frame_paths) != N:
        print(f'Warning: {len(frame_paths)} frames but {N} activation rows — using min')
        N = min(len(frame_paths), N)
        acts = acts[:N]
        frame_paths = frame_paths[:N]

    print(f'Loading {N} frames...')
    frames = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = len(top_concepts)
    n_cols = TOP_K_PATCHES + 1  # +1 for concept label column

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(3 * n_cols, 3.2 * n_rows), facecolor='#0d0d0d')
    fig.suptitle(
        'Top-Activating Terrain Patches per Landmark SAE Concept',
        fontsize=13, color='#F2CC1A', fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.15, wspace=0.08,
                           left=0.02, right=0.98, top=0.92, bottom=0.02)

    CONCEPT_COLORS = [
        '#F24D1A', '#1ABDF2', '#F2CC1A', '#7DDA2D',
        '#CC2DCC', '#3D7AF2', '#F28A1A', '#AAAAAA',
    ]

    for row, concept_idx in enumerate(top_concepts):
        color = CONCEPT_COLORS[row % len(CONCEPT_COLORS)]

        # Activation scores for this concept across all patches
        concept_scores = acts[:, concept_idx]  # (N,)
        top_patch_idxs = np.argsort(concept_scores)[::-1][:TOP_K_PATCHES]

        # Label column
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis('off')
        ax_label.set_facecolor('#111111')
        ax_label.text(0.5, 0.65, f'Concept\n{concept_idx}',
                      ha='center', va='center', fontsize=11,
                      fontweight='bold', color=color,
                      transform=ax_label.transAxes)
        ax_label.text(0.5, 0.28,
                      f'fires {(concept_scores > 0).mean()*100:.0f}% of patches\n'
                      f'max activation: {concept_scores.max():.2f}',
                      ha='center', va='center', fontsize=7,
                      color='#888888', transform=ax_label.transAxes)

        for col, patch_idx in enumerate(top_patch_idxs):
            ax = fig.add_subplot(gs[row, col + 1])
            img = frames[patch_idx]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.axis('off')

            score = concept_scores[patch_idx]
            # Activation bar overlay at bottom
            bar_width = score / (concept_scores.max() + 1e-6)
            ax.add_patch(plt.Rectangle(
                (0, img_rgb.shape[0] * 0.88),
                img_rgb.shape[1] * bar_width, img_rgb.shape[0] * 0.12,
                transform=ax.transData,
                color=color, alpha=0.75, zorder=5
            ))
            ax.set_title(f'{score:.2f}', fontsize=7, color=color, pad=2)

            # Highlight top patch with colored border
            if col == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

    print(f'Saving to {OUT_PATH}')
    plt.savefig(str(OUT_PATH), dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d')
    plt.close()
    print('Done.')


if __name__ == '__main__':
    main()
