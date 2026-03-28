# T(p)-Gated Sparse Landmark Mapping
**Reliability-Aware Cross-Domain Terrain Correspondence for Illumination-Invariant Planetary Navigation**
 
HEART AI Lab, ASU — Kacy Hatfield — Advisor: Dr. Lindsay Sanneman
 
---
 
## Overview
 
This repository contains three components:
 
1. **Mars pipeline** — T(p) computation, patch extraction, and DINOv2 fine-tuning on Curiosity NavCam / HiRISE orbital pairs from Gale Crater
2. **Earth analog pipeline** — the same T(p)-weighted correspondence framework applied to terrestrial analog imagery to test generalization
3. **Earth analog labeler** — an interactive GUI tool for labeling matched overhead/rover correspondence points in image pairs
 
The core idea: use the shape of the terrain (from a real elevation model) to predict which patches will look similar under different lighting conditions, then use that prediction both to train a visual matching model and to filter the reference map it searches.
 
---
 
## Repository Structure
 
```
.
├── mars/
│   ├── mars_pipeline.ipynb       # Main Colab notebook — patch extraction, T(p), fine-tuning
│   └── valid_records_clean.pkl   # 30,659 manually curated terrain-only NavCam patches
│
├── earth_analog/
│   ├── earth_analog_prototype.ipynb   # Earth analog pipeline notebook
│   └── earth_pairs/                   # Labeled correspondence pairs (output of labeler)
│       ├── IMG_7964__IMG_7973__R1/
│       │   ├── overhead_patch.jpg
│       │   ├── rover_patch.jpg
│       │   └── metadata.json
│       └── ...
│
├── labeler.py                    # Earth analog correspondence labeling tool (see below)
│
└── README.md
```
 
---
 
## Mars Pipeline
 
### Data
 
| Resource | Location | Description |
|----------|----------|-------------|
| HiRISE strip | `ESP_035350_1755_RED.JP2` | Gale Crater, 0.25 m/pixel, Feb 2014 |
| NavCam images | `NAVCAM/` | Curiosity sols 437-535 |
| Rover positions | `localized_pos.csv` | SPICE-verified GPS, ROVER frame |
| Patch scores cache | `navcam_patch_scores.pkl` | Precomputed T(p) and metadata for all patches |
| Clean records | `valid_records_clean.pkl` | 30,659 terrain-only patches after manual curation |
| Fine-tuned model | `dinov2_finetuned_v4.pth` | DINOv2-ViT-S/14, blocks.11 + norm unfrozen |
 
All data lives in Google Drive at `/content/drive/MyDrive/NASA_Mars_Curiosity/`.
 
### Reconnect sequence (after Colab disconnect)
 
```python
!pip install -q rasterio opencv-python-headless phasepack
 
from google.colab import drive
drive.mount('/content/drive')
 
import os, pickle, numpy as np, pandas as pd
import rasterio; from rasterio.windows import Window
import torch, torch.nn.functional as F
import torchvision.transforms as T
from scipy.stats import spearmanr
from tqdm import tqdm
from PIL import Image
 
# Load clean records
with open('/content/drive/MyDrive/NASA_Mars_Curiosity/valid_records_clean.pkl', 'rb') as f:
    valid_records_clean = pickle.load(f)
 
# Load rover positions
HIRISE_PATH = '/content/drive/MyDrive/NASA_Mars_Curiosity/ESP_035350_1755_RED.JP2'
pos = pd.read_csv('/content/drive/MyDrive/NASA_Mars_Curiosity/localized_pos.csv')
pos = pos[(pos['frame'] == 'ROVER') & (pos['sol'] >= 0)]
sol_positions = pos.groupby('sol')[['planetocentric_latitude', 'longitude']].median().reset_index()
sol_positions['sol_str'] = sol_positions['sol'].astype(int).apply(lambda x: f'{x:04d}')
 
def latlon_to_jp2_coords(lat, lon):
    R, lon0 = 3396190.0, 180.0
    return R * np.radians(lon - lon0), R * np.radians(lat)
 
sol_positions['x_m'], sol_positions['y_m'] = zip(*sol_positions.apply(
    lambda r: latlon_to_jp2_coords(r['planetocentric_latitude'], r['longitude']), axis=1))
 
# Load model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.load_state_dict(torch.load('/content/drive/MyDrive/Spring25/dinov2_finetuned_v4.pth'))
model.eval().to('cuda')
```
 
### T(p) score
 
T(p) is computed from DTM surface normals under two synthetic illumination directions:
- Orbital: az=135 deg, el=55 deg (afternoon HiRISE acquisition)
- Rover: az=45 deg, el=25 deg (early-morning MSR landing scenario)
 
```
T(p) = corr(I_synth_orbital, I_synth_rover)
```
 
T(p) near +1 indicates terrain stable under illumination change (ridges, rocks).
T(p) near -1 indicates terrain that inverts (flat regolith, shallow shadows).
 
In this dataset, DTM-derived T(p) achieved std=0.173, 22x greater dynamic range than image-based phase congruency (std=0.008).
 
### Fine-tuning
 
- Model: DINOv2-ViT-S/14
- Frozen layers: all except `blocks.11` and `norm` (1.78M / 22M parameters trainable)
- Loss: T(p)-weighted InfoNCE, temperature tau=0.07
- Optimizer: AdamW, lr=1e-5, weight decay=0.01, gradient clipping norm=1.0
- Epochs: 30, batch size: 16
- Dataset: 900 HiRISE/NavCam pairs across 25 sols
 
**v4 results (manually cleaned patches):**
- Same-pair cosine similarity: 0.175 vs random baseline 0.023 (delta=0.151)
- Spearman r(T(p), similarity): 0.505, p<0.0001
- T(p) thirds: Low=0.111, Mid=0.185, High=0.226
 
---
 
## Earth Analog Pipeline
 
### Setup
 
Eight HEIC image pairs captured at a Mars analog terrain site (dry soil, rock fragments, dry grass). Each pair consists of:
- **Overhead image** (nadir-looking, HiRISE analog): `IMG_7964.HEIC` - `IMG_7971.HEIC`
- **Rover image** (oblique ground-level, NavCam analog): `IMG_7973.HEIC` - `IMG_7980.HEIC`
 
Images stored at `/content/drive/MyDrive/trn_labeler/`.
Labeled pairs stored at `/content/drive/MyDrive/trn_labeler/earth_pairs/`.
 
### Dice exclusion
 
Scale markers (dice) were placed in the scene during capture. Their pixel locations are recorded in each `metadata.json` file under `overhead.pixel_xy` and `rover.pixel_xy`. These coordinates are used as exclusion zones during patch sampling — a circular mask is drawn around each dice location so no extracted patch overlaps with a marker.
 
```python
# Exclusion mask from labeled dice positions
oh_dice_positions = defaultdict(list)   # image_name -> [(x, y), ...]
rv_dice_positions = defaultdict(list)
 
for d in pair_dirs:
    with open(d / 'metadata.json') as f:
        meta = json.load(f)
    oh_dice_positions[Path(meta['overhead']['image']).name].append(
        tuple(meta['overhead']['pixel_xy']))
    rv_dice_positions[Path(meta['rover']['image']).name].append(
        tuple(meta['rover']['pixel_xy']))
```
 
### T(p) proxy (Earth analog)
 
A full DTM was unavailable for the Earth analog site. T(p) is approximated from image texture statistics:
 
```python
def compute_tp(pil_patch):
    arr = np.array(pil_patch.convert('L')).astype(np.float32) / 255.0
    # H(p): spectral entropy via SVD eigenvalue spectrum
    U, s, Vt = np.linalg.svd(arr, full_matrices=False)
    s_norm = s / (s.sum() + 1e-8)
    Hp = -np.sum(s_norm * np.log(s_norm + 1e-8))
    # R(p): gradient-based stability proxy
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    Rp = (gx + gy) / 2.0
    return float(Hp + Rp)
```
 
### Fine-tuning
 
- Initialized from Mars v4 weights (`dinov2_finetuned_v4.pth`)
- Same architecture: blocks.11 + norm unfrozen
- AdamW lr=5e-6, 30 epochs (lower LR — fine-tuning from v4)
- 106 geo-registered overhead-rover patch pairs across 8 image pairs
- Saved to: `dinov2_earth_analog_v1.pth`
 
**Earth analog v1 results:**
- Same-pair cosine similarity: 0.377 vs random baseline 0.124 (delta=0.253)
- Spearman r(T(p), similarity): 0.362, p=0.0001
- T(p) thirds: Low=0.313, Mid=0.415, High=0.403
 
---
 
## Earth Analog Labeler (`labeler.py`)
 
An interactive GUI tool for labeling matched overhead/rover correspondence points.
 
### Requirements
 
```bash
pip install pillow pillow-heif matplotlib numpy
```
 
### Usage
 
```bash
cd /path/to/your/images
python labeler.py
```
 
Images must be in the same directory as `labeler.py`, or edit `IMAGE_DIR` at the top of the file.
 
### Controls
 
| Key / Action | Effect |
|---|---|
| Click LEFT panel | Mark overhead point |
| Click RIGHT panel | Mark rover point |
| `1` - `9` | Select landmark ID (D1, D2, R1, R2, G1, G2, S1, S2, X1) |
| `M` | Save as Match |
| `N` | Save as Negative |
| `C` | Clear pending clicks |
| `Right arrow` / `Left arrow` | Next / previous image pair |
| `U` | Undo (delete last saved pair for current ID) |
| `Q` | Quit |
 
### Workflow
 
1. Navigate to a pair with `Left` / `Right` arrows
2. Press a number key (`1`-`9`) to select a landmark ID
3. If this landmark was labeled on a previous pair, a ghost marker shows its last position — press `M` to accept it or click to override
4. Click the overhead panel to place a point, then click the rover panel
5. Press `M` to save as a matched pair, or `N` to save as a negative
6. Repeat across all pairs
 
### Output format
 
Each saved pair creates a subdirectory in `earth_pairs/`:
 
```
earth_pairs/
  IMG_7964__IMG_7973__R1/
    overhead_patch.jpg    # 64x64 px crop from overhead image
    rover_patch.jpg       # 64x64 px crop from rover image
    metadata.json         # coordinates, patch statistics, match flag
```
 
**`metadata.json` structure:**
 
```json
{
  "pair_id": "IMG_7964__IMG_7973__R1",
  "landmark_id": "R1",
  "is_match": true,
  "overhead": {
    "image": "./IMG_7964.HEIC",
    "pixel_xy": [1045, 1737],
    "viewpoint": "overhead",
    "entropy_H": 4.827,
    "svd_rank": 0.313,
    "grad_mag": 0.208,
    "contrast": 0.233
  },
  "rover": {
    "image": "./IMG_7973.HEIC",
    "pixel_xy": [2112, 1485],
    "viewpoint": "rover",
    "entropy_H": 3.854,
    "svd_rank": 0.692,
    "grad_mag": 0.038,
    "contrast": 0.114
  },
  "scale_ratio": 4.62,
  "delta_entropy_H": 0.973,
  "delta_svd_rank": 0.379,
  "delta_grad_mag": 0.170,
  "min_entropy_H": 3.854,
  "min_grad_mag": 0.038,
  "score_sp": 2.881
}
```
 
**Landmark ID conventions used in this dataset:**
 
| ID | Meaning |
|----|---------|
| D1, D2 | Dice (scale markers — used as exclusion zones, not terrain features) |
| R1, R2 | Rock features |
| G1, G2 | Ground texture features |
| S1, S2 | Soil features |
| X1 | Extra / miscellaneous |
 
### Configuration
 
Edit the constants at the top of `labeler.py`:
 
```python
IMAGE_DIR         = '.'           # Directory containing HEIC images
OUTPUT_DIR        = 'earth_pairs' # Output directory for labeled pairs
OVERHEAD_HEIGHT_M = 2.0           # Approximate overhead camera height (metres)
ROVER_HEIGHT_M    = 0.5           # Approximate rover camera height (metres)
ROVER_ANGLE_DEG   = 30.0          # Rover camera angle from horizontal (degrees)
DISPLAY_SCALE     = 0.38          # Display thumbnail scale (1.0 = full size)
PATCH_SIZE        = 64            # Extracted patch size in pixels
```
 
The `scale_ratio` saved in metadata is computed automatically from these values:
 
```python
scale_ratio = OVERHEAD_HEIGHT_M / (ROVER_HEIGHT_M * cos(ROVER_ANGLE_DEG))
```
 
---
 
## Saved Models
 
| File | Description |
|------|-------------|
| `dinov2_finetuned_v2.pth` | DINOv2-ViT-S/14, noisy patches, cross-domain baseline |
| `dinov2_finetuned_v3.pth` | DINOv2-ViT-S/14, auto-filtered patches |
| `dinov2_finetuned_v4.pth` | DINOv2-ViT-S/14, manually curated terrain patches (primary Mars result) |
| `dinov2_earth_analog_v1.pth` | Earth analog fine-tune, initialized from v4 |
 
All models stored at `/content/drive/MyDrive/Spring25/`.
 
---
 
## Citation
 
If you use this work, please cite:
 
```
Hatfield, K. (2025). T(p)-Gated Sparse Landmark Mapping: Reliability-Aware
Cross-Domain Terrain Correspondence for Illumination-Invariant Planetary Navigation.
HEART AI Lab, Arizona State University.
```
 
---
 
## References
 
- Adams et al. (2025). Challenges and solutions for early morning TRN on Mars. AIAA SciTech. doi:10.2514/6.2025-2076
- Nash et al. (2024). Censible: A robust global localization framework for planetary surface missions. IEEE ICRA. doi:10.1109/ICRA57147.2024.10611697
- Oquab et al. (2024). DINOv2: Learning robust visual features without supervision. TMLR. arXiv:2304.07193
- Chen et al. (2020). A simple framework for contrastive learning of visual representations. ICML. arXiv:2002.05709
- Parker and Calef (2016). MSL Gale Merged DEM Mosaic 1m. USGS. doi:10.5066/F7X34VT7
- Kovesi (1999). Image features from phase congruency. Videre: J. Computer Vision Research, 1(3)
- Sanneman and Shah (2022). The SAFE-AI Framework for Explainable AI. Int. J. Human-Computer Interaction, 38(18-20). doi:10.1080/10447318.2022.2081282
