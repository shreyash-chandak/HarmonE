# C2 — iWildCam (Classification)

## Purpose in the study

iWildCam (from the WILDS benchmark) provides wildlife camera-trap images from
182 species across geographically diverse locations. Geographic drift — images
from new camera-trap locations not seen at training time — is largely invisible
to luminance-histogram detectors (the animals and backgrounds vary semantically,
not photometrically). This makes iWildCam the primary motivation dataset for
the embedding-based drift detector (MMD/Fréchet): backbone embeddings capture
the semantic shift that KL on luminance misses.

## Task & models

**Domain:** cv / classification  
**Task:** `classification`  
**Task adapter:** `adapters.tasks.classification.ClassificationAdapter`  
**num_classes:** `182`  
**Models:** EfficientNet-B0 (`efficientnet_b0`), ResNet-50 (`resnet50`),
  ResNet-101 (`resnet101`)  
**Proxy:** `max_softmax` (max predicted softmax probability per image)  
**Drift detector:** `mmd_embedding` (primary; luminance_kl is a baseline)  
**Embedding model:** `efficientnet_b0` (pinned per R4, lightest model)

## Expected RAW form

Install via the WILDS package (https://wilds.stanford.edu/):

```bash
pip install wilds
python -c "
from wilds import get_dataset
ds = get_dataset('iwildcam', download=True, root_dir='data/')
"
```

This downloads (~150 GB) to `data/iwildcam_v2.0/`. The dataset provides:
- JPEG images in `data/iwildcam_v2.0/train/`
- Metadata CSV with `location`, `split`, `y` (species label integer) columns

## Required PREPROCESSED form

A manifest CSV with columns:

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | path relative to `data_root` |
| `label_path` | string | path to per-image label .txt |
| `location` | int | WILDS metadata location ID |
| `split` | string | `train`, `val`, `test`, `id_val`, `ood_val`, etc. |

**Label files:** one integer per file (the species class ID 0–181):
```
42
```

**Preprocessing recipe:**
```python
from wilds import get_dataset
import pandas as pd
from pathlib import Path

ds = get_dataset("iwildcam", root_dir="data/")
metadata = ds.metadata_array   # (N, ...) tensor
labels_all = ds.y_array        # (N,) tensor of ints
filenames = ds._input_array    # list of relative paths

rows = []
for i, fname in enumerate(filenames):
    img_path = f"iwildcam_v2.0/{fname}"
    label = int(labels_all[i])
    lbl_path = img_path.replace(".jpg", "_label.txt")
    Path("data/" + lbl_path).parent.mkdir(parents=True, exist_ok=True)
    Path("data/" + lbl_path).write_text(str(label))
    rows.append({
        "image_path": img_path,
        "label_path": lbl_path,
        "location": int(metadata[i, 0]),
        "split": ds.split_array[i],
    })

df = pd.DataFrame(rows)
# Order by location cluster to construct geographic drift stream
df = df.sort_values("location").reset_index(drop=True)
df.to_csv("data/iwildcam/iwildcam_manifest.csv", index=False)
```

## Drift-stream construction

Order manifest by `location` field (ascending location ID): images from
early locations → images from geographically distant locations. This creates
progressive geographic drift. The first N_loc locations form the training
distribution; later locations are the drift stream.

## Config skeleton

`configs/datasets/iwildcam.json` — `"status": "awaiting_data"` until downloaded.

Key fields:
```json
{
  "task": "classification",
  "num_classes": 182,
  "embedding_model": "efficientnet_b0",
  "embedding_dim": 1280,
  "drift_detector": "mmd_embedding"
}
```

## Init & calibration

```bash
cd tool/
python scripts/init_cv.py --config iwildcam  # seeds reference embeddings
python experiments/calibrate_energy_bounds.py --config iwildcam
python experiments/calibrate_drift_threshold.py --config iwildcam
```

## Offline label availability

Per-image label .txt files (generated during preprocessing above). Used ONLY
by `experiments/offline_eval.py` for proxy calibration; runtime is label-free (I4).

## License / registration

iWildCam is part of the WILDS benchmark (Koh et al., 2021).
License: CC BY-NC 4.0. Academic use permitted; commercial use prohibited.
Dataset download: https://wilds.stanford.edu/ (~150 GB).
Registration: not required, but WILDS terms-of-use apply.
