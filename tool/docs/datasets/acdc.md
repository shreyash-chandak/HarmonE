# C3 — ACDC (Segmentation)

## Purpose in the study

ACDC (Adverse Conditions Dataset with Correspondences) provides Cityscapes-format
semantic segmentation images across four adverse conditions: fog, night, rain,
snow — each with a clear-weather correspondence image. Ordering the manifest from
clear → fog → rain → night → snow constructs a monotonic drift stream aligned
with increasing adverse-condition severity. The condition-wise mIoU breakdown
enables exact per-condition accuracy tracking, validating that HarmonE's
segmentation adapter correctly degrades and recovers across conditions.

## Task & models

**Domain:** cv / segmentation  
**Task:** `segmentation`  
**Task adapter:** `adapters.tasks.segmentation.SegmentationAdapter`  
**num_classes:** `19` (Cityscapes label set)  
**ignore_index:** `255` (Cityscapes void class)  
**Models:** SegFormer-B0 (`segformer_b0`, baseline), SegFormer-B2 (`segformer_b2`)  
  DeepLab variants are P2 (deferred — see `DECISIONS_PENDING.md`).  
**Proxy:** `max_logit` (max pre-softmax logit averaged over image spatial dims)  
**Drift detector:** `luminance_kl` (primary; the clear→night transition is visible)  
**Embedding model:** `segformer_b0` (pinned per R4)

## Expected RAW form

Download from https://acdc.vision.ee.ethz.ch/ (free academic registration).

Archive structure:
```
rgb_anon/
  fog/  night/  rain/  snow/
    train/  val/  test/
      <sequence>/  *.png
gt_trainval/
  fog/  night/  rain/  snow/
    train/  val/
      <sequence>/  *_gt_labelTrainIds.png   ← Cityscapes trainIds
```

Cityscapes trainId 255 = void/unlabelled (ignored in mIoU per `ignore_index: 255`).
Labels are PNG images where each pixel is a class integer 0–18 or 255.

## Required PREPROCESSED form

A manifest CSV with columns:

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | path to RGB `.png` relative to `data_root` |
| `label_path` | string | path to `*_gt_labelTrainIds.png` relative to `data_root` |
| `condition` | string | `fog`, `night`, `rain`, `snow`, `clear` |
| `split` | string | `train` or `val` |

**Drift-stream construction recipe:**

Order by condition severity: `clear` → `fog` → `rain` → `night` → `snow`.
Within each condition, order by sequence name (chronological within sequence).

```python
import pandas as pd
from pathlib import Path

DATA_ROOT = Path("data/acdc")
_CONDITION_ORDER = {"clear": 0, "fog": 1, "rain": 2, "night": 3, "snow": 4}

rows = []
for condition in ["fog", "night", "rain", "snow"]:
    for split in ["train", "val"]:
        img_dir = DATA_ROOT / "rgb_anon" / condition / split
        lbl_dir = DATA_ROOT / "gt_trainval" / condition / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.rglob("*.png")):
            # Cityscapes-format label path: replace rgb_anon → gt_trainval, _rgb_anon → _gt_labelTrainIds
            lbl_name = img_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
            lbl_path = lbl_dir / img_path.parent.name / lbl_name
            rows.append({
                "image_path": str(img_path.relative_to(DATA_ROOT)),
                "label_path": str(lbl_path.relative_to(DATA_ROOT)) if lbl_path.exists() else "",
                "condition": condition,
                "split": split,
            })

df = pd.DataFrame(rows)
# Add a "clear" reference from the correspondence subset if available
# (ACDC provides clear-condition correspondences; re-label them as condition="clear")

df["_order"] = df["condition"].map(_CONDITION_ORDER)
df = df.sort_values(["_order", "split", "image_path"]).drop(columns=["_order"])
df.to_csv(DATA_ROOT / "acdc_manifest.csv", index=False)
```

## Config skeleton

`configs/datasets/acdc.json` — `"status": "awaiting_data"` until downloaded.

Key fields:
```json
{
  "task": "segmentation",
  "num_classes": 19,
  "ignore_index": 255,
  "embedding_model": "segformer_b0",
  "embedding_dim": 256,
  "drift_detector": "luminance_kl"
}
```

## Init & calibration

```bash
cd tool/
python scripts/init_cv.py --config acdc  # seeds reference embeddings
python experiments/calibrate_energy_bounds.py --config acdc
python experiments/calibrate_drift_threshold.py --config acdc
```

## Offline label availability

`*_gt_labelTrainIds.png` masks are available for per-image mIoU.
Used ONLY by `experiments/offline_eval.py` — runtime is label-free (I4).

The bilinear-upsample-before-argmax fix (R5-b) ensures correct mIoU when
SegFormer output resolution differs from mask resolution.

## License / registration

ACDC: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
Dataset site: https://acdc.vision.ee.ethz.ch/ — free academic registration required.
Cityscapes labels (inherited): https://www.cityscapes-dataset.com/license/
