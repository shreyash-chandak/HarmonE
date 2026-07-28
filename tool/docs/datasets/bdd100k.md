# C1 — BDD100K (Detection)

## Purpose in the study

BDD100K is the primary CV detection benchmark. Its rich metadata (weather,
time-of-day) enables a principled drift stream ordered from clear-day to
night/rain, validating that HarmonE's luminance-KL detector correctly fires
when the distribution shifts from clear to adverse conditions. The YOLOv8
model spectrum (n/s/m) exercises the energy-accuracy trade-off core to the
scoring formula.

## Task & models

**Domain:** cv / detection  
**Task:** `detection`  
**Task adapter:** `adapters.tasks.detection.DetectionAdapter`  
**Models:** YOLOv8n (`yolo_n`), YOLOv8s (`yolo_s`), YOLOv8m (`yolo_m`)  
**Proxy:** `confidence` (mean detection confidence per frame)  
**Drift detector:** `luminance_kl` (primary); `mmd_embedding` optional  
**Embedding model:** `yolo_n` (pinned per R4)

## Expected RAW form

Download from https://bdd-data.berkeley.edu/ (free academic registration required).

- `bdd100k_images_100k.zip` (~7.6 GB): images in `images/100k/train/`, `/val/`
- `bdd100k_labels_release.zip` (~300 MB): JSON annotation files in `labels/`

JSON structure per image:
```json
{
  "name": "00001234.jpg",
  "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "highway"},
  "labels": [{"category": "car", "box2d": {...}}, ...]
}
```

## Required PREPROCESSED form

**Images:** JPEG files in a single flat directory (or manifest CSV).

**Manifest CSV** (`bdd100k_manifest.csv`) with columns:

| Column | Type | Notes |
|--------|------|-------|
| `image_path` | string | path relative to `data_root` |
| `label_path` | string | path to per-image YOLO-format .txt (or empty) |
| `weather` | string | from JSON attributes |
| `timeofday` | string | from JSON attributes |

**YOLO label format** (one line per box):
```
<class_id> <x_center> <y_center> <width> <height>
```
Coordinates normalised to [0,1]. Use `scripts/bdd_to_yolo_labels.py` (to be
provided or written by the user) to convert BDD JSON → YOLO txt.

**Drift-stream construction recipe:**

Order the manifest chronologically by drift severity:
```
clear/daytime → overcast/daytime → dusk/dawn → nighttime → rainy
```

```python
import pandas as pd, json, pathlib

images_dir = pathlib.Path("data/bdd100k/images/100k/train")
labels_json = pathlib.Path("data/bdd100k/labels/bdd100k_labels_images_train.json")

with open(labels_json) as f:
    annotations = {a["name"]: a for a in json.load(f)}

_ORDER = {"clear": 0, "overcast": 1, "partly cloudy": 1,
          "foggy": 2, "rainy": 3, "snowy": 4, "undefined": 5}
_TOD_ORDER = {"daytime": 0, "dawn/dusk": 1, "night": 2, "undefined": 3}

rows = []
for img_path in sorted(images_dir.glob("*.jpg")):
    ann = annotations.get(img_path.name, {})
    attr = ann.get("attributes", {})
    rows.append({
        "image_path": str(img_path),
        "label_path": str(img_path).replace("images/100k/train", "labels/train")
                                   .replace(".jpg", ".txt"),
        "weather": attr.get("weather", "undefined"),
        "timeofday": attr.get("timeofday", "undefined"),
    })

df = pd.DataFrame(rows)
df["_w"] = df["weather"].map(_ORDER).fillna(5)
df["_t"] = df["timeofday"].map(_TOD_ORDER).fillna(3)
df = df.sort_values(["_w", "_t"]).drop(columns=["_w", "_t"])
df.to_csv("data/bdd100k/bdd100k_manifest.csv", index=False)
```

## Config skeleton

`configs/datasets/bdd100k.json` — contains actual production values.
Status is NOT `awaiting_data` since this is the primary config; set it once
images are downloaded.

## Init & calibration

```bash
cd tool/
# Download and preprocess (manual, ~8 GB download)
python scripts/bdd_to_yolo_labels.py  # converts JSON → YOLO .txt
python scripts/init_cv.py --config bdd100k  # seeds versionedMR + reference hist
# After model weights (yolo_n.pt etc.) are in managed_system_cv/models/:
python experiments/calibrate_energy_bounds.py --config bdd100k
python experiments/calibrate_drift_threshold.py --config bdd100k
```

## Offline label availability

YOLO .txt labels available for per-image mAP computation. Used ONLY by
`experiments/offline_eval.py` — the runtime inference loop is label-free (I4).

## License / registration

BDD100K: BSD 3-Clause License; academic registration required via
https://bdd-data.berkeley.edu/. Not for redistribution of raw images.
