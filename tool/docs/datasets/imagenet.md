# C4 — ImageNet-1k (Classification)

**Status: trial addition (2026-08-28), proposed replacement candidate for
iWildCam.** iWildCam is NOT removed — see `context/idea.md` and
`context/DECISIONS_PENDING.md` DP18 for the decision record. This is the
fourth CV dataset, in addition to BDD100K, iWildCam, and ACDC.

## Purpose in the study

ImageNet-1k's practical advantage over iWildCam is that its 1000 classes
match torchvision's native pretrained classifier heads exactly — no
fine-tuning, no labeled training run, and no local weights_path checkpoint
need to be produced by hand before the dataset is runnable. It also has no
natural drift attribute on disk (unlike iWildCam's camera-trap `location`),
so a drift axis is engineered instead (see below), using the exact
corruption benchmark that TENT (Wang et al., ICLR 2021) is validated
against. This makes ImageNet the primary dataset for evaluating a
test-time-adaptation retrain strategy (Tent) rather than the primary dataset
for embedding-drift motivation — that role stays with iWildCam pending the
trial's outcome. Full rationale in `context/idea.md`.

## Task & models

**Domain:** cv / classification
**Task:** `classification`
**Task adapter:** `adapters.tasks.classification.ClassificationAdapter`
**num_classes:** `1000`
**Models:** EfficientNet-B0 (`efficientnet_b0`), ResNet-50 (`resnet50`),
  ResNet-101 (`resnet101`) — same three architectures as iWildCam, for
  cost_class parity (light/medium/heavy)
**Proxy:** `max_softmax`
**Drift detector:** `luminance_kl` (clean variant) / `mmd_embedding`
  (corrupted variant, `imagenet_c.json`) — see the important caveat below
**Embedding model:** `efficientnet_b0` (pinned per R4)

## Expected RAW form

Flat per-class directories, no train/val split on disk, no metadata file:

```
data/imagenet/<class_name>/<NNN>.jpg
```

`class_name` folders use human-readable words (e.g. `abacus`, `abaya`), not
synset IDs, and are NOT guaranteed to already be in torchvision's canonical
class-index order — `scripts/preprocess_imagenet.py` resolves this by
matching folder names against `torchvision.models.ResNet50_Weights
.IMAGENET1K_V2.meta["categories"]` at manifest-build time (no weight
download required for this — it is metadata on the Weights enum). See that
script's module docstring for the full correctness argument and the
loud-failure behavior when a folder name cannot be matched.

## Required PREPROCESSED form

`data/imagenet/imagenet_manifest.csv` — columns: `sample_id`, `input_path`,
`image_path` (duplicate of `input_path`; see the preprocessing script's
docstring for why), `label` (int, canonical class index), `class_name`,
`domain` (`"clean"` for every row in the base manifest), `split`
(`train`/`val`/`stream`).

`data/imagenet/class_index.json` — `index_to_class`, `class_to_index`,
`source` (`"torchvision_canonical"` or `"alphabetical_fallback"` — check
this before trusting any accuracy number computed against pretrained
weights).

## Drift-stream construction

The base manifest has a single domain (`"clean"`) — ImageNet has nothing
equivalent to BDD100K's weather attribute or iWildCam's location field.
`managed_system_cv/utility/drift/induce_imagenet_c.py` builds a genuinely
drift-ordered variant by applying simplified/approximate ImageNet-C-style
corruptions to a stratified subsample of the stream split:

```
clean → gaussian_noise → defocus_blur → fog → brightness_low → contrast_low
```

500 images per domain by default (matching the ~3 000-image CV target-size
convention used for BDD100K). Output: `data/imagenet_c/<domain>/*.jpg` +
`data/imagenet_c/imagenet_c_manifest.csv`, paired with
`configs/datasets/imagenet_c.json` (`train_frac=val_frac=0.0` — this
manifest is inference-only).

**Important caveat:** `configs/datasets/imagenet_c.json` declares
`drift_detector: "mmd_embedding"` as the intended signal, but as of
2026-08-28 `experiments/run_experiment.py`'s CV path hardcodes
`KLFixedRefDetector` on luminance values regardless of this config key (see
`context/DECISIONS_PENDING.md` DP20 — the same gap silently affects
iwildcam.json's `mmd_embedding` declaration too). Treat `drift_detector` in
both configs as a statement of intent until DP20 is resolved.

## Config skeleton

See `configs/datasets/imagenet.json` (clean) and `configs/datasets/imagenet_c.json`
(corruption-ordered).

## Init & calibration

```bash
cd tool/
python scripts/preprocess_imagenet.py --imagenet-root data/imagenet \
    --export-pretrained-weights          # no fine-tuning needed — see above
python managed_system_cv/utility/drift/induce_imagenet_c.py   # optional, for the drift variant
python scripts/init_cv.py --config imagenet --skip-embeddings
python scripts/validate_dataset.py configs/datasets/imagenet.json
python experiments/calibrate_drift_threshold.py --dataset imagenet_c   # DP4/Augur-style threshold calibration
```

## Offline-label availability

`label` is embedded directly in the manifest (matches iWildCam's convention,
not a separate label file) — read by `experiments/offline_eval.py` for
top-1 accuracy, never by the runtime MAPE-K loop (invariant I4).

## License/registration notes

Standard ImageNet-1k license terms apply to the underlying images (research,
non-commercial use; see image-net.org). No registration/download step is
documented here since the data is already present on disk for this project.
