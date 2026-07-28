# HarmonE Journal Extension — CV Integration Guide
> Complete context document for the computer vision extension of HarmonE.
> Written to prime a fresh chat with no prior context. Read every section
> before writing a single line of code.

---

## 1. Who Is Doing This and Why

**Researcher**: Shreyash, graduate researcher at SA4S Research Group, SERC, IIIT Hyderabad.
**Collaborator**: Shaunak Biswas, co-author of both HarmonE and Harmonica papers.
**Goal**: Journal extension of two published papers (HarmonE at ECSA 2025, Harmonica 2026),
targeting JSS as primary venue, TOSEM as fallback.
**Hardware**: Lenovo Legion 5 Gen 10, AMD Ryzen AI 7 350, NVIDIA RTX 5060 Laptop GPU (sm_120,
Blackwell architecture, very new — early 2025 release), 16GB RAM, running Linux.

The extension validates HarmonE across structurally diverse datasets and genuinely
new domains. The regression side is already running. This document covers only the
**CV side** of the extension.

---

## 2. The Two Papers — Minimum Context Needed

### HarmonE (ECSA 2025)
An architectural approach that wraps an MLOps pipeline in a MAPE-K loop to make it
self-adaptive with sustainability (accuracy + energy + cost + maintainability) as
explicit goals. Key mechanisms:

- **Decision Map**: design-time artifact where an architect defines sustainability goals
  as numeric thresholds
- **Monitor**: per interval, computes fused score S_i = β·A_i + (1−β)(1−Ē_i) where A_i
  is accuracy and Ē_i is normalised energy; smoothed via EMA
- **Analyzer**: compares EMA(S_i) to S_min (score violation), Ē_i to dynamic threshold
  τ_E (energy violation), KL divergence D to τ_drift (drift violation)
- **Planner**: on violation, selects among: switch to best-EMA alternative model
  (ε-greedy), reuse versioned model from VMR if distribution matches, retrain if
  no VMR match
- **Executor**: enacts the plan, versions retrained models into VMR with their training
  data distribution stored alongside
- **VMR (Versioned Model Repository)**: stores retrained models + the data distribution
  they were trained on; enables reuse when similar conditions recur

Validated on: single PeMS traffic sensor, LR/SVM/LSTM model spectrum, synthetic
scale-and-shift drift injected twice (so VMR reuse fires). Result: 95% of LSTM+PRT
accuracy at 45.5% of its energy.

### Harmonica (2026)
A reusable tool (Flask backend + web frontend) that wraps the HarmonE approach in a
configurable testbed. Two use cases demonstrated: (1) PeMS regression with full
HarmonE running, (2) BDD100K CV with YOLOv8 n/s/m — but **HarmonE did NOT run on
CV**. Only static YOLO baselines and a naive switch heuristic ran. The paper states
explicitly: "the original HarmonE approach was not generalised for this domain."

The tool repo: https://github.com/sa4s-serc/HarmonE-tool

---

## 3. Why HarmonE Cannot Run on CV Without Modification

This is the scientific core of the CV extension. There are three separate fundamental
problems, each requiring a distinct solution.

### 3.1 Problem 1 — A_i Does Not Exist at Runtime (The Core Problem)

HarmonE works because in autoregressive forecasting, the label for every prediction
arrives automatically. Predict flow at t+1; five minutes later the sensor reports
the actual flow at t+1. The environment auto-labels itself. A_i = R² computed against
reality, continuously, for free.

In CV object detection/classification/segmentation, the model makes a prediction on
a frame. Nobody delivers ground-truth annotations 5 minutes later. There is no
runtime A_i. Therefore:
- S_i = β·A_i + (1−β)(1−Ē_i) is uncomputable (half the score is undefined)
- EMA(S_i) < S_min cannot trigger correctly (it is based on a fabricated signal)
- Planner model ranking by historical EMA is ranking on garbage

The existing Harmonica CV code substitutes `mean detection confidence` for A_i. This
is problematic: under distribution shift, models stay confidently wrong — confidence
can plateau while true accuracy has collapsed, which means the signal you use to
detect degradation is least reliable exactly when you need it most.

**The solution**: a proxy validation experiment (described in Section 6) followed by
one of three proxy implementations depending on the result.

### 3.2 Problem 2 — Drift Detection Assumes Low-Dimensional Scalars

The regression monitor computes KL divergence between histograms of recent true
scalar flow values vs a reference distribution. This works because traffic flow is
one number per timestep and value distribution == semantics.

CV images are ~150k-dimensional pixel tensors. KL on pixel histograms is meaningless
in high dimensions. The existing Harmonica CV code uses **luminance histograms**
(grayscale intensity) — this detects brightness shifts and nothing else. It correctly
fires on their synthetic day/night drift (literally a brightness change) and is
completely blind to:
- Viewpoint/scale drift (VisDrone altitude changes — brightness barely changes but
  objects shrink to a few pixels)
- Geographic domain drift (iWildCam location changes — different background
  statistics, not just brightness)
- Semantic class distribution drift (new object categories appearing)

**The solution**: embedding-space drift detection using backbone feature vectors
rather than raw pixels, with MMD (Maximum Mean Discrepancy) or Fréchet distance as
the metric.

### 3.3 Problem 3 — VMR Matching and Retraining Assume Oracle Access

**VMR matching** in regression: each retrained model stored with a histogram of the
scalar values it was trained on. Matching current distribution to VMR is a histogram
comparison. Cheap and exact.

In the existing Harmonica CV code: VMR stores models alongside `average_histogram`
of luminance of training images. Matching is also luminance-based — same blindness
problem as drift detection. Two models trained under identical lighting but different
scene semantics are indistinguishable.

**Retraining** in the existing Harmonica CV code: `retrain.py` reads ground-truth
YOLO label files from disk (`data/bdd100k/labels/test/`). The augmentation factors
are hardcoded to match the synthetic drift inducer (`LUMINANCE_FACTOR = 0.35`,
`CONTRAST_FACTOR = 0.25`). This only works because the evaluation is oracle-coupled —
it has access to labels and drift parameters that a real deployed system would never
have.

**The solution**:
- VMR matching: embedding-space comparison (same feature space as drift detection)
- Retraining: pseudo-label fine-tuning using high-confidence predictions, or
  embedding-space contrastive fine-tuning, or human-in-the-loop flag

### 3.4 The Two-Axis Framing (Paper Narrative)

The reason HarmonE works for regression but not CV is not "time series vs images."
The real axes are:

| Axis | Regression (traffic) | CV (detection/classification) |
|---|---|---|
| Runtime ground truth | Available (auto-label) | Unavailable (label-free) |
| Data comparability | Direct (1-D scalar) | Requires embedding space |

Traffic forecasting is easy on BOTH axes. CV is hard on both. This is the paper's
contribution claim: generalising HarmonE to label-free, high-dimensional domains.

---

## 4. CV Dataset Choices

Three datasets chosen to cover three different task types and three maximally
distinct drift characters.

### C1 — BDD100K (Object Detection, Photometric/Weather Drift)

**Task**: Detect and localise vehicles, pedestrians, traffic lights, cyclists in
dashcam frames. Model outputs bounding boxes with class labels and confidence scores.

**Dataset**: Berkeley DeepDrive 100K. Dashcam footage from diverse conditions.

**Drift construction**: Use BDD100K's built-in attribute metadata to construct a
natural temporal sequence: clear-day → overcast → dusk → night → rain. These are
real images, not synthetic corruptions. Attribute metadata available in the official
annotation JSON files.

**Model spectrum**: YOLOv8n / YOLOv8s / YOLOv8m (continuity with Harmonica, ordered
by accuracy and energy).

**Proxy for A_i**: Mean detection confidence across boxes in each frame, averaged
over the monitoring interval. This is what Harmonica already uses. Whether it is
reliable is exactly what the proxy validation experiment (Section 6) tests.

**Why this dataset**: Continuity with Harmonica (Shaunak already set it up). Most
natural fit for the original HarmonE framing (deployed perception system in
intelligent transportation).

**Drift character**: Photometric — brightness, contrast, colour temperature shift
with weather and time of day. Detectable by luminance histograms, but the extension
replaces these with embedding-space detection anyway.

**Ground truth**: Full bounding box annotations available in BDD100K annotation
files. Used OFFLINE ONLY for proxy validation. Never seen by the runtime loop.

**Download**: https://bdd-data.berkeley.edu/ (requires account registration)

### C2 — iWildCam (Classification, Geographic Domain Drift)

**Task**: Classify animal species from camera trap images. Model outputs a class
label (species) and a softmax probability distribution. No bounding boxes — the
full image is classified.

**Dataset**: iWildCam from the WILDS benchmark. Camera trap images from hundreds of
geographic locations worldwide. Standard WILDS split provides in-distribution
(training locations) and out-of-distribution (new locations) evaluation sets.

**Drift construction**: Order inference stream by camera trap location. Moving from
one location cluster to another creates genuine geographic domain shift — background
statistics, vegetation, lighting distribution, and species prevalence all change.
Use the WILDS metadata to construct the ordered stream.

**Model spectrum**: EfficientNet-B0 / ResNet-50 / ResNet-101, ordered by accuracy
and energy. These are classification backbones, different from the YOLO family used
for BDD100K. This is intentional — different task type, different model family.

**Proxy for A_i**: Max softmax probability (top-1 confidence) per image, averaged
over the monitoring interval.

**Why this dataset**: Geographic domain shift is fundamentally different from weather
shift. Luminance histograms are completely blind to it (background texture and
species composition change, not image brightness). This motivates embedding-space
drift detection most strongly. Also part of the WILDS benchmark which has a
standardised evaluation protocol that reviewers trust.

**Drift character**: Semantic/geographic — distribution of visual contexts and
species changes by location. Not detectable by pixel-level methods.

**Ground truth**: Species labels available in WILDS annotation files. Used OFFLINE
ONLY for proxy validation.

**Download**: https://wilds.stanford.edu/datasets/ (free, no registration required)

### C3 — ACDC (Segmentation, Adverse Condition Visibility Drift)

**Task**: Pixel-level semantic segmentation of driving scenes into categories
(road, sidewalk, building, vegetation, vehicle, person, etc.). Model assigns a class
to every pixel.

**Dataset**: Adverse Conditions Dataset with Correspondences (ACDC). Uses Cityscapes
as the normal-condition reference (clear daytime driving in European cities) and
provides the SAME scenes photographed under fog, night, rain, and snow with full
pixel-level segmentation annotations.

**Drift construction**: Stream ordered as clear → fog → rain → night → snow.
Each condition is a genuine adverse deployment scenario. The correspondence structure
(same geographic location under multiple conditions) enables precise measurement of
how much accuracy drops under each condition.

**Model spectrum**: DeepLabV3+MobileNetV2 / DeepLabV3+ResNet50 / SegFormer-B0,
ordered by accuracy and energy. Segmentation models are much larger and slower
than classification models — this naturally creates a wider accuracy-energy tradeoff
spectrum than BDD100K or iWildCam, which makes the Pareto planner more interesting
to evaluate here.

**Proxy for A_i**: Mean pixel-level confidence (max softmax probability per pixel,
averaged over all pixels in the frame, averaged over the monitoring interval).
For segmentation this is meaningful because a well-calibrated segmentation model
assigns high confidence to pixels it labels correctly under normal conditions and
lower confidence under degraded visibility.

**Why this dataset**: Genuine segmentation task (pixel-level output, not image-level
or box-level). Adverse condition drift is a distinct third type alongside weather
shift (BDD100K) and geographic shift (iWildCam). ACDC has excellent annotation
quality and is purpose-built for this evaluation scenario. The correspondence
structure means you can compute true mIoU offline precisely.

**Drift character**: Visibility/atmospheric — fog reduces contrast globally, night
removes colour information, rain adds motion blur and reflection artefacts, snow
changes scene appearance dramatically. All detectable in embedding space.

**Ground truth**: Full pixel-level segmentation masks available. Used OFFLINE ONLY
for proxy validation (computing true mIoU vs runtime confidence proxy).

**Download**: https://acdc.vision.ee.ethz.ch/ (requires registration, free for research)

---

## 5. The Adapter Architecture (Shaunak's Requirement)

Shaunak explicitly asked for a unified config-driven system rather than three
parallel hardcoded managed systems. His exact words: "Try to still have a unified
system for these, with configs, if it isn't too coupled that way."

### 5.1 What Not To Do

Do not create:
```
managed_system_cv_detection/
managed_system_cv_classification/
managed_system_cv_segmentation/
```

Three parallel directories with duplicated MAPE-K logic that only differs in how
the model output is read. Bug fixes would need to be applied three times.

### 5.2 The Correct Structure

```
managed_system_cv/
    inference.py              # task-agnostic inference loop
    mape_logic/
        monitor.py            # reads proxy via adapter, not hardcoded
        analyse.py            # identical for all tasks
        plan.py               # identical for all tasks
        execute.py            # identical for all tasks
    adapters/
        base.py               # CVAdapter interface
        detection.py          # BDD100K / VisDrone
        classification.py     # iWildCam / CIFAR-10-C
        segmentation.py       # ACDC / EuroSAT
    configs/
        bdd100k.json
        iwildcam.json
        acdc.json
    knowledge/                # per-run state (cleared between runs)
        model.csv
        mape_info.json
        thresholds.json
        predictions.csv
        reference_distribution.json
        command.txt
```

### 5.3 The Config Schema

Each dataset config supplies everything task-specific:

```json
{
    "task": "detection",
    "dataset": "bdd100k",
    "adapter": "detection",
    "model_spectrum": ["yolo_n", "yolo_s", "yolo_m"],
    "model_paths": {
        "yolo_n": "models/yolov8n.pt",
        "yolo_s": "models/yolov8s.pt",
        "yolo_m": "models/yolov8m.pt"
    },
    "data_path": "data/bdd100k/images/",
    "offline_labels_path": "data/bdd100k/labels/",
    "proxy_metric": "mean_confidence",
    "drift_detector": "embedding_kl",
    "embedding_layer": "backbone_final",
    "thresholds": {
        "min_score": 0.54,
        "max_energy": 0.43,
        "beta": 0.96,
        "gamma": 0.8,
        "alpha": 0.15,
        "tau_drift": 0.07,
        "E_ref": 0.4,
        "delta": 0.1,
        "E_m": 0,
        "E_M": 10000000
    }
}
```

The thresholds are dataset-specific and must be calibrated per dataset via a pilot
run (see Section 8.4). Do not copy thresholds from one config to another.

### 5.4 The Adapter Interface

```python
# managed_system_cv/adapters/base.py

class CVAdapter:
    """
    Base adapter interface. All CV tasks implement these four methods.
    inference.py and monitor.py call ONLY these methods — never task-specific code.
    """

    def load_model(self, model_path: str) -> None:
        """Load model weights from path into self.model."""
        raise NotImplementedError

    def run_inference(self, input_path: str) -> object:
        """
        Run inference on a single input (image path or array).
        Returns a raw result object — adapter-specific, opaque to caller.
        """
        raise NotImplementedError

    def extract_proxy(self, result: object) -> float:
        """
        Extract the runtime accuracy proxy A_i from a raw result.
        Must return a float in [0, 1].
        Returns 0.0 if result is None or contains no valid outputs.
        """
        raise NotImplementedError

    def extract_embedding(self, input_path: str) -> "np.ndarray":
        """
        Extract a feature embedding vector from the input for drift detection.
        Should use an intermediate layer of the model, not the output.
        Returns a 1-D numpy array.
        """
        raise NotImplementedError

    def compute_offline_accuracy(
        self, result: object, label_path: str
    ) -> float:
        """
        Compute true accuracy metric against offline ground-truth labels.
        Used ONLY in the proxy validation experiment, never at runtime.
        Returns the task-appropriate metric:
          - detection: mAP@0.5
          - classification: top-1 accuracy
          - segmentation: mIoU
        """
        raise NotImplementedError
```

### 5.5 Detection Adapter (BDD100K)

```python
# managed_system_cv/adapters/detection.py

import numpy as np
from .base import CVAdapter

class DetectionAdapter(CVAdapter):

    def __init__(self):
        self.model = None
        self._embedding_hook = None
        self._embedding_buffer = []

    def load_model(self, model_path: str) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        # Register hook on backbone final layer for embedding extraction
        self._register_embedding_hook()

    def _register_embedding_hook(self):
        """Hook into backbone final feature map for embedding extraction."""
        def hook_fn(module, input, output):
            # Global average pool spatial dimensions to get 1-D vector
            self._embedding_buffer = [output.mean(dim=[2, 3]).squeeze().cpu().numpy()]
        # Hook the last backbone layer before the detection head
        backbone = self.model.model.model[9]  # layer 9 = backbone output in YOLOv8
        self._embedding_hook = backbone.register_forward_hook(hook_fn)

    def run_inference(self, input_path: str) -> object:
        if self.model is None:
            return None
        results = self.model(input_path, verbose=False)
        return results[0] if results else None

    def extract_proxy(self, result: object) -> float:
        if result is None:
            return 0.0
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return 0.0
        return float(boxes.conf.mean().item())

    def extract_embedding(self, input_path: str) -> np.ndarray:
        """Run a forward pass and capture the backbone embedding via hook."""
        self._embedding_buffer = []
        _ = self.model(input_path, verbose=False)
        if self._embedding_buffer:
            return self._embedding_buffer[0]
        return np.zeros(512)  # fallback

    def compute_offline_accuracy(self, result, label_path: str) -> float:
        """Compute mAP@0.5 against YOLO-format label file."""
        if result is None or not label_path:
            return 0.0
        # Use ultralytics built-in metric if available, otherwise manual IoU
        try:
            metrics = result.speed  # placeholder — implement proper mAP
            # Full implementation: parse label_path, compute IoU per box,
            # compute AP per class, mean across classes
            return 0.0  # TODO: implement
        except Exception:
            return 0.0
```

### 5.6 Classification Adapter (iWildCam)

```python
# managed_system_cv/adapters/classification.py

import numpy as np
import torch
from .base import CVAdapter

class ClassificationAdapter(CVAdapter):

    def __init__(self):
        self.model = None
        self.transform = None
        self._embedding = None

    def load_model(self, model_path: str) -> None:
        import torchvision.models as models
        from torchvision import transforms

        # Determine architecture from path name
        if "efficientnet_b0" in model_path:
            self.model = models.efficientnet_b0(pretrained=False)
        elif "resnet50" in model_path:
            self.model = models.resnet50(pretrained=False)
        elif "resnet101" in model_path:
            self.model = models.resnet101(pretrained=False)

        # Load weights
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # Register hook on penultimate layer for embeddings
        self._register_embedding_hook()

    def _register_embedding_hook(self):
        self._embedding = None
        def hook_fn(module, input, output):
            self._embedding = output.squeeze().detach().cpu().numpy()
        # Penultimate layer — works for ResNet and EfficientNet
        penultimate = list(self.model.children())[-2]
        penultimate.register_forward_hook(hook_fn)

    def run_inference(self, input_path: str) -> object:
        from PIL import Image
        img = Image.open(input_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            logits = self.model(x)
        return logits

    def extract_proxy(self, result: object) -> float:
        if result is None:
            return 0.0
        probs = torch.softmax(result, dim=-1)
        return float(probs.max().item())

    def extract_embedding(self, input_path: str) -> np.ndarray:
        _ = self.run_inference(input_path)
        if self._embedding is not None:
            return self._embedding
        return np.zeros(2048)

    def compute_offline_accuracy(self, result, label_path: str) -> float:
        """Top-1 accuracy: check if argmax matches ground truth label."""
        if result is None or not label_path:
            return 0.0
        with open(label_path) as f:
            true_class = int(f.read().strip())
        pred_class = int(result.argmax(dim=-1).item())
        return 1.0 if pred_class == true_class else 0.0
```

### 5.7 Segmentation Adapter (ACDC)

```python
# managed_system_cv/adapters/segmentation.py

import numpy as np
import torch
from .base import CVAdapter

class SegmentationAdapter(CVAdapter):

    def __init__(self):
        self.model = None
        self._embedding = None

    def load_model(self, model_path: str) -> None:
        # SegFormer-B0 from HuggingFace transformers
        from transformers import SegformerForSemanticSegmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self._register_embedding_hook()

    def _register_embedding_hook(self):
        self._embedding = None
        def hook_fn(module, input, output):
            # Pool the encoder's final hidden state
            if hasattr(output, "last_hidden_state"):
                self._embedding = output.last_hidden_state.mean(dim=[1, 2, 3]).detach().cpu().numpy()
        self.model.segformer.encoder.register_forward_hook(hook_fn)

    def run_inference(self, input_path: str) -> object:
        from PIL import Image
        from transformers import SegformerImageProcessor
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-512")
        image = Image.open(input_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def extract_proxy(self, result: object) -> float:
        """Mean max-softmax confidence across all pixels."""
        if result is None:
            return 0.0
        logits = result.logits  # shape: (1, num_classes, H, W)
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1).values  # (1, H, W)
        return float(max_probs.mean().item())

    def extract_embedding(self, input_path: str) -> np.ndarray:
        _ = self.run_inference(input_path)
        if self._embedding is not None:
            return self._embedding
        return np.zeros(256)

    def compute_offline_accuracy(self, result, label_path: str) -> float:
        """Compute mean IoU against pixel-level segmentation mask."""
        if result is None or not label_path:
            return 0.0
        import numpy as np
        from PIL import Image

        # Upsample logits to original image size
        logits = result.logits
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()

        gt = np.array(Image.open(label_path))

        # Compute per-class IoU
        num_classes = logits.shape[1]
        iou_per_class = []
        for c in range(num_classes):
            pred_c = pred == c
            gt_c = gt == c
            intersection = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            if union > 0:
                iou_per_class.append(intersection / union)
        return float(np.mean(iou_per_class)) if iou_per_class else 0.0
```

---

## 6. The Proxy Validation Experiment

**This must run before any other CV experiment and before iWildCam or ACDC
integration begins.** The result determines which proxy implementation to use
for all three datasets.

### 6.1 What It Is

Run BDD100K inference through the detection adapter. At each monitoring interval,
compute two numbers:
1. **Runtime proxy**: mean detection confidence (what the MAPE-K loop sees)
2. **Offline truth**: true mAP@0.5 computed against BDD100K bounding box annotations

Plot both over time across the clear → overcast → night → rain drift sequence.
Compute Spearman correlation ρ between them.

### 6.2 What the Result Determines

**If ρ > 0.75 consistently across drift conditions**: confidence is a defensible
proxy. Use temperature scaling (Guo et al. 2017) to calibrate it — apply temperature
scaling on a held-out validation set per model, store the temperature T in the
knowledge base, divide logits by T before softmax. This is Option A.

**If ρ collapses under severe drift** (e.g. stays high while mAP falls): confidence
is unreliable. Use an agreement-based proxy — run two model-scale variants on the
same input; when outputs agree, proxy-accuracy is high; when they disagree, flag
degradation. More expensive but label-free and drift-robust. This is Option B.

**If neither works well**: use the detection-specific box count proxy — track the
mean number of boxes per image across a rolling window. Sudden changes in box count
indicate scene distribution change. This is Option C (weakest, use as secondary
signal only).

### 6.3 Implementation

```python
# experiments/proxy_validation.py

import csv
import json
import numpy as np
from scipy.stats import spearmanr
from managed_system_cv.adapters.detection import DetectionAdapter

def run_proxy_validation(
    image_dir: str,
    label_dir: str,
    model_path: str,
    output_csv: str,
    interval_size: int = 50
):
    """
    Run proxy validation experiment on BDD100K drift sequence.

    For each monitoring interval:
      - Computes mean detection confidence (runtime proxy)
      - Computes mean mAP@0.5 against offline labels (ground truth)

    Outputs a CSV with columns: interval, confidence_proxy, true_map, model
    """
    adapter = DetectionAdapter()
    adapter.load_model(model_path)

    image_files = sorted(os.listdir(image_dir))  # ordered by drift sequence
    results = []

    for i in range(0, len(image_files), interval_size):
        batch = image_files[i:i + interval_size]
        confidences = []
        maps = []

        for fname in batch:
            img_path = os.path.join(image_dir, fname)
            label_path = os.path.join(label_dir, fname.replace(".jpg", ".txt"))

            result = adapter.run_inference(img_path)
            conf = adapter.extract_proxy(result)
            map_score = adapter.compute_offline_accuracy(result, label_path)

            confidences.append(conf)
            maps.append(map_score)

        results.append({
            "interval": i // interval_size,
            "confidence_proxy": np.mean(confidences),
            "true_map": np.mean(maps),
            "model": model_path
        })

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Compute correlation
    proxies = [r["confidence_proxy"] for r in results]
    truths = [r["true_map"] for r in results]
    rho, pval = spearmanr(proxies, truths)
    print(f"Spearman ρ = {rho:.4f}, p = {pval:.4f}")
    print("Proxy is", "RELIABLE (>0.75)" if rho > 0.75 else "UNRELIABLE (<0.75)")

    return rho
```

Run separately for each YOLOv8 variant (n, s, m) and under each BDD100K split
(clear, overcast, night, rain). Report ρ per model per condition. This table alone
is publishable content that directly justifies the CV generalisation work.

---

## 7. Embedding-Space Drift Detection

### 7.1 Why Needed

Luminance histograms (current Harmonica implementation) detect brightness shifts
only. Three of the four drift types across the chosen datasets are invisible to them:
- iWildCam geographic shift: background texture changes, not brightness
- ACDC fog/night: partially visible (night reduces brightness) but fog and rain
  produce almost no luminance change while decimating model accuracy
- VisDrone viewpoint: zero luminance change as drone ascends

### 7.2 The MMD Approach

Maximum Mean Discrepancy (MMD) with RBF kernel computes the distance between two
sets of samples in a feature space without requiring density estimation. It works
reliably with moderate sample sizes (200–1000 embeddings) in moderate dimensions
(256–2048).

```python
# core/drift/embedding_mmd.py

import numpy as np

def rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float = None) -> np.ndarray:
    """RBF kernel matrix between rows of X and rows of Y."""
    if bandwidth is None:
        # Median heuristic: bandwidth = median of pairwise distances
        all_dists = np.linalg.norm(X[:, None] - Y[None, :], axis=-1)
        bandwidth = np.median(all_dists) + 1e-8
    dists_sq = np.sum((X[:, None] - Y[None, :]) ** 2, axis=-1)
    return np.exp(-dists_sq / (2 * bandwidth ** 2))

def mmd_squared(X: np.ndarray, Y: np.ndarray, bandwidth: float = None) -> float:
    """
    Unbiased estimator of MMD² between sample sets X and Y.
    X: (n, d) — reference window embeddings
    Y: (m, d) — current window embeddings
    Returns: float, MMD² value. Higher = more distribution shift.
    """
    n, m = len(X), len(Y)
    Kxx = rbf_kernel(X, X, bandwidth)
    Kyy = rbf_kernel(Y, Y, bandwidth)
    Kxy = rbf_kernel(X, Y, bandwidth)

    # Unbiased estimator: exclude diagonal terms
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) +
        (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) -
        2 * Kxy.mean()
    )
    return float(max(mmd2, 0.0))  # clamp numerical negatives to 0
```

### 7.3 Integration Into Monitor

Replace `kl_divergence` calls in `managed_system_cv/mape_logic/monitor.py`:

```python
# In monitor_drift():

def monitor_drift(self, adapter, config):
    """
    Compute embedding-space drift using MMD between reference and current windows.
    Falls back to luminance KL if embedding extraction fails.
    """
    from core.drift.embedding_mmd import mmd_squared

    window_size = config.get("drift_window_size", 500)

    # Load recent embeddings from predictions.csv
    df = pd.read_csv(PREDICTIONS_FILE)
    if len(df) < window_size * 2:
        return {"kl_div": None, "mmd": None}

    # Embeddings stored as JSON strings in predictions.csv
    ref_embeddings = np.array([
        json.loads(e) for e in df["embedding"].iloc[-2*window_size:-window_size]
    ])
    cur_embeddings = np.array([
        json.loads(e) for e in df["embedding"].iloc[-window_size:]
    ])

    mmd = mmd_squared(ref_embeddings, cur_embeddings)

    # Also compute luminance KL as secondary signal for comparison
    kl = _compute_luminance_kl(df, window_size)  # keep old method as secondary

    return {
        "mmd": round(mmd, 6),           # primary drift signal
        "kl_div": kl,                    # secondary, kept for comparison plots
        "drift_detector": "embedding_mmd"
    }
```

### 7.4 Storing Embeddings in predictions.csv

`inference.py` must extract and log an embedding alongside each prediction:

```python
# In inference loop, per frame:
result = adapter.run_inference(image_path)
proxy = adapter.extract_proxy(result)
embedding = adapter.extract_embedding(image_path)

row = {
    "image": image_path,
    "confidence": proxy,
    "energy_uJ": energy_measurement,
    "embedding": json.dumps(embedding.tolist()),  # stored as JSON string
    "timestamp": time.time()
}
```

Note: embeddings can be large (512–2048 floats per row). For 10,000 frames at
1024-dim embeddings this is ~80MB in predictions.csv. Use float16 serialisation
if file size becomes a problem:
```python
"embedding": json.dumps(embedding.astype(np.float16).tolist())
```

### 7.5 VMR Matching in Embedding Space

Replace luminance histogram VMR matching with embedding-space Fréchet distance:

```python
# In analyse_drift(), when searching VMR for a matching model:

def _embedding_distance(current_embeddings, vmr_embeddings):
    """
    Fréchet distance between two sets of embeddings.
    Faster than MMD for VMR lookup, sufficient for retrieval.
    """
    mu1, sigma1 = current_embeddings.mean(0), np.cov(current_embeddings.T)
    mu2, sigma2 = vmr_embeddings.mean(0), np.cov(vmr_embeddings.T)

    diff = mu1 - mu2
    # Matrix square root via eigendecomposition
    vals, vecs = np.linalg.eigh(sigma1 @ sigma2)
    sqrt_product = vecs @ np.diag(np.sqrt(np.abs(vals))) @ vecs.T

    return float(np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * sqrt_product))
```

Store per-version embedding statistics (mean + covariance of training set embeddings)
in the VMR alongside model weights:
```
versionedMR/
    yolo_s_v1.pt
    yolo_s_v1_embeddings.npz   # np.savez: mean, covariance, n_samples
    yolo_s_v1_hist.json        # keep old luminance hist for backward compat
```

---

## 8. Energy Measurement for CV

### 8.1 The Problem

The current codebase uses pyRAPL throughout. pyRAPL reads Intel RAPL registers.
The machine has an AMD CPU and an NVIDIA RTX 5060 GPU. Two separate problems:

**AMD CPU**: pyRAPL may return zero on AMD. Test with:
```bash
sudo modprobe msr
python3 -c "
import pyRAPL; pyRAPL.setup()
m = pyRAPL.Measurement('test'); m.begin()
import numpy as np
for _ in range(500000): np.dot(np.random.rand(100), np.random.rand(100))
m.end()
print('pkg:', m.result.pkg)
"
```
If pkg is zero or None, AMD RAPL is inaccessible via pyRAPL on this chip.

**RTX 5060 GPU**: pyJoules's NVML backend calls
`nvmlDeviceGetTotalEnergyConsumption()`. This function is not implemented on all
consumer GPUs. The RTX 5060 (sm_120, Blackwell, new in 2025) may return
`NVML_ERROR_NOT_SUPPORTED`. However, `nvmlDeviceGetPowerUsage()` (instantaneous
power draw) IS available — nvidia-smi confirmed 10.41W average, 11.74W instantaneous.

### 8.2 The PyTorch CUDA Problem (Blocks All CV)

PyTorch 2.13.0+cu126 does not include kernels for sm_120 (RTX 5060). Fix:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

Verify:
```bash
python3 -c "
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_capability(0))  # should be (12, 0)
x = torch.tensor([1.0]).cuda()
print('GPU working:', x)
"
```

**This must be fixed before ANY CV experiment runs.** Until PyTorch is reinstalled,
YOLO crashes on warmup and all CV metrics are fabricated.

### 8.3 GPU Energy — Polling Meter Fallback

If pyJoules NVML fails (test with the probe in Section 8.4), use power.draw polling:

```python
# core/energy/gpu_polling.py

import subprocess
import threading
import time
import numpy as np

class NvidiaPowerPollingMeter:
    """
    Estimates GPU energy by integrating instantaneous power.draw at fixed intervals.
    Used when nvmlDeviceGetTotalEnergyConsumption() is unsupported (e.g. RTX 5060).

    Accuracy: At 50ms intervals, error per inference call is bounded at
    ±25ms × mean_power. For YOLO inference (~200ms, ~15W GPU draw) this is
    ±0.375 mJ per call, ~1-2% relative error. Acceptable for the paper.
    """

    def __init__(self, interval_ms: int = 50, gpu_index: int = 0):
        self.interval_s = interval_ms / 1000.0
        self.gpu_index = gpu_index
        self._readings_w: list = []
        self._running: bool = False
        self._thread = None
        self._lock = threading.Lock()
        self.joules: float = 0.0

    def start(self):
        self._readings_w = []
        self.joules = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits"
        ]
        while self._running:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=0.5)
                val = r.stdout.strip()
                if val and val.lower() not in ("[n/a]", ""):
                    with self._lock:
                        self._readings_w.append(float(val))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._lock:
            self.joules = sum(self._readings_w) * self.interval_s if self._readings_w else 0.0

    @property
    def microjoules(self) -> float:
        return self.joules * 1e6

    @property
    def mean_power_watts(self) -> float:
        with self._lock:
            return float(np.mean(self._readings_w)) if self._readings_w else 0.0

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._readings_w)
```

### 8.4 EnergyContext — Unified Abstraction

> **SUPERSEDED** — This design was not implemented. The actual implementation is
> `core/energy.py::EnergyMeter` with `_PyJoulesRaplBackend`, `_PyJoulesNvmlBackend`,
> and `_PollingGPUBackend` backends (see `files.md` and CHANGES_FROM_PAPER.md Phase 6).
> All new CV code uses `core/energy.py::EnergyMeter`. Do not create `core/energy/context.py`.

```python
# core/energy/context.py  [SUPERSEDED — for reference only, do not port]

import logging
logger = logging.getLogger(__name__)

_RAPL_OK = None
_NVML_OK = None

def _probe_rapl() -> bool:
    global _RAPL_OK
    if _RAPL_OK is not None:
        return _RAPL_OK
    try:
        from pyJoules.device.rapl_device import RaplPackageDomain
        from pyJoules.energy_meter import EnergyMeter
        import time
        m = EnergyMeter([RaplPackageDomain(0)])
        m.start(tag="probe"); time.sleep(0.05); m.stop()
        for s in m.get_trace():
            if s.energy and any(v > 0 for v in s.energy.values()):
                _RAPL_OK = True
                logger.info("RAPL available for CPU energy.")
                return True
        _RAPL_OK = False
        logger.warning("RAPL returned zero — CPU energy will be 0.0")
    except Exception as e:
        _RAPL_OK = False
        logger.warning(f"RAPL unavailable: {e} — CPU energy will be 0.0")
    return _RAPL_OK

def _probe_nvml() -> bool:
    global _NVML_OK
    if _NVML_OK is not None:
        return _NVML_OK
    try:
        from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
        from pyJoules.energy_meter import EnergyMeter
        import time
        m = EnergyMeter([NvidiaGPUDomain(0)])
        m.start(tag="probe"); time.sleep(0.2); m.stop()
        for s in m.get_trace():
            if s.energy and any(v > 0 for v in s.energy.values()):
                _NVML_OK = True
                logger.info("NVML available for GPU energy.")
                return True
        _NVML_OK = False
        logger.warning("NVML returned zero — falling back to power.draw polling")
    except Exception as e:
        _NVML_OK = False
        logger.warning(f"NVML unavailable: {e} — falling back to power.draw polling")
    return _NVML_OK


class EnergyContext:
    """
    Context manager for energy measurement.

    Usage:
        with EnergyContext("gpu") as m:
            result = model(frame)
        log_energy(m.total_joules)

    Args:
        domain: "cpu" | "gpu" | "both"
            "cpu"  — regression (RAPL only, no GPU)
            "gpu"  — CV inference (GPU energy only)
            "both" — CV retraining (GPU + CPU)
    """

    def __init__(self, domain: str = "cpu"):
        self.domain = domain
        self.cpu_joules = 0.0
        self.gpu_joules = 0.0
        self.total_joules = 0.0
        self._cpu_meter = None
        self._gpu_meter = None

        if domain in ("cpu", "both"):
            if _probe_rapl():
                from pyJoules.device.rapl_device import RaplPackageDomain
                from pyJoules.energy_meter import EnergyMeter
                self._cpu_meter = EnergyMeter([RaplPackageDomain(0)])
            # else: None meter, cpu_joules stays 0.0

        if domain in ("gpu", "both"):
            if _probe_nvml():
                from pyJoules.device.nvidia_gpu import NvidiaGPUDomain
                from pyJoules.energy_meter import EnergyMeter
                self._gpu_meter = ("nvml", EnergyMeter([NvidiaGPUDomain(0)]))
            else:
                from core.energy.gpu_polling import NvidiaPowerPollingMeter
                self._gpu_meter = ("polling", NvidiaPowerPollingMeter(interval_ms=50))

    def __enter__(self):
        if self._cpu_meter:
            self._cpu_meter.start(tag="measure")
        if self._gpu_meter:
            kind, meter = self._gpu_meter
            if kind == "nvml":
                meter.start(tag="measure")
            else:
                meter.start()
        return self

    def __exit__(self, *args):
        # Stop CPU
        if self._cpu_meter:
            self._cpu_meter.stop()
            for s in self._cpu_meter.get_trace():
                if s.energy:
                    self.cpu_joules = sum(s.energy.values()) / 1e6

        # Stop GPU
        if self._gpu_meter:
            kind, meter = self._gpu_meter
            meter.stop()
            if kind == "nvml":
                for s in meter.get_trace():
                    if s.energy:
                        self.gpu_joules = sum(s.energy.values()) / 1e6
            else:
                self.gpu_joules = meter.joules

        self.total_joules = self.cpu_joules + self.gpu_joules

    def to_dict(self) -> dict:
        return {
            "cpu_joules": self.cpu_joules,
            "gpu_joules": self.gpu_joules,
            "total_joules": self.total_joules,
        }


def get_backend_status() -> dict:
    """Call at experiment start to log which backends are active."""
    return {
        "rapl": _probe_rapl(),
        "nvml": _probe_nvml(),
        "gpu_backend": "nvml" if _probe_nvml() else "polling_50ms",
        "cpu_backend": "rapl" if _probe_rapl() else "unavailable",
    }
```

CV `inference.py` uses `EnergyContext("gpu")` per frame.
CV `retrain.py` uses `EnergyContext("both")` for the full fine-tuning run.
Regression `execute.py` uses `EnergyContext("cpu")` — unchanged behaviour.

---

## 9. Retraining Without Oracle Labels

### 9.1 Current State (Oracle-Coupled, Broken for Real Deployment)

`managed_system_cv/retrain.py` reads ground-truth YOLO labels from disk and
uses hardcoded augmentation factors matching the synthetic drift inducer. This
is not a real deployment scenario — it only works because the evaluation is rigged.

### 9.2 Three Retraining Tactics (Planner Selects Based on Config)

**Tactic RT1 — Pseudo-Label Fine-Tuning** (default, label-free):
Use the current model's own high-confidence predictions as training signal.
Only predictions where confidence > τ_pseudo (default 0.85) are retained as
pseudo-labels. Fine-tune for a small number of epochs (3–5) on these pseudo-labeled
frames. Bias risk: if the model is wrong confidently, it reinforces its own errors.
Mitigation: if fine-tuned model's proxy score is lower than before fine-tuning,
roll back to pre-retrain weights.

```python
def retrain_pseudo_labels(adapter, recent_frames, tau_pseudo=0.85, epochs=3):
    pseudo_labeled = []
    for frame_path in recent_frames:
        result = adapter.run_inference(frame_path)
        proxy = adapter.extract_proxy(result)
        if proxy >= tau_pseudo:
            pseudo_labeled.append((frame_path, result))

    if len(pseudo_labeled) < 50:
        logger.warning(f"Only {len(pseudo_labeled)} high-confidence frames — skipping retrain")
        return False

    # Fine-tune on pseudo-labeled set
    # ... task-specific fine-tuning call ...
    return True
```

**Tactic RT2 — Embedding-Space Contrastive Fine-Tuning** (label-free, no task labels):
Fine-tune only the feature extractor (backbone), not the task head, using
contrastive learning between embeddings from the reference distribution and the
current distribution. Pulls the current distribution's representations toward the
reference space. Does not require task-level labels.

**Tactic RT3 — Human-in-the-Loop Flag** (honest fallback):
When neither RT1 nor RT2 is configured or available, log that manual annotation
is needed, temporarily switch to the most energy-efficient model (as a conservative
fallback), and set a flag in knowledge that a human label batch is required.
This is the honest option — document in the paper that some drift scenarios
genuinely require human intervention.

### 9.3 Tactic Selection in Planner

```json
// In dataset config:
{
    "retrain_tactic": "pseudo_labels",  // or "contrastive" or "human_in_loop"
    "pseudo_label_threshold": 0.85,
    "max_retrain_frames": 500,
    "rollback_if_worse": true
}
```

For the paper's ablation: run each dataset with all three retraining tactics
and compare. For datasets with offline labels (all three chosen datasets have them),
also run a fourth oracle tactic (actual labels) as an upper bound.

---

## 10. Experiment Design for CV

### 10.1 Baseline Grid (Per Dataset)

| Baseline | What It Isolates |
|---|---|
| Static YOLOv8n / ResNet50 / SegFormer-B0 | No adaptation lower bound |
| Static YOLOv8m / ResNet101 / SegFormer-B2 | Always-heavy upper energy bound |
| Static + PRT every N frames | Cost of unconditional retraining |
| Random switch (current Harmonica "Switch") | Dumb adaptation |
| Greedy switch (EMA-based, no retrain) | Informed adaptation without memory |
| Oracle switch (omniscient, offline labels) | Upper bound |
| HarmonE-original (luminance KL, confidence proxy) | Reproduce Harmonica result |
| HarmonE-embedding (MMD drift, calibrated proxy) | Proposed CV generalisation |
| HarmonE-Pareto (Pareto-aware planner) | Novel planner contribution |

### 10.2 Metrics Per Dataset

- **Accuracy proxy**: mean confidence per interval (runtime, what the loop sees)
- **True accuracy**: mAP@0.5 (detection), top-1 accuracy (classification),
  mIoU (segmentation) — offline, computed against labels, NOT used at runtime
- **Proxy correlation**: Spearman ρ between proxy and true accuracy over time
- **GPU energy** (joules, from EnergyContext)
- **CPU energy** (joules, from EnergyContext, may be 0 on AMD)
- **Total energy** (sum)
- **Inference latency** (ms per frame, excluding retrain)
- **Adaptation counts**: switches (S), retrains (R), VMR events (V) separately
- **MAPE-K overhead energy** (joules for the control loop itself)

### 10.3 Protocol

- 5 independent runs per approach per dataset
- 20-minute cooldown between runs (hardware thermals)
- Knowledge base fully reset between runs (command.txt cleared, mape_info.json
  zeroed, predictions.csv header only, EMA scores reset to 0.5)
- Chronological data ordering — no shuffling
- Pilot run before main grid to calibrate τ_drift, min_score, energy thresholds
  per dataset (see Section 10.4)

### 10.4 Threshold Calibration Per Dataset

Do not reuse thresholds across datasets. For each dataset:

1. Run the static heavy model (YOLOv8m / ResNet101 / SegFormer) on the
   in-distribution portion of the data (first 20% chronologically)
2. Compute the distribution of EMA(S_i) scores — set S_min at the 10th percentile
3. Compute the distribution of MMD values on in-distribution adjacent windows —
   set τ_drift at the 99th percentile of the null distribution
4. Compute E_m and E_M from the observed energy range per model
5. Set E_ref at the 70th percentile of in-distribution energy

Store calibrated thresholds in `configs/<dataset>.json` under the `thresholds` key.
Document all calibration choices — reviewers will ask.

---

## 11. Current Blockers (As of July 25 2026)

In priority order. Nothing can start until each prior blocker is resolved.

**Blocker 1 — PyTorch CUDA incompatibility (blocks all CV inference)**
PyTorch 2.13.0+cu126 does not support sm_120. Reinstall with cu129 or cu130.
Cannot test any CV code until this is fixed.

**Blocker 2 — pyJoules NVML validation (blocks energy abstraction)**
Cannot know which GPU energy backend to implement until NVML is tested against
the RTX 5060. Cannot test until Blocker 1 is fixed (environment is broken).

**Blocker 3 — Proxy validation experiment (blocks iWildCam and ACDC integration)**
Must run on BDD100K before deciding which proxy implementation to use.
Cannot run until Blocker 1 is fixed.

**Blocker 4 — Adapter architecture (can build in parallel with Blockers 1–3)**
`CVAdapter` base class and all three adapters need to exist before dataset
integration. This is the one thing that can be built while waiting for
PyTorch reinstall. Build the interface and stub all four methods; fill in
implementations once inference is running.

**Blocker 5 — Dataset downloads and preprocessing**
BDD100K requires account registration at bdd-data.berkeley.edu.
iWildCam available at wilds.stanford.edu/datasets/ without registration.
ACDC requires registration at acdc.vision.ee.ethz.ch.
Each needs a drift-ordered stream constructed from metadata (not just raw download).

---

## 12. For the Paper — CV Section Structure

### Framing (what to say in the introduction)

> "HarmonE's MAPE-K signals assume runtime ground truth; we generalise the
> approach to label-free domains by replacing those signals with validated
> proxies and embedding-space drift detection, and demonstrate the generalised
> framework across three CV task types (object detection, image classification,
> semantic segmentation) spanning five distinct drift characters."

### Research Questions (CV-specific)

- **RQ-CV1**: Does mean detection/classification/segmentation confidence provide
  a reliable proxy for runtime accuracy under distribution shift? Under which
  drift conditions does it fail?
- **RQ-CV2**: Does embedding-space drift detection (MMD) provide earlier and more
  semantically valid adaptation triggers than pixel-space detection (luminance KL)?
- **RQ-CV3**: Does generalised HarmonE maintain sustainability goals across all
  three CV task types and drift characters?
- **RQ-CV4**: How does the Pareto-aware planner compare to the original EMA-greedy
  planner on the accuracy-energy trade-off in CV settings?

### Threats (CV-specific additions)

**Proxy validity**: Runtime accuracy proxies (confidence scores) are unvalidated
assumptions in existing CV systems. We validate them empirically via offline label
comparison (proxy validation experiment) but acknowledge this validation is itself
conducted on the evaluation data rather than a separate held-out proxy calibration set.

**GPU energy measurement**: RTX 5060 (sm_120) does not expose cumulative energy
counters via NVML. GPU energy is estimated via 50ms power.draw polling with
bounded integration error of approximately ±2% over full experimental runs.

**Retraining without labels**: The pseudo-label retraining tactic introduces
confirmation bias risk. We mitigate via rollback gating but cannot eliminate it.
Results for the retrain arm of the tactics table should be interpreted as
optimistic upper bounds on real-deployment retraining performance.

**Dataset selection**: Three CV datasets were selected to span task types and drift
characters. Results may not generalise to medical imaging, industrial inspection,
or other domains with different calibration characteristics.

---

## 13. Quick Reference — File Locations

```
tool/
├── core/
│   ├── energy/
│   │   ├── context.py          # EnergyContext (unified abstraction)
│   │   └── gpu_polling.py      # NvidiaPowerPollingMeter
│   └── drift/
│       └── embedding_mmd.py    # mmd_squared, rbf_kernel
├── managed_system_cv/
│   ├── inference.py            # task-agnostic, loads adapter from config
│   ├── mape_logic/
│   │   ├── monitor.py          # calls adapter.extract_proxy, embedding_mmd
│   │   ├── analyse.py          # threshold comparison, unchanged logic
│   │   ├── plan.py             # all planner strategies (S1-S6)
│   │   └── execute.py          # calls adapter methods, EnergyContext("gpu")
│   ├── adapters/
│   │   ├── base.py             # CVAdapter interface
│   │   ├── detection.py        # BDD100K / VisDrone
│   │   ├── classification.py   # iWildCam / CIFAR-10-C
│   │   └── segmentation.py     # ACDC / EuroSAT
│   └── configs/
│       ├── bdd100k.json
│       ├── iwildcam.json
│       └── acdc.json
└── experiments/
    └── proxy_validation.py     # MUST RUN FIRST on BDD100K
```

---

*Document generated July 25 2026. Based on full conversation context covering
paper analysis, codebase audit, live run debugging, dataset selection, and
architectural design discussions with Shaunak.*