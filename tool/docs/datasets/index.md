# Dataset Expectation Specs

Dataset-specific documentation lives here. Each file follows a common template:
Purpose · Task & models · Raw form · Required preprocessed form · Drift-stream
construction · Config skeleton pointer · Init & calibration · Offline labels ·
License/registration.

Config skeletons (all `"status": "awaiting_data"`) live in `configs/datasets/`.
The validator reports SKIPPED (not FAIL) when a path is absent and status is set.

| File | Domain | Validator SKIPPED until |
|------|--------|------------------------|
| [pems_node2.md](pems_node2.md) | regression | `pems_node2.csv` downloaded |
| [uci_electricity.md](uci_electricity.md) | regression | `LD2011_2014.txt` downloaded and preprocessed |
| [spot_prices.md](spot_prices.md) | regression | ERCOT CSV downloaded |
| [bdd100k.md](bdd100k.md) | cv/detection | BDD100K images unzipped |
| [iwildcam.md](iwildcam.md) | cv/classification | iWildCam manifest generated |
| [acdc.md](acdc.md) | cv/segmentation | ACDC images + masks unzipped |

## Cross-reference

All schema statements in these docs are validated against
`core/dataset_validator.py`. If the validator changes, update the relevant
doc here.

See `DATA_CONTRACT.md` for the full plug-and-play contract.
Per-dataset CV task keys are documented in `configs/datasets/_template.json`.
