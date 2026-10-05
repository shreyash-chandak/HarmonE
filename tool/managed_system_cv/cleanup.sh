#!/bin/bash
# LEGACY RESET SCRIPT — kept for reference; canonical per-run reset is:
#   python3 experiments/run_reset.py --domain cv
# This script resets the CV managed system to a clean slate (models, knowledge,
# predictions.csv). Use run_reset.py for experiment automation — it is idempotent,
# config-driven, and preserves training artifacts (scaler, versionedMR weights).
set -e

DO_BACKUP=true
SUFFIX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -x)
      DO_BACKUP=false
      shift
      ;;
    -s)
      shift
      if [[ -n "$1" ]]; then
        SUFFIX="_$1"
        shift
      else
        echo "Error: -s requires an argument."
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if $DO_BACKUP; then
  cp -r knowledge "runs_artifact/knowledge_$(date +%d_%H:%M:%S)${SUFFIX}"
fi

# Remove models and versionedMR directories if they exist
rm -rf models # versionedMR

cp -r base_models models

# Reset predictions.csv with header
echo "image_name,confidence,model_used,inference_time,energy_uJ,histogram" > knowledge/predictions.csv

rm -f knowledge/mape_log.csv

rm -f knowledge/drift_kl.json

rm -rf knowledge/inferences

# Reset mape_info.json with given content
cat > knowledge/mape_info.json <<EOL
{
  "last_line": 0,
  "current_energy_threshold": 0.43,
  "ema_scores": {
    "yolo_n": 0.53,
    "yolo_s": 0.55,
    "yolo_m": 0.57
  },
  "recovery_cycles": 0
}
EOL

echo "yolo_m" > knowledge/model.csv

# Remove all __pycache__ directories recursively
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup complete."

