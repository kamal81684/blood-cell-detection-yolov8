#!/usr/bin/env bash
# End-to-end pipeline: download -> prepare (with preprocessing) -> train -> evaluate.
set -euo pipefail

bash scripts/download_bccd.sh data
python -m bloodcell.prepare_data \
    --src data/BCCD_Dataset-master/BCCD \
    --out data/bccd \
    --preprocess
python -m bloodcell.dataset --root data/bccd
python -m bloodcell.train --data data/bccd/data.yaml --epochs 50 --imgsz 640 --batch 16
python -m bloodcell.evaluate --weights runs/detect/train/weights/best.pt --data data/bccd/data.yaml
echo "All done. Launch the demo with: streamlit run app/streamlit_app.py"
