.PHONY: install data prepare train eval predict app test clean

install:
	pip install -r requirements.txt && pip install -e .

data:
	bash scripts/download_bccd.sh

prepare:
	python -m bloodcell.prepare_data --src data/BCCD_Dataset-master/BCCD --out data/bccd

train:
	python -m bloodcell.train --data data/bccd/data.yaml --epochs 50 --imgsz 640 --batch 16

eval:
	python -m bloodcell.evaluate --weights runs/detect/train/weights/best.pt --data data/bccd/data.yaml

predict:
	python -m bloodcell.predict --weights runs/detect/train/weights/best.pt --source data/bccd/images/test

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

clean:
	rm -rf runs __pycache__ src/bloodcell/__pycache__
