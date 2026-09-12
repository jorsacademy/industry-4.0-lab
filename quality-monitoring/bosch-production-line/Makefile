.PHONY: test download validate build train early

test:
	pytest -q

download:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py --full

build:
	python -m src.build_features --config config.yaml --prefix 1.0

train:
	python -m src.train --config config.yaml

early:
	python -m src.early_warning --config config.yaml
