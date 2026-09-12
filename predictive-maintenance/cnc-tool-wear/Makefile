.PHONY: validate test train infer

validate:
	python scripts/validate_data.py

test:
	pytest -q

train:
	python -m src.train --config config.yaml

infer:
	python -m src.infer data/raw/experiment_18.csv --config config.yaml
