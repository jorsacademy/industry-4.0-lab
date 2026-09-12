.PHONY: install download validate test train

install:
	python -m pip install -r requirements.txt

download:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py

test:
	pytest -q

train:
	python -m src.train --config config.yaml
