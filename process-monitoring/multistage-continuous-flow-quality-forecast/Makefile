.PHONY: install download validate test benchmark

install:
	pip install -r requirements.txt

download:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py

test:
	pytest -q

benchmark: download validate
	python -m src.train --config config.yaml
