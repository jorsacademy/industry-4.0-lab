.PHONY: test download validate benchmark

test:
	pytest -q

download:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py

benchmark:
	python -m src.train --config config.yaml
