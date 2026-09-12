install:
	pip install -r requirements.txt

download:
	python scripts/download_data.py

validate:
	python scripts/validate_data.py

test:
	pytest -q

benchmark:
	python -m src.train --config config.yaml
