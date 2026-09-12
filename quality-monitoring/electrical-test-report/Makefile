.PHONY: install download validate test train monitor

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

monitor:
	python -m src.monitoring data/raw/Electrical_Test_Report.csv
