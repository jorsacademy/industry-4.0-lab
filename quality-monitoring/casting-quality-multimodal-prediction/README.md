# Casting Quality Multimodal Prediction

A smart-quality prototype that demonstrates why casting inspection should combine process context with inspection evidence instead of treating a camera classifier as an isolated system. The synthetic benchmark compares three models: process-only, vision-score-only, and multimodal.

Process variables include mold and pouring temperatures, cooling time, pressure and alloy index. The two image features are **synthetic inspection scores**, not a computer-vision model or real images. They stand in for outputs that could later come from radiography, CT, surface vision or another NDT pipeline.

The project is explicitly a software prototype. Replace the synthetic generator with traceable casting-lot/process records and real NDT features before drawing industrial conclusions.

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
