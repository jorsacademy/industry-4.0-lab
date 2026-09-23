# Laser Cutting Process Optimization

Physics-informed synthetic prototype for laser-cutting parameter selection. A surrogate model maps laser power, cutting speed, assist-gas pressure, focus offset and sheet thickness to roughness, burr height, kerf width and energy per cut length. A constrained multi-response search then recommends settings for a specified thickness.

The simulator is deliberately compact and educational; it is not a substitute for material-specific cutting charts or machine OEM process windows. Calibrate it with real cut-quality, power, gas, material and metrology data before industrial use.

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
