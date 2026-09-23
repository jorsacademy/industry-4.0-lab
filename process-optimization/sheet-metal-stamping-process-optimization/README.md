# Sheet-Metal Stamping Process Optimization

Industry 4.0 prototype for learning a stamping-process response surface and recommending press settings under quality constraints. The synthetic process twin links press force, ram speed, blank-holder force and lubrication to springback, thinning, crack/wrinkle risk and an energy proxy.

The decision layer fixes sheet thickness and material yield strength as job context, then uses Extra Trees surrogates plus differential evolution to search the experimentally supported parameter box.

This project is a software prototype. The process equations and thresholds are illustrative and require calibration against press, die, material-batch, dimensional metrology and defect data before plant use.

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
