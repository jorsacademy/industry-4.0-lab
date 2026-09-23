# Grinding Process Adaptive Control

A compact adaptive-control prototype for grinding. The process twin relates wheel speed, work speed, depth of cut, feed, coolant flow and wheel age to roughness, burn risk, spindle power and wheel-wear rate. The optimizer treats wheel age as observed machine state and recommends new operating settings as that state changes.

The synthetic equations are intended for software prototyping only. A plant deployment would replace them with models learned from spindle power/current, vibration or acoustic emission, coolant, dressing history, wheel metadata, burn inspection and roughness measurements.

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
