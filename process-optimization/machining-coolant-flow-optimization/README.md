# Machining Coolant Flow Optimization

Industry 4.0 prototype for adaptive coolant set-point recommendation in machining. A synthetic process twin links machining load and coolant settings to tool temperature, surface roughness, wear rate and pump-energy demand. Cutting speed, feed and depth are production context; coolant flow, inlet temperature and concentration are optimized controls.

The decision layer penalizes excessive tool temperature while trading off quality, wear and coolant-system energy. It is a software architecture demonstration, not a real machining-fluid prescription; plant calibration and EHS/process validation are required.

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
