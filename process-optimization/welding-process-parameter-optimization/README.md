# Welding Process Parameter Optimization

A software prototype for Industry 4.0 welding decision support. It creates a physics-informed synthetic welding dataset, fits data-driven surrogate models, and searches for process settings that balance weld strength, porosity risk, penetration, and energy input.

## Process variables

Controllable settings are welding current, voltage, travel speed, wire-feed rate, and shielding-gas flow. Plate thickness is treated as production context rather than a control variable.

## Decision layer

Random-forest surrogates estimate penetration, tensile strength, porosity risk, and heat input. Differential evolution then minimizes a penalized objective while enforcing a supported penetration window relative to plate thickness.

This is a **prototype**, not a validated welding procedure specification. The equations are synthetic and must be replaced or calibrated with real WPS/PQR, weld-monitoring, NDT, destructive-test, and production data before industrial use.

## Run

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
