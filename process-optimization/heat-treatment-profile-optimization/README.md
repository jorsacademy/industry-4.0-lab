# Heat-Treatment Profile Optimization

Industry 4.0 prototype that learns heat-treatment response surfaces and recommends a thermal cycle for a specified carbon content. The synthetic twin models heating rate, soak temperature, soak duration and cooling rate against hardness, tensile strength, distortion and energy.

A tree-based surrogate is coupled to differential evolution with minimum hardness and tensile-strength targets. The project is intended to demonstrate the software architecture for data-driven thermal-process decision support, not to prescribe a metallurgical recipe.

```bash
pip install -r requirements.txt
python prototype.py
pytest -q
```
