from prototype import fit_surrogates, generate_dataset, recommend_coolant

def test_physical_outputs():
    df = generate_dataset(300, 12)
    assert (df.tool_temp_c > 0).all()
    assert (df.pump_energy_index >= 0).all()

def test_coolant_recommendation_controls_temperature():
    rec = recommend_coolant(fit_surrogates(generate_dataset(450, 12), 12), seed=12)
    assert rec.predicted["tool_temp_c"] < 130
