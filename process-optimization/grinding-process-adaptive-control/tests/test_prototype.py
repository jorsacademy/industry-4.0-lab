from prototype import fit_surrogates, generate_dataset, recommend_settings

def test_risk_and_power_ranges():
    df = generate_dataset(300, 9)
    assert df.burn_risk.between(0, 1).all()
    assert (df.spindle_power_kw > 0).all()

def test_adaptive_recommendation():
    rec = recommend_settings(fit_surrogates(generate_dataset(450, 9), 9), wheel_age=0.65, seed=9)
    assert rec.predicted["burn_risk"] < 0.35
    assert rec.predicted["roughness_um"] < 1.8
