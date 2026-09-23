from prototype import fit_surrogates, generate_dataset, recommend_settings

def test_response_ranges():
    df = generate_dataset(300, 5)
    assert df.hardness_hrc.between(0, 70).all()
    assert (df.energy_kwh > 0).all()

def test_optimizer_meets_targets_approximately():
    rec = recommend_settings(fit_surrogates(generate_dataset(450, 5), 5), seed=5)
    assert rec.predicted["hardness_hrc"] >= 45
    assert rec.predicted["tensile_mpa"] >= 900
