from prototype import fit_surrogates, generate_dataset, recommend_settings

def test_positive_responses():
    df = generate_dataset(300, 2)
    assert (df[["roughness_um", "burr_mm", "kerf_mm", "energy_kwh_m"]] >= 0).all().all()

def test_optimizer_quality_window():
    rec = recommend_settings(fit_surrogates(generate_dataset(450, 2), 2), 4, 2)
    assert rec.predicted["burr_mm"] < 0.22
    assert rec.predicted["roughness_um"] < 4.5
