from prototype import FEATURES, fit_surrogates, generate_dataset, recommend_settings

def test_dataset_and_ranges():
    df = generate_dataset(300, seed=7)
    assert set(FEATURES).issubset(df.columns)
    assert df["porosity_risk"].between(0, 1).all()
    assert (df["energy_kj_mm"] > 0).all()

def test_recommendation_is_feasible():
    models = fit_surrogates(generate_dataset(450, seed=8), seed=8)
    rec = recommend_settings(models, thickness_mm=5.0, seed=8)
    assert 2.75 <= rec.predicted["penetration_mm"] <= 4.60
    assert 0 <= rec.predicted["porosity_risk"] <= 1
