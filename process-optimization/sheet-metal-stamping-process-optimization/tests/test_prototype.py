from prototype import fit_surrogates, generate_dataset, recommend_settings

def test_risk_bounds():
    df = generate_dataset(300, 3)
    assert df.wrinkle_risk.between(0, 1).all()
    assert df.crack_risk.between(0, 1).all()

def test_optimizer_returns_supported_solution():
    rec = recommend_settings(fit_surrogates(generate_dataset(450, 4), 4), seed=4)
    assert rec.predicted["thinning_pct"] < 22
    assert rec.predicted["crack_risk"] < 0.35
    assert rec.predicted["wrinkle_risk"] < 0.35
