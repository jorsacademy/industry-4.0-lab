from prototype import benchmark, generate_dataset

def test_dataset_scores_are_bounded():
    df = generate_dataset(500, 15)
    assert df.image_porosity_score.between(0, 1).all()
    assert set(df.defect.unique()) <= {0, 1}

def test_multimodal_model_is_informative():
    b = benchmark(generate_dataset(1000, 15), 15)
    assert b.process_auc > 0.65
    assert b.vision_auc > 0.75
    assert b.multimodal_auc > 0.80
