import pandas as pd

from src.modeling import encode_binary_target, make_splitter


def test_encode_binary_target():
    values = pd.Series(["worn", "unworn", "worn"])
    assert encode_binary_target(values, "worn").tolist() == [1, 0, 1]


def test_group_split_never_leaks_experiment():
    groups = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    splitter = make_splitter(y, groups, maximum_splits=4, random_state=42)
    X = pd.DataFrame({"x": range(len(y))})
    for train_idx, test_idx in splitter.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)
