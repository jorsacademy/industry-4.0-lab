from src.schema import StationKey, group_columns_by_station, parse_feature_name


def test_parse_bosch_feature_name() -> None:
    parsed = parse_feature_name("L3_S36_F3939")
    assert parsed == (StationKey(3, 36), "F", 3939)
    assert parse_feature_name("Response") is None


def test_group_columns_by_station() -> None:
    groups = group_columns_by_station(["Id", "L0_S0_F0", "L0_S0_F2", "L1_S2_D4"], kind="F")
    assert groups[StationKey(0, 0)] == ["L0_S0_F0", "L0_S0_F2"]
    assert StationKey(1, 2) not in groups
