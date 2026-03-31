from ace_atlas.config import ACEAtlasConfig


def test_small_config_is_serializable() -> None:
    config = ACEAtlasConfig.small()
    data = config.to_dict()
    assert data["model_dim"] == 512
    assert data["num_layers"] == 8
    assert data["moe"]["enabled"] is True

