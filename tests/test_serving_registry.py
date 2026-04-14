from backend.serving.model_registry import get_serving_info


def test_serving_info_shape() -> None:
    info = get_serving_info()
    d = info.to_public_dict()
    assert "extraction_fast_model" in d
    assert "build_version" in d
