import pytest

from app.vision.vlm import OllamaVisionClient, TemporalVLMObservation, VLMConfig, VLMError


@pytest.fixture
def triplet(tmp_path):
    import numpy as np

    from app.vision.video import encode_jpeg

    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        image = np.zeros((80, 120, 3), np.uint8)
        image[20:40, 10 + i * 25 : 30 + i * 25] = 255
        path.write_bytes(encode_jpeg(image))
        paths.append(path)
    return paths


def test_temporal_schema_and_occlusion_cannot_assert_transfer(sdk, triplet):
    sdk.payload = TemporalVLMObservation.unknown().model_dump(mode="json")
    sdk.payload.update(
        object_transfer="yes",
        transferred_object="cash",
        cash_visible="yes",
        object_movement="yes",
        hand_object_interaction="yes",
        position_change_trackable="unknown",
        object_occluded="yes",
        reflection_possible="no",
    )
    client = OllamaVisionClient(VLMConfig(max_images=3, num_ctx=4096), sdk)
    result = client.observe(
        triplet, [1.2, 1.6, 2.1], temporal=True, roles=["before", "during", "after"]
    )
    assert result.observation.object_transfer == "unknown"
    assert result.observation.tri_state()["transferred_object_cash"] == "unknown"
    call = sdk.calls[0]
    assert call["format"]["additionalProperties"] is False
    assert "object_occluded" in call["format"]["required"]
    assert "occlusion do not establish" in call["messages"][0]["content"]
    assert "roles: ['before', 'during', 'after']" in call["messages"][1]["content"]


@pytest.mark.parametrize("payload", ["not json", {}, {"object_occluded": "maybe"}])
def test_temporal_json_is_not_repaired(sdk, triplet, payload):
    sdk.payload = payload
    client = OllamaVisionClient(VLMConfig(max_images=3), sdk)
    with pytest.raises(VLMError, match="invalid_vlm_json"):
        client.observe(triplet, [0, 0.5, 1], temporal=True, roles=["before", "during", "after"])


def test_missing_temporal_role_forces_unknown_even_for_affirmative_model(sdk, triplet):
    sdk.payload = TemporalVLMObservation.unknown().model_dump(mode="json")
    sdk.payload.update(
        object_transfer="yes",
        transferred_object="cash",
        cash_visible="yes",
        object_movement="yes",
        hand_object_interaction="yes",
        position_change_trackable="yes",
        object_occluded="no",
        reflection_possible="no",
    )
    client = OllamaVisionClient(VLMConfig(max_images=3), sdk)
    result = client.observe(triplet[:2], [1, 2], temporal=True, roles=["during", "after"])
    assert result.observation.object_transfer == "unknown"
    with pytest.raises(ValueError, match="roles"):
        client.observe(triplet, [0, 1, 2], temporal=True, roles=["after", "during", "before"])
