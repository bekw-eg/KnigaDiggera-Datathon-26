import httpx
import numpy as np
import ollama
import pytest
from pydantic import ValidationError

from app.schemas import Observation
from app.vision.video import encode_jpeg
from app.vision.vlm import (
    ObjectKind,
    OllamaVisionClient,
    VLMConfig,
    VLMError,
    VLMObservation,
    inference_lock,
)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "control.jpg"
    path.write_bytes(encode_jpeg(np.zeros((120, 160, 3), np.uint8)))
    return path


def test_valid_image_sdk_options_and_json(sdk, image):
    client = OllamaVisionClient(VLMConfig(max_side=64), sdk)
    result = client.observe([image], [1.2])
    assert result.observation.cash_visible == Observation.UNKNOWN
    assert result.elapsed_sec >= 0 and result.image_count == 1
    call = sdk.calls[0]
    assert call["options"] == {"temperature": 0, "num_ctx": 2048, "num_predict": 200}
    assert call["format"]["additionalProperties"] is False
    assert call["messages"][1]["images"][0].startswith(b"\xff\xd8")


@pytest.mark.parametrize(
    "payload", ["not json", '{"cash_visible":"yes"}', "```json\n{}\n```", "[]"]
)
def test_invalid_json_is_error(sdk, image, payload):
    sdk.payload = payload
    with pytest.raises(VLMError, match="invalid_vlm_json"):
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])


def test_invalid_enum_is_error(sdk, image):
    sdk.payload["cash_visible"] = "probably"
    with pytest.raises(VLMError, match="invalid_vlm_json"):
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])


@pytest.mark.parametrize(
    "error,code",
    [
        (ConnectionError("secret"), "ollama_unavailable"),
        (httpx.ReadTimeout("secret"), "ollama_timeout"),
        (ollama.ResponseError("secret", status_code=404), "model_missing"),
    ],
)
def test_ollama_errors_are_explicit_and_sanitized(sdk, image, error, code):
    sdk.error = error
    with pytest.raises(VLMError, match=code) as captured:
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])
    assert "secret" not in str(captured.value)


def test_no_vision_model_rejected(sdk, image):
    sdk.capabilities = ["completion"]
    with pytest.raises(VLMError, match="no_vision"):
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])
    assert not sdk.calls


def test_truncated_output_rejected(sdk, image):
    sdk.chat = lambda **kwargs: {"done_reason": "length", "message": {"content": "{}"}}
    with pytest.raises(VLMError, match="truncated"):
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])


def test_count_chronology_and_bad_image(sdk, image, tmp_path):
    client = OllamaVisionClient(VLMConfig(), sdk)
    for paths, times in [
        ([], []),
        ([image] * 4, [0, 1, 2, 3]),
        ([image] * 2, [1, 0]),
        ([image] * 2, [0, 0]),
        ([image], [float("nan")]),
    ]:
        with pytest.raises(ValueError):
            client.observe(paths, times)
    with pytest.raises(VLMError, match="invalid_local_image"):
        client.observe([tmp_path / "missing.jpg"], [0])


@pytest.mark.parametrize(
    "host",
    [
        "https://api.example.com",
        "http://192.168.1.1:11434",
        "http://localhost:11434/path",
        "http://x:y@localhost:11434",
    ],
)
def test_only_loopback_endpoint(host):
    with pytest.raises(ValidationError):
        VLMConfig(host=host)


def test_cloud_model_rejected_even_on_loopback(sdk, image):
    with pytest.raises(ValidationError):
        VLMConfig(model="qwen:cloud")
    sdk.show = lambda model: {"capabilities": ["vision"], "remote_host": "remote"}
    with pytest.raises(VLMError, match="remote_model_forbidden"):
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])


def test_ambiguous_transfer_and_reflection_are_unknown(sdk, image):
    sdk.payload.update(
        object_transfer="unknown",
        transferred_object="cash",
        reflection_possible="yes",
        physical_presence="yes",
    )
    observation = OllamaVisionClient(VLMConfig(), sdk).observe([image], [0]).observation
    assert observation.transferred_object == ObjectKind.UNKNOWN
    assert observation.physical_presence == Observation.UNKNOWN
    assert observation.tri_state()["transferred_object_cash"] == Observation.UNKNOWN
    assert VLMObservation.unknown().observations == []


def test_inference_lock_timeout_releases_cleanly():
    with inference_lock(1):
        with pytest.raises(VLMError, match="vlm_lock_timeout"):
            with inference_lock(0.01):
                pass
    with inference_lock(1):
        pass


def test_context_overflow_has_own_safe_diagnostic(sdk, image):
    def overflow(**kwargs):
        raise ollama.ResponseError(
            "request (3270 tokens) exceeds the available context size (2048 tokens)",
            status_code=400,
        )

    sdk.chat = overflow
    with pytest.raises(VLMError, match="vlm_context_exceeded"):
        OllamaVisionClient(VLMConfig(), sdk).observe([image], [0])


def test_single_frame_cannot_establish_transfer_but_multiframe_schema_can(sdk, image):
    sdk.payload.update(object_transfer="yes", transferred_object="cash", cash_visible="yes")
    client = OllamaVisionClient(VLMConfig(max_images=3), sdk)
    assert client.observe([image], [0]).observation.object_transfer == Observation.UNKNOWN
    multi = client.observe([image, image], [0, 1]).observation
    assert multi.object_transfer == Observation.YES and multi.transferred_object == ObjectKind.CASH
