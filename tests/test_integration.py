"""Opt-in checks; file contents never leave the loopback Ollama service."""

import os
from pathlib import Path

import pytest

from app.vision.config import VisionConfig
from app.vision.pipeline import analyze_video
from app.vision.vlm import OllamaVisionClient, VLMConfig


@pytest.mark.integration
def test_local_model_images():
    folder = os.environ.get("VISION_INTEGRATION_JPEG_DIR")
    if not folder:
        pytest.skip("set VISION_INTEGRATION_JPEG_DIR to explicitly opt in")
    paths = sorted(Path(folder).glob("*.jpg"))[:3]
    assert paths, "explicit integration directory must contain JPEGs"
    config = VLMConfig(max_images=3, num_ctx=4096)
    result = OllamaVisionClient(config).observe(paths, list(map(float, range(len(paths)))))
    assert result.image_count == len(paths)


@pytest.mark.integration
def test_local_mov_pipeline(tmp_path):
    video = os.environ.get("VISION_INTEGRATION_VIDEO")
    if not video:
        pytest.skip("set VISION_INTEGRATION_VIDEO to explicitly opt in")
    OllamaVisionClient(VLMConfig()).check_ready()
    result = analyze_video(Path(video), VisionConfig(output_dir=tmp_path / "evidence"))
    assert result.status == "complete", [d.code for d in result.diagnostics]
    assert result.coverage.decoded_complete
