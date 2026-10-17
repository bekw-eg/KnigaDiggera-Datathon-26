"""Only generated, non-sensitive media. No network required by ordinary tests."""

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest


def make_video(path: Path, moving: bool = True, times: list[int] | None = None) -> Path:
    times = times if times is not None else list(range(0, 2000, 100))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        for index, pts in enumerate(times):
            image = np.zeros((96, 128, 3), np.uint8)
            if moving and index >= 4:
                x = 8 + (index * 7) % 80
                image[25:60, x : x + 20] = 255
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "synthetic.mov")


@pytest.fixture
def sdk():
    from app.vision.vlm import VLMObservation

    class FakeSDK:
        def __init__(self):
            self.payload = VLMObservation.unknown().model_dump(mode="json")
            self.calls = []
            self.capabilities = ["completion", "vision"]
            self.error = None

        def show(self, model):
            if self.error:
                raise self.error
            return {"capabilities": self.capabilities}

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if self.error:
                raise self.error
            content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
            return {"message": {"content": content}, "done_reason": "stop"}

    return FakeSDK()
