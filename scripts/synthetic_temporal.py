"""Generate a non-sensitive moving-square MOV with two actions and an optional pulse."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def make_temporal_scene(path: Path, *, brief_pulse: bool = True) -> Path:
    """7 s / 20 FPS. Motion at 1.5–2.25, 4.5–5.25; pulse at 3.40–3.45 s.

    The pulse returns between 2 FPS samples and is outside overview proposal rolls;
    it demonstrates a sampling miss, not a ground-truth cash-transfer benchmark.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        for index in range(140):
            t = index / 20
            x = round(
                10 + 65 * min(1, max(0, (t - 1.5) / 0.75)) - 45 * min(1, max(0, (t - 4.5) / 0.75))
            )
            if brief_pulse and 3.4 <= t < 3.45:
                x = 105
            image = np.zeros((120, 160, 3), np.uint8)
            image[50:68, x : x + 18] = 255
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index * 50, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
