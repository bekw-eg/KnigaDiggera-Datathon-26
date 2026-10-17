from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
from conftest import make_video

from app.vision.video import TimestampError, VideoError, VideoReader, encode_jpeg, presentation_time


def test_real_mov_sampling_and_jpeg(video):
    with VideoReader(video) as reader:
        frames = list(reader.sample(fps=2))
        assert reader.metadata.width == 128
        assert reader.metadata.duration_sec == pytest.approx(2)
        assert reader.metadata.fps == pytest.approx(10)
        assert [f.timestamp_sec for f in frames] == pytest.approx([0, 0.5, 1, 1.5])
        assert reader.exhausted and reader.decoded_frames == 20
        assert encode_jpeg(frames[0].image).startswith(b"\xff\xd8")


def test_missing_and_corrupt(tmp_path):
    with pytest.raises(VideoError, match="missing"):
        with VideoReader(tmp_path / "missing.mov"):
            pass
    corrupt = tmp_path / "bad.mov"
    corrupt.write_bytes(b"not video")
    with pytest.raises(VideoError, match="open_failed"):
        with VideoReader(corrupt):
            pass


def test_no_video_stream(tmp_path):
    path = tmp_path / "audio.wav"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=8000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 800), dtype=np.int16), layout="mono")
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(VideoError, match="no_video_stream"):
        with VideoReader(path):
            pass


@pytest.mark.parametrize(
    "pts,base,previous",
    [
        (None, Fraction(1, 10), None),
        (1, None, None),
        (1, Fraction(-1, 10), None),
        (-1, Fraction(1, 10), None),
        (1, Fraction(1, 10), 0.2),
        (1, Fraction(1, 10), 0.1),
    ],
)
def test_bad_pts(pts, base, previous):
    with pytest.raises(TimestampError):
        presentation_time(SimpleNamespace(pts=pts, time_base=base), 0, previous)


def test_nonzero_origin_and_vfr_are_not_frame_index_over_fps(tmp_path):
    frame = SimpleNamespace(pts=127, time_base=Fraction(1, 10))
    assert presentation_time(frame, 12, 0.5) == pytest.approx(0.7)
    path = make_video(tmp_path / "vfr.mov", times=[0, 100, 400, 900, 1300])
    with VideoReader(path) as reader:
        frames = list(reader.sample(fps=10))
    assert [f.timestamp_sec for f in frames] == pytest.approx([0, 0.1, 0.4, 0.9, 1.3])


def test_interval_sampling(video):
    with VideoReader(video) as reader:
        frames = list(reader.sample(fps=5, start_sec=0.3, end_sec=0.8, max_side=64))
    assert [f.timestamp_sec for f in frames] == pytest.approx([0.3, 0.5, 0.7])
    assert frames[0].image.shape[1] == 64
