"""Streaming PyAV decoder with explicit, validated presentation timestamps."""

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from app.schemas import ContractModel, Seconds

Image = NDArray[np.uint8]


class VideoError(RuntimeError):
    """Missing, unreadable or undecodable video."""


class TimestampError(VideoError):
    """Exact timestamp unavailable; no frame-index/FPS fallback is applied."""


class VideoMetadata(ContractModel):
    width: int
    height: int
    duration_sec: Seconds | None
    fps: float | None
    stream_start_sec: float | None
    time_base: str


@dataclass
class VideoFrame:
    image: Image
    timestamp_sec: float
    source_pts: int
    time_base: str
    source_time_sec: float


def presentation_time(frame: Any, origin: float, previous: float | None) -> float:
    """Return source PTS relative to stream origin, rejecting bad/nonmonotonic PTS."""
    if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
        raise TimestampError("missing_or_invalid_pts")
    timestamp = float(frame.pts * frame.time_base) - origin
    if not math.isfinite(timestamp) or timestamp < -1e-6:
        raise TimestampError("invalid_pts_value")
    timestamp = max(0.0, timestamp)
    if previous is not None and timestamp <= previous:
        raise TimestampError("nonmonotonic_pts")
    return timestamp


def resize_image(image: Image, max_side: int) -> Image:
    """Bound image resolution without upscaling or changing its aspect ratio."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))


def encode_jpeg(image: Image, quality: int = 85) -> bytes:
    """Encode a BGR frame; writing bytes also supports Unicode Windows paths."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be 1..100")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("jpeg_encode_failed")
    return encoded.tobytes()


class VideoReader:
    """Context-managed single-pass decoder. All times are seconds from stream start."""

    def __init__(self, path: Path, max_input_pixels: int = 40_000_000):
        self.path = Path(path)
        self.max_input_pixels = max_input_pixels
        self.decoded_frames = 0
        self.decoded_until_sec: float | None = None
        self.exhausted = False
        self._container = None

    def __enter__(self) -> "VideoReader":
        if not self.path.is_file():
            raise VideoError("video_file_missing")
        try:
            self._container = av.open(str(self.path))
            if not self._container.streams.video:
                raise VideoError("no_video_stream")
            self._stream = self._container.streams.video[0]
            stream = self._stream
            width, height = stream.codec_context.width, stream.codec_context.height
            if width <= 0 or height <= 0 or width * height > self.max_input_pixels:
                raise VideoError("invalid_or_excessive_resolution")
            if stream.time_base is None or stream.time_base <= 0:
                raise TimestampError("missing_stream_time_base")
            duration = (
                float(stream.duration * stream.time_base) if stream.duration is not None else None
            )
            # Container duration can include audio/edit offsets: don't silently treat it as video.
            rate = stream.average_rate or stream.base_rate
            self.metadata = VideoMetadata(
                width=width,
                height=height,
                duration_sec=duration if duration is not None and duration >= 0 else None,
                fps=float(rate) if rate and rate > 0 else None,
                stream_start_sec=(
                    float(stream.start_time * stream.time_base)
                    if stream.start_time is not None
                    else None
                ),
                time_base=str(stream.time_base),
            )
            return self
        except Exception as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, VideoError):
                raise
            raise VideoError("video_open_failed") from None

    def __exit__(self, *_: Any) -> None:
        if self._container is not None:
            self._container.close()

    def sample(
        self,
        fps: float = 2.0,
        start_sec: float = 0,
        end_sec: float | None = None,
        max_side: int = 1280,
        deadline: float | None = None,
        max_decoded_frames: int | None = None,
    ) -> Iterator[VideoFrame]:
        """Decode sequentially, yielding first frame at/after each sampling target.

        Timestamp validation applies to skipped frames too. Decode remains streaming;
        only a selected frame is converted to a numpy image. Sparse sampling can miss
        short actions. No approximate timestamp fallback exists.
        """
        if not math.isfinite(fps) or fps <= 0 or max_side < 1:
            raise ValueError("invalid sampling configuration")
        if start_sec < 0 or (end_sec is not None and end_sec < start_sec):
            raise ValueError("invalid sampling interval")
        if self._container is None:
            raise VideoError("reader_not_open")
        target = start_sec
        origin = self.metadata.stream_start_sec
        previous = None
        try:
            decoded = iter(self._container.decode(self._stream))
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise VideoError("processing_deadline")
                if max_decoded_frames is not None and self.decoded_frames >= max_decoded_frames:
                    raise VideoError("decode_frame_limit")
                try:
                    frame = next(decoded)
                except StopIteration:
                    break
                if origin is None:
                    if frame.pts is None or frame.time_base is None:
                        raise TimestampError("missing_initial_pts")
                    origin = float(frame.pts * frame.time_base)
                timestamp = presentation_time(frame, origin, previous)
                previous = timestamp
                self.decoded_frames += 1
                self.decoded_until_sec = timestamp
                if end_sec is not None and timestamp > end_sec:
                    return
                if timestamp + 1e-9 < target:
                    continue
                yield VideoFrame(
                    image=resize_image(frame.to_ndarray(format="bgr24"), max_side),
                    timestamp_sec=timestamp,
                    source_pts=frame.pts,
                    time_base=str(frame.time_base),
                    source_time_sec=float(frame.pts * frame.time_base),
                )
                # Avoid duplicate frames after a large VFR gap.
                target = start_sec + (math.floor((timestamp - start_sec) * fps + 1e-9) + 1) / fps
            self.exhausted = True
            if self.decoded_frames == 0:
                raise VideoError("empty_video_stream")
        except VideoError:
            raise
        except Exception:
            raise VideoError("video_decode_failed") from None
