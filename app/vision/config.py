"""Validated runtime configuration; .env loading belongs to the backend."""

from pathlib import Path

from pydantic import Field

from app.schemas import ContractModel
from app.vision.motion import MotionConfig
from app.vision.roi import CameraProfile
from app.vision.vlm import VLMConfig


class VisionConfig(ContractModel):
    sample_fps: float = Field(default=2.0, gt=0, le=30)
    max_seconds: float = Field(default=300, gt=0, le=300)
    max_sampled_frames: int = Field(default=1200, ge=1, le=10000)
    max_candidates: int = Field(default=24, ge=1, le=1000)
    max_vlm_calls: int = Field(default=6, ge=0, le=100)
    max_frame_side: int = Field(default=1280, ge=64, le=2048)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    output_dir: Path = Path("artifacts/vision")
    roi_path: Path | None = None
    camera_profile: CameraProfile | None = None
    motion: MotionConfig = Field(default_factory=MotionConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    yolo_enabled: bool = False
    yolo_weights: Path = Path("models/yolo11n.pt")
    yolo_device: str = "cuda:0"
