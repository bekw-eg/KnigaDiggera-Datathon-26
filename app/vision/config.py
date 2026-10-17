"""Validated runtime configuration; .env loading belongs to the backend."""

from pathlib import Path

from pydantic import Field, model_validator

from app.schemas import ContractModel
from app.vision.motion import MotionConfig
from app.vision.roi import CameraProfile
from app.vision.temporal import TemporalConfig
from app.vision.vlm import VLMConfig


class VisionConfig(ContractModel):
    sample_fps: float = Field(default=2.0, gt=0, le=30)
    max_seconds: float = Field(default=300, gt=0, le=300)
    worker_shutdown_reserve_sec: float = Field(default=2, ge=0, le=10)
    max_decoded_frames: int = Field(default=100000, ge=1, le=1000000)
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
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    yolo_enabled: bool = False
    yolo_weights: Path = Path("models/yolo11n.pt")
    yolo_device: str = "cuda:0"

    @model_validator(mode="after")
    def refine_more_densely(self) -> "VisionConfig":
        if self.temporal.enabled and self.temporal.refine_fps <= self.sample_fps:
            raise ValueError("temporal refinement must sample more densely than overview")
        return self
