"""Optional, lazy YOLO adapter. Existing local weights are required; no downloads."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from app.schemas import ContractModel
from app.vision.roi import ROI
from app.vision.video import Image


class Detection(ContractModel):
    class_name: str
    bbox: tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    roi_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_box(self) -> Self:
        x1, y1, x2, y2 = self.bbox
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("invalid bounding box")
        return self


class YOLOAdapter:
    """Class labels/scores are detector outputs, not identity or legal conclusions."""

    def __init__(
        self,
        weights: Path = Path("models/yolo11n.pt"),
        enabled: bool = False,
        device: str = "cuda:0",
        score_threshold: float = 0.5,
    ):
        self.weights, self.enabled = Path(weights), enabled
        self.device, self.score_threshold = device, score_threshold
        self._model = None

    def detect(self, image: Image, rois: list[ROI] | None = None) -> list[Detection]:
        """Load only on demand, falling back to CPU when the requested GPU is unavailable."""
        if not self.enabled:
            return []
        if not self.weights.is_file():
            raise RuntimeError("yolo_local_weights_missing")
        import torch
        from ultralytics import YOLO

        if self._model is None:
            self._model = YOLO(str(self.weights))
        device = self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu"
        try:
            results = self._model.predict(
                image, device=device, conf=self.score_threshold, verbose=False
            )
        except RuntimeError:
            if device == "cpu":
                raise
            results = self._model.predict(
                image, device="cpu", conf=self.score_threshold, verbose=False
            )
        height, width = image.shape[:2]
        detections = []
        for result in results:
            for box in result.boxes:
                coords = tuple(float(x) for x in box.xyxy[0].cpu().tolist())
                coords = (
                    max(0, coords[0]),
                    max(0, coords[1]),
                    min(width, coords[2]),
                    min(height, coords[3]),
                )
                detections.append(
                    Detection(
                        class_name=result.names[int(box.cls[0])],
                        bbox=coords,
                        score=float(box.conf[0]),
                        roi_ids=[
                            roi.roi_id
                            for roi in rois or []
                            if roi.box_overlap(coords, width, height) > 0
                        ],
                    )
                )
        return detections
