"""Conservative MVP reflection guard; no biometric matching or identity analysis."""

from app.schemas import Observation
from app.vision.detector import Detection
from app.vision.roi import ROI
from app.vision.vlm import VLMObservation


class ReflectionGuard:
    """Treat glass overlap as possible reflection, not proof of presence/absence."""

    def apply(
        self,
        observation: VLMObservation,
        detections: list[Detection],
        glass: ROI | None,
        width: int,
        height: int,
        region_is_glass: bool = False,
    ) -> VLMObservation:
        output = observation.model_copy(deep=True)
        glass_hit = glass is not None and any(
            d.class_name == "person" and glass.box_overlap(d.bbox, width, height) > 0
            for d in detections
        )
        if glass_hit or (region_is_glass and output.people_visible != Observation.NO):
            output.reflection_possible = Observation.YES
            output.physical_presence = Observation.UNKNOWN
        elif output.people_visible != Observation.NO and (glass is None or not detections):
            # Without calibrated glass geometry this observation cannot establish presence.
            output.physical_presence = Observation.UNKNOWN
        if output.reflection_possible != Observation.NO:
            output.physical_presence = Observation.UNKNOWN
        return output
