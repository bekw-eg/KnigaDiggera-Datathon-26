from app.schemas import Observation
from app.vision.detector import Detection, YOLOAdapter
from app.vision.reflection import ReflectionGuard
from app.vision.roi import ROI
from app.vision.vlm import VLMObservation


def test_person_intersecting_glass_does_not_prove_presence():
    glass = ROI(roi_id="glass_partition", polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)])
    person = Detection(class_name="person", bbox=(60, 20, 90, 80), score=0.9)
    observation = VLMObservation.unknown()
    observation.people_visible = Observation.YES
    observation.reflection_possible = Observation.NO
    observation.physical_presence = Observation.YES
    guarded = ReflectionGuard().apply(observation, [person], glass, 100, 100)
    assert guarded.reflection_possible == Observation.YES
    assert guarded.physical_presence == Observation.UNKNOWN
    assert observation.physical_presence == Observation.YES


def test_uncalibrated_glass_and_no_boxes_remain_unknown():
    observation = VLMObservation.unknown()
    observation.people_visible = Observation.YES
    observation.reflection_possible = Observation.NO
    observation.physical_presence = Observation.YES
    assert ReflectionGuard().apply(observation, [], None, 100, 100).physical_presence == "unknown"


def test_disabled_yolo_does_not_import_or_download():
    assert YOLOAdapter().detect(None) == []
