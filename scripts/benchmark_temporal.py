"""Compare supervised legacy and temporal pipelines on one explicit local/synthetic MOV."""

import argparse
import json
from pathlib import Path
from typing import Any

from app.vision.config import VisionConfig
from app.vision.pipeline import analyze_video
from app.vision.result import VisionAnalysisResult
from app.vision.roi import ROI, ROIConfig
from scripts.benchmark_vlm import GPUProbe
from scripts.synthetic_temporal import make_temporal_scene


def technical_summary(result: VisionAnalysisResult) -> dict[str, Any]:
    """Counts, intervals and codes only. Never include media paths or model free text."""
    return {
        "status": result.status,
        "metrics": result.metrics.model_dump(mode="json"),
        "coverage": result.coverage.model_dump(mode="json"),
        "event_intervals": [[e.start_sec, e.end_sec] for e in result.events],
        "event_durations_sec": [e.end_sec - e.start_sec for e in result.events],
        "temporal_complete_count": sum(r.temporal_complete for r in result.event_replay),
        "diagnostic_codes": sorted({d.code for d in result.diagnostics}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video", type=Path, help="Explicit authorized local MOV; never discovered"
    )
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/vision.temporal.example.json"))
    parser.add_argument("--roi", type=Path)
    parser.add_argument("--camera-profile")
    parser.add_argument("--no-vlm", action="store_true", help="CV-only; no model-quality claims")
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/temporal-benchmark-report.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/temporal-benchmark"))
    args = parser.parse_args()
    config = VisionConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    config.output_dir = args.output
    if args.no_vlm:
        config.max_vlm_calls = 0
    if args.synthetic:
        path = make_temporal_scene(args.output / "synthetic.mov")
        roi_path = args.output / "synthetic-roi.local.json"
        roi_path.write_text(
            ROIConfig(
                camera_profile="synthetic-only",
                calibrated=True,
                demonstration=False,
                rois=[
                    ROI(roi_id=key, polygon=[(0, 0.3), (1, 0.3), (1, 1), (0, 1)])
                    for key in ("desk", "cash_area", "transfer_zone")
                ],
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        config.roi_path, config.camera_profile = roi_path, "synthetic-only"
    else:
        path = args.video
        if args.roi:
            config.roi_path = args.roi
        if args.camera_profile:
            config.camera_profile = args.camera_profile
    rows = {}
    for mode in ("legacy", "temporal"):
        runtime = config.model_copy(deep=True)
        runtime.temporal.enabled = mode == "temporal"
        if mode == "legacy":
            runtime.vlm.max_images = 1
            runtime.vlm.num_ctx = 2048
        with GPUProbe() as probe:
            result = analyze_video(path, runtime)
        rows[mode] = technical_summary(result)
        rows[mode]["gpu"] = probe.summary()
        print(
            f"{mode}: {result.status}, {result.metrics.total_sec:.3f}s, "
            f"{len(result.events)} candidates, {result.metrics.vlm_calls} VLM attempts",
            flush=True,
        )
    report = {
        "source_kind": "synthetic" if args.synthetic else "explicit_local_video",
        "inference": "disabled" if args.no_vlm else "local_ollama",
        "comparison_order": ["legacy", "temporal"],
        "runs": rows,
        "limitations": [
            "sequential runs; warm state/cache may differ",
            "VRAM is sampled device-wide, not model-only",
            "no precision/recall for cash transfer or real footage",
        ],
    }
    if args.synthetic:
        report["synthetic_motion_controls"] = [
            {
                "interval": [start, end],
                "overview_sample_inside": any(
                    start < i / config.sample_fps < end for i in range(int(7 * config.sample_fps))
                ),
            }
            for start, end in [(1.5, 2.25), (3.4, 3.45), (4.5, 5.25)]
        ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Technical report: {args.report}")
    return 1 if any(row["status"] in {"failed", "timed_out"} for row in rows.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
