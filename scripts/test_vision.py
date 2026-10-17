"""Run the supervised pipeline on an explicitly provided local video."""

import argparse
from pathlib import Path

from app.vision.config import VisionConfig
from app.vision.pipeline import analyze_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, help="Optional VisionConfig JSON")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vision"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/vision-report.json"))
    parser.add_argument("--roi", type=Path)
    parser.add_argument("--camera-profile")
    parser.add_argument("--no-vlm", action="store_true", help="CV-only; candidates remain unknown")
    args = parser.parse_args()
    config = (
        VisionConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if args.config
        else VisionConfig()
    )
    config.output_dir = args.output
    if args.roi:
        config.roi_path = args.roi
    if args.camera_profile:
        config.camera_profile = args.camera_profile
    if args.no_vlm:
        config.max_vlm_calls = 0
    result = analyze_video(args.video, config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"Status: {result.status}")
    print(
        f"Decoded / sampled frames: {result.metrics.decoded_frames} / "
        f"{result.metrics.sampled_frames}"
    )
    print(
        f"Candidates / VLM attempts: {result.metrics.candidates_total} / {result.metrics.vlm_calls}"
    )
    if config.temporal.enabled:
        print(
            "Complete triplets / VLM triplet attempts / single-frame fallback: "
            f"{sum(r.temporal_complete for r in result.event_replay)} / "
            f"{sum(len(r.vlm_frame_paths) == 3 for r in result.event_replay)} / "
            f"{sum(len(r.vlm_frame_paths) == 1 for r in result.event_replay)}"
        )
    print(f"Total seconds: {result.metrics.total_sec:.3f}")
    print(f"Decoded complete: {result.coverage.decoded_complete}")
    print("Diagnostics: " + ", ".join(sorted({d.code for d in result.diagnostics})))
    print(f"Local report: {args.report}")
    return 0 if result.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
