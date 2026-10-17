"""Manually calibrate polygons on a local JPEG/PNG or a PTS-selected MOV frame."""

import argparse
import math
import sys
from pathlib import Path

import cv2
from pydantic import ValidationError

from app.vision.calibration import CalibrationError, CalibrationSession, load_calibration_frame
from app.vision.roi import load_rois
from app.vision.video import VideoError
from scripts.roi_editor import ROIEditor, require_gui


def main(argv: list[str] | None = None) -> int:
    """Run a local GUI; 0 = saved/viewed, 2 = error or closed without saving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local JPEG/PNG or MOV (never modified)")
    parser.add_argument("--camera-profile", required=True, help="Exact camera slug, e.g. camera-01")
    parser.add_argument("--output", type=Path, help="JSON destination (required for editing)")
    parser.add_argument("--load", type=Path, help="Load existing ROIConfig to edit or preview")
    parser.add_argument(
        "--time-sec", type=float, default=0, help="MOV seconds relative to stream start"
    )
    parser.add_argument(
        "--max-side", type=int, default=720, help="Initial display size; +/- rescales"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace existing output"
    )
    parser.add_argument(
        "--demonstration", action="store_true", help="Demo: activation always forbidden"
    )
    parser.add_argument(
        "--safe-visible", action="store_true", help="I verified the safe is visible"
    )
    parser.add_argument("--preview", action="store_true", help="Read-only overlay; requires --load")
    args = parser.parse_args(argv)
    if args.preview and not args.load:
        parser.error("--preview requires --load")
    if not args.preview and args.output is None:
        parser.error("editing requires --output")
    if not 160 <= args.max_side <= 2048:
        parser.error("--max-side must be 160..2048")
    if not math.isfinite(args.time_sec) or args.time_sec < 0:
        parser.error("--time-sec must be finite and nonnegative")
    try:
        config = load_rois(args.load) if args.load else None
        session = CalibrationSession(
            args.camera_profile,
            config,
            demonstration=args.demonstration,
            safe_visible=args.safe_visible,
        )
        if args.output is not None and not args.preview:
            if args.output.resolve() == Path("configs/roi.json").resolve():
                raise CalibrationError("repository_demo_config_is_read_only")
            if args.output.suffix.lower() != ".json":
                raise CalibrationError("output_must_be_json")
            if args.load and args.output.resolve() == args.load.resolve() and not args.overwrite:
                raise CalibrationError("existing_output_requires_--overwrite")
            if args.output.exists() and not args.overwrite:
                raise CalibrationError("existing_output_requires_--overwrite")
        require_gui()  # Fail before decoding, opening a native window or entering a wait loop.
        image, frame = load_calibration_frame(args.source, args.time_sec)
        source_status = (
            f"Relative PTS: {frame.timestamp_sec:.6f}s | source PTS: {frame.source_pts} "
            f"| time_base: {frame.time_base} | requested: {args.time_sec:.6f}s"
            if frame
            else "JPEG/PNG | decoded raster (no auto-rotation)"
        )
        status = (
            f"Loaded: calibrated={config.calibrated}, demonstration={config.demonstration}"
            if config
            else "New INACTIVE draft. Draw only visible zones."
        )
        saved = ROIEditor(
            image,
            session,
            args.output,
            max_side=args.max_side,
            overwrite=args.overwrite,
            read_only=args.preview,
            source_status=source_status,
            config_status=status,
        ).run()
        print(
            "ROI saved/viewed locally." if saved else "Closed with unsaved edits or without saving."
        )
        return 0 if saved else 2
    except ValidationError:
        print(
            "Invalid ROI config/profile. Check polygon geometry and camera slug (see README).",
            file=sys.stderr,
        )
    except (CalibrationError, VideoError) as exc:
        print(str(exc), file=sys.stderr)
    except (OSError, cv2.error):
        print(
            "Local file read/write or image decode failed; check paths/permissions.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
