"""OpenCV-only local polygon editor; import is safe in headless tests/CI."""

import os
import re
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from app.vision.calibration import (
    ZONE_IDS,
    CalibrationError,
    CalibrationSession,
    DisplayTransform,
    save_rois,
)
from app.vision.video import Image

GUI_HELP = (
    "OpenCV GUI unavailable or disabled in CI. Use an interactive desktop and a separate "
    "GUI venv: python -m pip uninstall -y opencv-python-headless; "
    'python -m pip install "opencv-python>=4.11,<5". See README ROI calibration. '
    "Do not install both OpenCV distributions in the same environment."
)


def require_gui() -> None:
    """Reject known headless/CI environments before calling native window APIs."""
    ci = os.environ.get("CI", "").lower() not in {"", "0", "false", "no"}
    no_display = sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )
    backend = re.search(r"^\s*GUI:\s*(.+)$", cv2.getBuildInformation(), re.MULTILINE)
    if ci or no_display or backend is None or backend.group(1).strip().upper() == "NONE":
        raise CalibrationError(GUI_HELP)


class ROIEditor:
    """Render/edit a session with controlled scaling; no image is written to disk."""

    def __init__(
        self,
        image: Image,
        session: CalibrationSession,
        output: Path | None,
        *,
        max_side: int = 720,
        overwrite: bool = False,
        read_only: bool = False,
        source_status: str = "JPEG/PNG | decoded raster (no auto-rotation)",
        config_status: str = "new inactive draft",
    ) -> None:
        self.image = image
        self.session = session
        self.output = output
        self.max_side = max_side
        self.overwrite = overwrite
        self.read_only = read_only
        self.preview = read_only
        self.pending = False
        self.dragging: int | None = None
        self.saved = False
        self.saved_revision: int | None = None
        self.source_status = source_status
        self.status = config_status
        self.transform = DisplayTransform.fit(image.shape[1], image.shape[0], max_side)

    def mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Add points, drag a nearby selected vertex, or undo with right-click."""
        if self.read_only or self.preview:
            return
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging = None
            return
        try:
            point = self.transform.normalize(x, y)
        except ValueError:
            return  # Toolbar, margin or an out-of-window drag is not an ROI point.
        points = list(self.session.points.get(self.session.selected, []))
        if event == cv2.EVENT_RBUTTONDOWN:
            self.session.change(points[:-1])
        elif event == cv2.EVENT_LBUTTONDOWN:
            near = [
                (i, (px - x) ** 2 + (py - y) ** 2)
                for i, vertex in enumerate(points)
                for px, py in [self.transform.pixel(vertex)]
            ]
            closest = min(near, key=lambda pair: pair[1]) if near else None
            if closest is not None and closest[1] <= 10**2:
                self.dragging = closest[0]
            else:
                try:
                    self.session.change([*points, point])
                except CalibrationError:
                    self.status = "Polygon limit: 100 vertices. Undo or redraw."
                    return
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            points[self.dragging] = point
            self.session.change(points)
        else:
            return
        self.status = "Unsaved geometry. P: validate and review all zones."

    def render(self) -> Image:
        """Overlay all geometry plus controls outside the clickable image raster."""
        view = cv2.resize(self.image, (self.transform.width, self.transform.height))
        overlay = view.copy()
        colors = [(70, 210, 70), (255, 190, 30), (220, 80, 220), (70, 90, 255), (255, 200, 80)]
        for i, (zone_id, points) in enumerate(self.session.points.items()):
            if not points:
                continue
            pixels = np.array([self.transform.pixel(p) for p in points], np.int32)
            color = (0, 255, 255) if zone_id == self.session.selected else colors[i % len(colors)]
            if len(points) >= 3:
                cv2.fillPoly(overlay, [pixels], color)
            cv2.polylines(view, [pixels], len(points) >= 3, color, 2)
            for number, (x, y) in enumerate(pixels):
                cv2.circle(view, (int(x), int(y)), 4, color, -1)
                cv2.putText(view, str(number + 1), (int(x) + 5, int(y) + 14), 0, 0.4, color, 1)
            x, y = map(int, pixels[0])
            cv2.putText(view, zone_id, (max(0, x - 5), max(16, y - 8)), 0, 0.5, color, 1)
        view = cv2.addWeighted(view, 0.8, overlay, 0.2, 0)
        lines = [
            f"Camera: {self.session.camera_profile} | selected: {self.session.selected}",
            self.source_status,
            "1 main_work_area | 2 desk | 3 glass_partition | 4 transfer_zone "
            "| 5 cash_area | 6 safe_area",
            "Click: add | drag vertex: edit | right-click/U: undo | R: redraw/delete selected zone",
            "+/-: scale window | P: preview/edit | S: save INACTIVE draft | Esc/Q: quit",
            "P then C then Y: confirm geometry AND matching camera, save ACTIVE | N: cancel",
            ("READ ONLY | " if self.read_only else "") + self.status,
            "CONFIRM camera and ALL polygons: Y = activate and save, N = cancel"
            if self.pending
            else "Demo never activates. safe_area needs --safe-visible. English keyboard controls.",
        ]
        lines = [part for line in lines for part in textwrap.wrap(line, width=110)]
        panel_height = max(184, 22 * len(lines) + 8)
        canvas = np.zeros(
            (self.transform.height + panel_height, max(960, self.transform.width), 3), np.uint8
        )
        canvas[: self.transform.height, : self.transform.width] = view
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (8, self.transform.height + 20 + 22 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _save(self, confirm: bool) -> None:
        if self.output is None:
            raise CalibrationError("output_required")
        config = self.session.configuration(confirm=confirm)
        save_rois(self.output, config, overwrite=self.overwrite)
        # Subsequent saves may replace the file this editor just created.
        self.overwrite = True
        self.saved = True
        self.saved_revision = self.session.revision
        self.status = "Saved ACTIVE calibration." if confirm else "Saved INACTIVE draft."

    def key(self, key: int) -> bool:
        """Handle one key; return False to close. Errors preserve the last saved file."""
        if key in (27, ord("q")):
            return False
        try:
            if key in (ord("+"), ord("="), ord("-")):
                self.max_side = min(
                    2048, max(160, round(self.max_side * (0.8 if key == 45 else 1.25)))
                )
                self.transform = DisplayTransform.fit(
                    self.image.shape[1], self.image.shape[0], self.max_side
                )
                self.dragging = None
                self.pending = False
            elif self.read_only:
                pass
            elif self.pending:
                if key == ord("y"):
                    self._save(confirm=True)
                    self.pending = False
                elif key == ord("n"):
                    self.pending = False
                    self.status = "Confirmation cancelled; no active calibration saved."
            elif key == ord("p"):
                self.dragging = None
                if not self.preview:
                    self.session.configuration()  # Validate before declaring it reviewed.
                    self.session.reviewed = True
                else:
                    self.session.reviewed = False
                self.preview = not self.preview
                self.status = (
                    "Review ALL polygons and camera. C: confirm." if self.preview else "Edit mode."
                )
            elif key == ord("c"):
                if not self.preview:
                    raise CalibrationError("press_P_to_review_first")
                self.session.configuration(confirm=True)  # Includes demonstration gate.
                self.pending = True
            elif key == ord("s"):
                self._save(confirm=False)
            elif not self.preview:
                self.dragging = None
                if ord("1") <= key <= ord("6"):
                    self.session.select(ZONE_IDS[key - ord("1")])
                elif key in (ord("u"), 8, 127):
                    self.session.change(self.session.points.get(self.session.selected, [])[:-1])
                elif key == ord("r"):
                    self.session.change([])
        except ValidationError:
            self.pending = False
            self.status = (
                "Invalid polygon: need 3+ unique vertices, nonzero area, no crossing/overlap."
            )
        except CalibrationError as exc:
            self.pending = False
            self.status = str(exc)
        except OSError:
            self.pending = False
            self.status = "Save failed (permissions/existing file/disk). Previous JSON preserved."
        return True

    def run(self) -> bool:
        """Open the native editor only after GUI preflight; close cleanly on failure."""
        require_gui()
        name = "Local ROI calibration"
        opened = False
        try:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            opened = True
            cv2.setMouseCallback(name, self.mouse)
            while True:
                cv2.imshow(name, self.render())
                if not self.key(cv2.waitKeyEx(30) & 0xFF):
                    break
                try:
                    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break  # Some desktop backends throw when the user closes the window.
        except cv2.error:
            raise CalibrationError(GUI_HELP) from None
        finally:
            if opened:
                try:
                    cv2.destroyWindow(name)
                except cv2.error:
                    pass
        return self.read_only or (self.saved and self.saved_revision == self.session.revision)
