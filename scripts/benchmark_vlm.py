"""Measure local vision latency and optional allocation/VRAM; never upload images."""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.vision.vlm import OllamaVisionClient, VLMConfig, VLMError, inference_lock


def gpu_snapshot() -> list[dict[str, int]] | None:
    """Device-wide memory from nvidia-smi; unavailable is distinct from zero."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        devices = []
        for line in completed.stdout.splitlines():
            index, used, total = (int(item.strip()) for item in line.split(","))
            devices.append({"index": index, "used_mib": used, "total_mib": total})
        return devices or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class GPUProbe:
    """Periodically sample device memory; peak is observed, not exact instantaneous peak."""

    def __init__(self):
        self.samples: list[list[dict[str, int]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshot = gpu_snapshot()
            if snapshot is not None:
                self.samples.append(snapshot)
            self._stop.wait(0.5)

    def __enter__(self) -> "GPUProbe":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def summary(self) -> dict[str, Any]:
        flat = [device for sample in self.samples for device in sample]
        return {
            "available": bool(flat),
            "sample_count": len(self.samples),
            "observed_peak_used_mib": max((d["used_mib"] for d in flat), default=None),
            "scope": "all processes; largest used memory across devices, not model-only",
        }


def allocation(client: OllamaVisionClient) -> dict[str, Any] | None:
    """Use Ollama /api/ps allocation sizes, never guess utilization from VRAM."""
    sdk = client._sdk(3)
    try:
        entries = sdk.ps().models
        for entry in entries:
            if entry.model == client.config.model or entry.name == client.config.model:
                total, vram = entry.size, entry.size_vram
                return {
                    "loaded": True,
                    "total_bytes": total,
                    "vram_bytes": vram,
                    "cpu_bytes": max(0, total - vram)
                    if total is not None and vram is not None
                    else None,
                }
        return {"loaded": False, "total_bytes": None, "vram_bytes": None, "cpu_bytes": None}
    except Exception:
        return None
    finally:
        sdk._client.close()


def latency_summary(samples: list[float]) -> dict[str, float | int | None]:
    """P95 uses nearest rank and is only shown for >=20 successful samples."""
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "mean_sec": statistics.mean(samples) if samples else None,
        "median_sec": statistics.median(samples) if samples else None,
        "p95_sec": ordered[math.ceil(0.95 * len(ordered)) - 1] if len(ordered) >= 20 else None,
    }


def benchmark(
    folder: Path,
    config: VLMConfig,
    repeats: int = 1,
    images_per_request: int = 1,
    cold_start: bool = False,
) -> dict[str, Any]:
    """Benchmark every local JPEG; report indexes/counts/timing, never image contents/names."""
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    if not paths:
        raise ValueError("no_jpeg_images")
    if repeats < 1 or not 1 <= images_per_request <= config.max_images:
        raise ValueError("invalid benchmark request count")
    client = OllamaVisionClient(config)
    before = allocation(client)
    setup_diagnostics = []
    if cold_start and before is not None and before["loaded"]:
        try:
            with inference_lock(config.timeout_sec):
                sdk = client._sdk(config.timeout_sec)
                try:
                    sdk.generate(model=config.model, keep_alive=0)
                finally:
                    sdk._client.close()
        except Exception:
            setup_diagnostics.append("model_unload_failed")
        before = allocation(client)
    rows = []
    started = time.perf_counter()
    with GPUProbe() as probe:
        for repeat in range(repeats):
            for offset in range(0, len(paths), images_per_request):
                batch = paths[offset : offset + images_per_request]
                call_started = time.perf_counter()
                row: dict[str, Any] = {
                    "request_index": len(rows),
                    "repeat_index": repeat,
                    "image_count": len(batch),
                    "error": None,
                }
                try:
                    client.observe(batch, [float(i) for i in range(len(batch))])
                    row["success"] = True
                except VLMError as exc:
                    row.update(success=False, error=exc.code)
                row["elapsed_sec"] = time.perf_counter() - call_started
                rows.append(row)
                print(
                    f"{row['request_index'] + 1:>4} | {len(batch):>6} | "
                    f"{row['elapsed_sec']:>8.3f} | {row['error'] or 'ok'}",
                    flush=True,
                )
                if row["error"] in {
                    "ollama_unavailable",
                    "model_missing",
                    "model_has_no_vision_capability",
                    "remote_model_forbidden",
                }:
                    break
            else:
                continue
            break
    cold = bool(before is not None and before["loaded"] is False)
    successful = [row["elapsed_sec"] for row in rows if row["success"]]
    warm = [
        row["elapsed_sec"]
        for row in rows
        if row["success"]
        and (row["request_index"] > 0 or (before is not None and before["loaded"]))
    ]
    cold_seconds = rows[0]["elapsed_sec"] if cold and rows and rows[0]["success"] else None
    recommendations = []
    if any(row["error"] == "vlm_output_truncated" for row in rows):
        recommendations.append("Increase num_predict and repeat: output truncation was measured.")
    if any(row["error"] == "ollama_timeout" for row in rows):
        recommendations.append("Measure smaller max_side / fewer images after observed timeouts.")
    if any(row["error"] == "vlm_context_exceeded" for row in rows):
        recommendations.append(
            "Reduce image count/resolution or benchmark a larger num_ctx after context overflow."
        )
    if warm:
        estimate = 6 * statistics.mean(warm)
        recommendations.append(
            f"Six requests at measured warm mean need about {estimate:.1f}s of VLM time; "
            "decode/CV and real-video validation are additional."
        )
    if len(warm) < 20:
        recommendations.append("Collect at least 20 successful warm requests before reporting P95.")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": config.model,
        "configuration": config.model_dump(mode="json"),
        "available_images": len(paths),
        "repeats_requested": repeats,
        "images_per_request": images_per_request,
        "attempted_images": sum(row["image_count"] for row in rows),
        "processed_images": sum(row["image_count"] for row in rows if row["success"]),
        "errors": sum(not row["success"] for row in rows),
        "requests": rows,
        "all_successful": latency_summary(successful),
        "warm": latency_summary(warm),
        "cold_start_sec": cold_seconds,
        "first_request_sec": rows[0]["elapsed_sec"],
        "cold_basis": "not loaded in /api/ps before first request; OS disk cache not reset",
        "unload_requested": cold_start,
        "setup_diagnostics": setup_diagnostics,
        "allocation_before": before,
        "allocation_after": allocation(client),
        "gpu_memory": probe.summary(),
        "total_sec": time.perf_counter() - started,
        "recommendations": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", type=Path, help="VLMConfig JSON, not VisionConfig")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--images-per-request", type=int, default=1)
    parser.add_argument(
        "--cold-start", action="store_true", help="Unload selected local model first"
    )
    parser.add_argument("--report", type=Path, default=Path("artifacts/benchmark-report.json"))
    args = parser.parse_args()
    config = (
        VLMConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if args.config
        else VLMConfig()
    )
    if args.model:
        config.model = args.model
    print("Call | Images | Seconds  | Result", flush=True)
    report = benchmark(args.folder, config, args.repeats, args.images_per_request, args.cold_start)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for label in (
        "cold_start_sec",
        "first_request_sec",
        "all_successful",
        "warm",
        "errors",
        "processed_images",
        "gpu_memory",
        "allocation_after",
    ):
        print(f"{label}: {report[label]}")
    for recommendation in report["recommendations"]:
        print(recommendation)
    print(f"Local report: {args.report}")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
