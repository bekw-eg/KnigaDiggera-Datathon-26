from scripts.benchmark_vlm import gpu_snapshot, latency_summary


def test_latency_p95_requires_sufficient_samples():
    assert latency_summary([])["mean_sec"] is None
    assert latency_summary([1, 2])["p95_sec"] is None
    result = latency_summary(list(range(1, 21)))
    assert result["p95_sec"] == 19 and result["median_sec"] == 10.5


def test_missing_gpu_tool_is_unavailable(monkeypatch):
    monkeypatch.setattr("scripts.benchmark_vlm.shutil.which", lambda executable: None)
    assert gpu_snapshot() is None
