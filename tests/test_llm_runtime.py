from types import SimpleNamespace

import pytest

from llm.qwen3_1_7b import runtime


def test_no_gpu_fails_without_loading_model(monkeypatch):
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(xpu=None),
        xpu=SimpleNamespace(is_available=lambda: False),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(runtime, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="CPU inference is forbidden"):
        runtime.QwenRuntime()


def test_xpu_initializes_before_discovery(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(xpu="20260100"),
        xpu=SimpleNamespace(
            init=lambda: events.append("init"),
            is_available=lambda: events.append("discover") or True,
        ),
        device=lambda value: value,
    )
    monkeypatch.setattr(runtime, "torch", fake_torch)
    assert runtime.gpu_device() == "xpu:0"
    assert events == ["init", "discover"]


def test_xpu_failure_propagates_without_cpu_fallback(monkeypatch):
    def fail():
        raise RuntimeError("Driver failed")

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(xpu="20260100"),
        xpu=SimpleNamespace(init=fail),
    )
    monkeypatch.setattr(runtime, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="Driver failed"):
        runtime.gpu_device()
