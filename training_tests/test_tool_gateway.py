from __future__ import annotations

import importlib.util
from pathlib import Path

import requests


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gateway = load_module("tool_gateway", ROOT / "agent/services/tool_gateway.py")


def test_jina_network_failure_uses_direct_reader(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(gateway.requests, "get", fail)
    monkeypatch.setattr(gateway, "_direct_read", lambda url: f"direct:{url}")
    assert gateway._jina_read("https://example.com") == "direct:https://example.com"


def test_direct_reader_removes_script_content(monkeypatch) -> None:
    class Response:
        text = "<html><body><h1>Evidence</h1><script>secret()</script><p>Grounded text.</p></body></html>"
        headers = {"content-type": "text/html; charset=utf-8"}

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(gateway.requests, "get", lambda *_args, **_kwargs: Response())
    text = gateway._direct_read("https://example.com")
    assert "Evidence" in text
    assert "Grounded text." in text
    assert "secret()" not in text
