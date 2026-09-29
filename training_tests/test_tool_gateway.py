from __future__ import annotations

import importlib.util
from pathlib import Path

import requests


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("tool_gateway", ROOT / "agent/services/tool_gateway.py")
gateway = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gateway)


def test_explicit_serper_reader_skips_unreachable_jina(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "PAGE_READER_PROVIDER", "serper")
    monkeypatch.setattr(gateway, "_serper_scrape", lambda url: f"scraped {url}")
    monkeypatch.setattr(gateway, "_jina_only", lambda url: (_ for _ in ()).throw(AssertionError("Jina must not run")))
    assert gateway._read_page("https://example.com") == "scraped https://example.com"


def test_auto_reader_falls_back_from_jina_to_serper(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "PAGE_READER_PROVIDER", "auto")
    monkeypatch.setattr(gateway, "_jina_only", lambda url: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    monkeypatch.setattr(gateway, "_serper_scrape", lambda url: "fallback body")
    assert gateway._read_page("https://example.com") == "fallback body"
