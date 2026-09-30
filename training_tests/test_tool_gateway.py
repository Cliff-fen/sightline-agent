from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("tool_gateway", ROOT / "agent/services/tool_gateway.py")
gateway = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gateway)


@pytest.fixture(autouse=True)
def clear_page_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "TOOL_CACHE_PATH", tmp_path / "tool-cache.sqlite3")
    gateway._read_page.cache_clear()
    yield
    gateway._read_page.cache_clear()


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


def test_page_reader_caches_successful_content(monkeypatch) -> None:
    calls = 0

    def read(url: str) -> str:
        nonlocal calls
        calls += 1
        return f"body for {url}"

    monkeypatch.setattr(gateway, "PAGE_READER_PROVIDER", "jina")
    monkeypatch.setattr(gateway, "_jina_only", read)
    assert gateway._read_page("https://example.com/cached") == "body for https://example.com/cached"
    assert gateway._read_page("https://example.com/cached") == "body for https://example.com/cached"
    assert calls == 1


def test_tool_cache_survives_memory_cache_clear(monkeypatch) -> None:
    calls = 0

    def read(url: str) -> str:
        nonlocal calls
        calls += 1
        return "persistent body"

    monkeypatch.setattr(gateway, "PAGE_READER_PROVIDER", "jina")
    monkeypatch.setattr(gateway, "_jina_only", read)
    assert gateway._read_page("https://example.com/persistent") == "persistent body"
    gateway._read_page.cache_clear()
    assert gateway._read_page("https://example.com/persistent") == "persistent body"
    assert calls == 1


def test_search_falls_back_to_bing_when_serper_credits_are_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "BING_RSS_URL", "https://www.bing.com/search")
    monkeypatch.setattr(
        gateway,
        "_serper_search",
        lambda query, top_k: (_ for _ in ()).throw(
            gateway.ToolFailure("SERPER_CREDITS_EXHAUSTED", "credits exhausted")
        ),
    )
    monkeypatch.setattr(
        gateway,
        "_bing_search",
        lambda query, top_k: [{"title": "Fallback", "link": "https://example.com", "snippet": query}],
    )
    assert gateway._search("evidence query", 3)[0]["title"] == "Fallback"


def test_search_fails_closed_without_explicit_fallback(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "BING_RSS_URL", "")
    monkeypatch.setattr(
        gateway,
        "_serper_search",
        lambda query, top_k: (_ for _ in ()).throw(
            gateway.ToolFailure("SERPER_CREDITS_EXHAUSTED", "credits exhausted")
        ),
    )
    monkeypatch.setattr(
        gateway,
        "_bing_search",
        lambda query, top_k: (_ for _ in ()).throw(AssertionError("Bing must not run")),
    )
    try:
        gateway._search("evidence query", 3)
    except gateway.ToolFailure as exc:
        assert exc.code == "SERPER_CREDITS_EXHAUSTED"
    else:
        raise AssertionError("search must propagate Serper exhaustion")


def test_lens_empty_match_list_is_a_valid_result(monkeypatch) -> None:
    class Response:
        ok = True

        @staticmethod
        def json():
            return {"organic": [], "credits": 1}

    monkeypatch.setattr(gateway, "SERPER_API_KEY", "configured")
    monkeypatch.setattr(gateway.requests, "post", lambda *args, **kwargs: Response())
    payload = gateway.json.loads(
        gateway._serper_lens("https://example.com/image.png", include_source_url=True)
    )
    assert payload["provider"] == "serper_lens"
    assert payload["matches"] == []
