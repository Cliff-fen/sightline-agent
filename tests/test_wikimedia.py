from __future__ import annotations

import requests

from data.generation.wikimedia import WikimediaClient


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"query": {"ok": True}}


def test_wikimedia_api_retries_transient_network_failure(monkeypatch) -> None:
    client = WikimediaClient(timeout=0.1)
    calls = 0

    def get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.exceptions.SSLError("temporary EOF")
        return _Response()

    monkeypatch.setattr(client.session, "get", get)
    monkeypatch.setattr("data.generation.wikimedia.time.sleep", lambda _: None)

    assert client._api("https://example.test", {}) == {"query": {"ok": True}}
    assert calls == 2
