from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SightlineDataBuilder/0.1 (research; contact repository maintainer)"
DISALLOWED_PREFIXES = ("list of ", "outline of ", "index of ", "timeline of ")
DISALLOWED_NAMESPACES = ("template:", "category:", "file:", "user:", "help:", "portal:", "wikipedia:")


@dataclass(frozen=True)
class Page:
    title: str
    extract: str
    image_url: str | None
    url: str
    incoming_links: int
    outgoing: tuple[tuple[str, str], ...]
    domain: str
    aliases: tuple[str, ...]


class WikimediaClient:
    def __init__(self, *, seed: int = 3407, hub_cap: int = 10_000, min_incoming: int = 50, timeout: float = 30):
        self.random = random.Random(seed)
        self.hub_cap = hub_cap
        self.min_incoming = min_incoming
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout
        self._page_cache: dict[str, Page | None] = {}

    def _api(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: requests.RequestException | None = None
        for attempt in range(4):
            try:
                response = self.session.get(
                    endpoint,
                    params={"format": "json", "formatversion": 2, **params},
                    timeout=self.timeout,
                )
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < 3:
                    delay = float(response.headers.get("Retry-After", min(2 ** attempt, 8)))
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise RuntimeError(str(payload["error"]))
                return payload
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Wikimedia request failed: {last_error}")

    @staticmethod
    def valid_title(title: str, *, allow_short: bool = False) -> bool:
        lowered = title.casefold().strip()
        if not title or (not allow_short and len(title) < 3):
            return False
        if any(lowered.startswith(prefix) for prefix in DISALLOWED_PREFIXES):
            return False
        if any(lowered.startswith(prefix) for prefix in DISALLOWED_NAMESPACES):
            return False
        return "(disambiguation)" not in lowered

    def random_titles(self, limit: int = 20) -> list[str]:
        try:
            payload = self._api(WIKI_API, {"action": "query", "list": "random", "rnnamespace": 0, "rnlimit": min(limit, 20)})
            titles = [item["title"] for item in payload.get("query", {}).get("random", []) if self.valid_title(item.get("title", ""))]
            if titles:
                return titles
        except requests.HTTPError:
            pass
        titles: list[str] = []
        for _ in range(limit):
            response = self.session.get("https://en.wikipedia.org/wiki/Special:Random", timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            title = response.url.rsplit("/", 1)[-1].replace("_", " ")
            if self.valid_title(title):
                titles.append(title)
        return titles

    def _incoming_count(self, title: str) -> int:
        count = 0
        continuation: dict[str, Any] = {}
        while count <= self.hub_cap:
            payload = self._api(WIKI_API, {"action": "query", "titles": title, "prop": "linkshere", "lhnamespace": 0, "lhlimit": 500, **continuation})
            pages = payload.get("query", {}).get("pages", [])
            links = pages[0].get("linkshere", []) if pages else []
            count += len(links)
            if count > self.hub_cap or "continue" not in payload:
                break
            continuation = payload["continue"]
        return count

    def _outgoing_links(self, title: str, limit: int = 500) -> tuple[tuple[str, str], ...]:
        payload = self._api(WIKI_API, {"action": "parse", "page": title, "prop": "text", "disablelimitreport": 1})
        html = payload.get("parse", {}).get("text", "")
        soup = BeautifulSoup(html, "html.parser")
        links: list[tuple[str, str]] = []
        seen: set[str] = set()
        for anchor in soup.select("a[href^='/wiki/']"):
            href = anchor.get("href", "").split("#", 1)[0]
            title_value = href.removeprefix("/wiki/").replace("_", " ")
            if not self.valid_title(title_value) or title_value in seen:
                continue
            seen.add(title_value)
            relation = " ".join(anchor.get_text(" ", strip=True).split())[:160] or "linked from the article"
            links.append((title_value, relation))
            if len(links) >= limit:
                break
        return tuple(links)

    def page(self, title: str, *, require_image: bool = False) -> Page | None:
        if title in self._page_cache:
            return self._page_cache[title]
        if not self.valid_title(title):
            self._page_cache[title] = None
            return None
        try:
            payload = self._api(WIKI_API, {
                "action": "query", "titles": title, "redirects": 1,
                "prop": "extracts|pageimages|pageprops|info|templates", "explaintext": 1,
                "tlnamespace": 10, "tllimit": 50, "rdnamespace": 0, "rdlimit": 20,
                "inprop": "url", "piprop": "thumbnail|original", "pithumbsize": 768,
            })
            item = (payload.get("query", {}).get("pages") or [{}])[0]
            if item.get("missing") or "disambiguation" in (item.get("pageprops") or {}):
                self._page_cache[title] = None
                return None
            extract = " ".join(str(item.get("extract", "")).split())
            image = (item.get("original") or item.get("thumbnail") or {}).get("source")
            if len(extract) < 200 or (require_image and not image):
                self._page_cache[title] = None
                return None
            template_titles = [
                str(template.get("title", "")).casefold()
                for template in item.get("templates", [])
                if isinstance(template, dict)
            ]
            domain = self._infer_domain(extract, template_titles)
            has_infobox = any("infobox" in template for template in template_titles)
            if require_image and not has_infobox:
                self._page_cache[title] = None
                return None
            canonical = item.get("title", title)
            aliases = tuple(
                str(redirect.get("title", "")).strip()
                for redirect in payload.get("query", {}).get("redirects", [])
                if isinstance(redirect, dict) and redirect.get("title")
            )
            page = Page(
                title=canonical,
                extract=extract[:30_000],
                image_url=image,
                url=item.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(canonical.replace(' ', '_'))}",
                incoming_links=self._incoming_count(canonical),
                outgoing=self._outgoing_links(canonical),
                domain=domain,
                aliases=aliases,
            )
        except (requests.RequestException, RuntimeError, KeyError, IndexError):
            page = None
        self._page_cache[title] = page
        return page

    def commons_images(self, title: str, limit: int = 8) -> list[dict[str, Any]]:
        payload = self._api(COMMONS_API, {
            "action": "query", "generator": "search", "gsrsearch": title,
            "gsrnamespace": 6, "gsrlimit": min(limit * 3, 50),
            "prop": "imageinfo", "iiprop": "url|mime|size", "iiurlwidth": 768,
        })
        results: list[dict[str, Any]] = []
        for item in (payload.get("query", {}).get("pages") or []):
            info = (item.get("imageinfo") or [{}])[0]
            mime = str(info.get("mime", ""))
            width, height = int(info.get("width", 0) or 0), int(info.get("height", 0) or 0)
            url = info.get("thumburl") or info.get("url")
            if url and mime.startswith("image/") and width >= 512 and height >= 512:
                results.append({"title": item.get("title", ""), "url": url, "width": width, "height": height, "mime": mime})
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _infer_domain(extract: str, template_titles: list[str]) -> str:
        text = f"{' '.join(template_titles)} {extract[:1800]}".casefold()
        if any(token in text for token in ("person", "biography", "birth date", "occupation", "born ")):
            return "person"
        if any(token in text for token in ("building", "structure", "architecture", "museum", "church", "stadium", "zoo")):
            return "building_place"
        if any(token in text for token in ("animal", "species", "taxobox", "plant", "fungus", "genus")):
            return "organism"
        if any(token in text for token in ("location", "place", "settlement", "geography", "coordinates", "country")):
            return "location"
        return "artifact"

    def sample_path(self, *, attempts: int = 10, min_hops: int = 2, max_hops: int = 4, domain: str | None = None) -> tuple[Page, list[tuple[Page, str]], Page] | None:
        for _ in range(attempts):
            titles = self.random_titles(20)
            self.random.shuffle(titles)
            seed = next(
                (
                    page
                    for title in titles
                    for page in [self.page(title, require_image=True)]
                    if page is not None and (domain is None or page.domain == domain)
                ),
                None,
            )
            if seed is None:
                continue
            sampled = self.sample_path_from_seed(seed, attempts=1, min_hops=min_hops, max_hops=max_hops)
            if sampled:
                return sampled
        return None

    def sample_path_from_seed(self, seed: Page, *, attempts: int = 1, min_hops: int = 2, max_hops: int = 4) -> tuple[Page, list[tuple[Page, str]], Page] | None:
        if not (self.min_incoming <= seed.incoming_links <= self.hub_cap):
            return None
        for _ in range(attempts):
            hop_values = [hop for hop in (2, 3, 4) if min_hops <= hop <= max_hops]
            if not hop_values:
                raise ValueError("min_hops/max_hops must include at least one of 2, 3, or 4")
            weights = [0.4 if hop == 2 else 0.4 if hop == 3 else 0.2 for hop in hop_values]
            hops = self.random.choices(hop_values, weights=weights, k=1)[0]
            current = seed
            visited = {seed.title}
            bridges: list[tuple[Page, str]] = []
            for hop in range(hops):
                candidates = [(candidate, relation) for candidate, relation in current.outgoing if candidate not in visited and self.valid_title(candidate)]
                self.random.shuffle(candidates)
                next_page = None
                relation = "linked from the article"
                for candidate, candidate_relation in candidates[:40]:
                    page = self.page(candidate)
                    if page and page.incoming_links <= self.hub_cap:
                        if page.title in visited:
                            continue
                        next_page, relation = page, candidate_relation
                        break
                if next_page is None:
                    break
                visited.add(next_page.title)
                if hop < hops - 1:
                    bridges.append((next_page, relation))
                current = next_page
            if len(bridges) + 1 == hops:
                return seed, bridges, current
        return None
