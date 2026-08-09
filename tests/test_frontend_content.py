"""Regression checks for the lab's public information architecture and voice."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"
FINDING_PAGES = sorted(FRONTEND.glob("finding-*.html"))


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def visible_text(source: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(source)
    return " ".join(parser.parts)


def test_global_navigation_is_reduced_to_four_reader_routes() -> None:
    expected = ["Start", "Questions", "Models", "Method"]
    for page in sorted(FRONTEND.glob("*.html")):
        source = page.read_text(encoding="utf-8")
        nav = re.search(r'<nav class="site-nav".*?</nav>', source, flags=re.DOTALL)
        assert nav, page.name
        labels = re.findall(r">([^<]+)</a>", nav.group(0))
        assert labels == expected, page.name


def test_public_voice_is_institutional_and_impersonal() -> None:
    personal_pronouns = re.compile(r"\bI\b|(?i:\b(?:we|our|ours|my|mine|me)\b)")
    for page in sorted(FRONTEND.glob("*.html")):
        source = page.read_text(encoding="utf-8")
        text = visible_text(source)
        assert "Kyle" not in source, page.name
        assert "Wisniewski" not in source, page.name
        assert not personal_pronouns.search(text), page.name
        assert "How I work" not in source, page.name
        assert "Professional relevance" not in source, page.name


def test_question_hub_leads_with_six_plain_language_answers() -> None:
    source = (FRONTEND / "findings.html").read_text(encoding="utf-8")
    assert source.count('class="finding-answer"') == 6
    assert "Six questions investors should ask quantitative models" in source


def test_every_finding_uses_the_same_progressive_depth_path() -> None:
    anchors = ["#bottom-line", "#evidence", "#practice", "#limitations", "#explore"]
    labels = ["Answer", "Evidence", "Application", "Limits", "Method and code"]
    assert len(FINDING_PAGES) == 6
    for page in FINDING_PAGES:
        source = page.read_text(encoding="utf-8")
        depth_nav = re.search(r'<nav class="depth-nav".*?</nav>', source, flags=re.DOTALL)
        assert depth_nav, page.name
        for anchor, label in zip(anchors, labels, strict=True):
            assert f'href="{anchor}"' in depth_nav.group(0), page.name
            assert f"<strong>{label}</strong>" in depth_nav.group(0), page.name


def test_lab_palette_uses_the_parent_brand_foundation() -> None:
    source = (FRONTEND / "css" / "main.css").read_text(encoding="utf-8").lower()
    expected_tokens = {
        "--paper": "#faf8f4",
        "--paper-deep": "#f3f0e9",
        "--ink": "#1c2430",
        "--navy": "#14202e",
        "--muted": "#5c6672",
        "--rule": "#e4dfd3",
        "--accent": "#7e6238",
        "--accent-dim": "#9c7c46",
    }
    for token, value in expected_tokens.items():
        assert re.search(rf"{re.escape(token)}:\s*{value}", source)


def test_internal_frontend_links_resolve() -> None:
    for page in sorted(FRONTEND.glob("*.html")):
        source = page.read_text(encoding="utf-8")
        for href in re.findall(r'href="([^"]+)"', source):
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or href.startswith(("mailto:", "data:")):
                continue
            target = page if not parsed.path else (page.parent / unquote(parsed.path)).resolve()
            assert target.exists(), f"{page.name}: missing {href}"
            if parsed.fragment and target.suffix == ".html":
                target_source = target.read_text(encoding="utf-8")
                fragment = re.escape(unquote(parsed.fragment))
                assert re.search(rf'\bid=["\']{fragment}["\']', target_source), (
                    f"{page.name}: missing fragment {href}"
                )
