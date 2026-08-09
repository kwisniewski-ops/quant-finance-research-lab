"""Regression checks for the lab's public information architecture and voice."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"
PUBLIC_HTML_NAME = re.compile(r"^[a-z0-9-]+\.html$")
PUBLIC_PAGES = sorted(
    page for page in FRONTEND.glob("*.html") if PUBLIC_HTML_NAME.fullmatch(page.name)
)
FINDING_PAGES = sorted(FRONTEND.glob("finding-*.html"))
DOSSIER = FRONTEND / "stock-bond-regime-dossier.html"
BRAND_NAME = "Quantitative Markets & Institutions Lab"
BRAND_NAME_HTML = "Quantitative Markets &amp; Institutions Lab"
OLD_BRAND_NAME = "Quantitative Markets Research Lab"
SOCIAL_IMAGE_URL = "https://www.kylewisniewski.com/lab/og-lab-v3.png"
SOCIAL_IMAGE_ALT_HTML = (
    "Quantitative Markets &amp; Institutions Lab: evidence for decisions under uncertainty, "
    "shown beside a market chart and open research ledger."
)
REGIME_LENSES = {
    "finding-diversification-regimes.html": "#dossier-diversification",
    "finding-portfolio-estimation.html": "#dossier-allocation",
    "finding-var-backtest.html": "#dossier-risk",
}


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
    for page in PUBLIC_PAGES:
        source = page.read_text(encoding="utf-8")
        nav = re.search(r'<nav class="site-nav".*?</nav>', source, flags=re.DOTALL)
        assert nav, page.name
        labels = re.findall(r">([^<]+)</a>", nav.group(0))
        assert labels == expected, page.name


def test_public_voice_is_institutional_and_impersonal() -> None:
    personal_pronouns = re.compile(r"\bI\b|(?i:\b(?:we|our|ours|my|mine|me)\b)")
    for page in PUBLIC_PAGES:
        source = page.read_text(encoding="utf-8")
        text = visible_text(source)
        assert "Kyle" not in source, page.name
        assert "Wisniewski" not in source, page.name
        assert not personal_pronouns.search(text), page.name
        assert "How I work" not in source, page.name
        assert "Professional relevance" not in source, page.name


def test_approved_lab_identity_is_consistent_across_public_surfaces() -> None:
    expected_wordmark = (
        'Quantitative Markets &amp; Institutions <span class="tail">Lab</span>'
    )
    for page in PUBLIC_PAGES:
        source = page.read_text(encoding="utf-8")
        assert OLD_BRAND_NAME not in source, page.name
        assert expected_wordmark in source, page.name
        assert f"{BRAND_NAME_HTML} · 2026" in source, page.name
        if page.name != "404.html":
            assert f'<meta property="og:site_name" content="{BRAND_NAME_HTML}">' in source, page.name
            assert f'<meta property="og:image" content="{SOCIAL_IMAGE_URL}">' in source, page.name
            assert f'<meta name="twitter:image" content="{SOCIAL_IMAGE_URL}">' in source, page.name
            assert f'<meta property="og:image:alt" content="{SOCIAL_IMAGE_ALT_HTML}">' in source, page.name
            assert f'<meta name="twitter:image:alt" content="{SOCIAL_IMAGE_ALT_HTML}">' in source, page.name

    image = FRONTEND / "og-lab-v3.png"
    header = image.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    assert int.from_bytes(header[16:20], "big") == 1200
    assert int.from_bytes(header[20:24], "big") == 630

    supporting_surfaces = (
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        FRONTEND / "feed.xml",
        FRONTEND / "js" / "reading.js",
    )
    for path in supporting_surfaces:
        source = path.read_text(encoding="utf-8")
        assert BRAND_NAME in source or BRAND_NAME_HTML in source, path.name
        assert OLD_BRAND_NAME not in source, path.name


def test_lab_mission_is_stated_at_the_front_door_and_about_page() -> None:
    for page_name in ("index.html", "about.html"):
        source = visible_text((FRONTEND / page_name).read_text(encoding="utf-8"))
        assert "financial models" in source, page_name
        assert "market regimes" in source, page_name
        assert "institutions" in source, page_name
        assert "investor decisions" in source, page_name


def test_formal_lab_urls_use_the_main_sites_canonical_www_host() -> None:
    checked = [*PUBLIC_PAGES, FRONTEND / "feed.xml", ROOT / "README.md", ROOT / "DEPLOY.md"]
    for path in checked:
        source = path.read_text(encoding="utf-8")
        assert "https://kylewisniewski.com" not in source, path.name


def test_indexable_metadata_is_complete_and_structured_data_is_parseable() -> None:
    for page in (page for page in PUBLIC_PAGES if page.name != "404.html"):
        source = page.read_text(encoding="utf-8")
        for marker in (
            "<title>",
            '<meta name="description"',
            '<link rel="canonical"',
            '<meta property="og:url"',
            '<meta property="og:image"',
            '<meta name="twitter:image"',
        ):
            assert source.count(marker) == 1, f"{page.name}: {marker}"

        canonical = re.search(r'<link rel="canonical" href="([^"]+)">', source)
        open_graph = re.search(r'<meta property="og:url" content="([^"]+)">', source)
        assert canonical and open_graph
        assert canonical.group(1) == open_graph.group(1), page.name

        schemas = re.findall(
            r'<script(?: id="[^"]+")? type="application/ld\+json">(.*?)</script>',
            source,
            flags=re.DOTALL,
        )
        assert schemas, page.name
        for schema in schemas:
            json.loads(schema)


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


def test_regime_dossier_is_a_questions_synthesis_not_a_seventh_brief() -> None:
    source = DOSSIER.read_text(encoding="utf-8")
    assert '<nav class="site-nav" aria-label="Primary navigation"><a href="index.html">Start</a><a href="findings.html" aria-current="page">Questions</a>' in source
    assert 'class="depth-nav"' not in source
    assert '<link rel="canonical" href="https://www.kylewisniewski.com/lab/stock-bond-regime-dossier.html">' in source
    assert '"@type": "TechArticle"' in source
    assert '"author": {"@id": "https://www.kylewisniewski.com/lab#organization"}' in source
    assert '"mainEntityOfPage": "https://www.kylewisniewski.com/lab/stock-bond-regime-dossier.html"' in source
    assert '"datePublished": "2026-08-09"' in source
    assert '"dateModified": "2026-08-09"' in source
    assert source.index('<script src="js/common.js"></script>') < source.index(
        '<script src="js/reading.js"></script>'
    )

    section_ids = [
        "dossier-observation",
        "dossier-regime-map",
        "dossier-diversification",
        "dossier-allocation",
        "dossier-risk",
        "dossier-limits",
        "dossier-sources",
    ]
    for section_id in section_ids:
        assert f'id="{section_id}"' in source
        assert f'href="#{section_id}"' in source


def test_only_three_findings_receive_one_bounded_regime_lens() -> None:
    total = 0
    for page in FINDING_PAGES:
        source = page.read_text(encoding="utf-8")
        count = source.count('class="regime-lens"')
        total += count
        if page.name in REGIME_LENSES:
            assert count == 1, page.name
            lens_index = source.index('class="regime-lens"')
            assert source.index('id="explore"') < lens_index < source.index('id="related"')
            assert f'stock-bond-regime-dossier.html{REGIME_LENSES[page.name]}' in source
            assert "Measured here" in source[lens_index:]
            assert "Established context" in source[lens_index:]
            assert "Interpretive implication" in source[lens_index:]
            assert "Causal boundary:" in source[lens_index:]
            assert 'data-analytics-event="research_engagement"' in source[lens_index:]
        else:
            assert count == 0, page.name
    assert total == 3


def test_regime_dossier_is_discoverable_and_returns_to_the_record() -> None:
    for page_name in ("index.html", "findings.html"):
        source = (FRONTEND / page_name).read_text(encoding="utf-8")
        assert 'href="stock-bond-regime-dossier.html"' in source

    source = DOSSIER.read_text(encoding="utf-8")
    for href in (
        "finding-diversification-regimes.html",
        "finding-portfolio-estimation.html",
        "finding-var-backtest.html",
        "research-log.html#portfolio-construction",
        "research-log.html#market-risk",
    ):
        assert f'href="{href}"' in source

    record = (FRONTEND / "research-log.html").read_text(encoding="utf-8")
    assert record.count("stock-bond-regime-dossier.html") == 2


def test_research_feed_includes_the_regime_dossier_once() -> None:
    root = ET.parse(FRONTEND / "feed.xml").getroot()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    assert root.findtext("atom:title", namespaces=namespace) == f"{BRAND_NAME} — Research"
    assert root.findtext("atom:author/atom:name", namespaces=namespace) == BRAND_NAME
    alternate = root.find("atom:link[@rel='alternate']", namespace)
    assert alternate is not None
    assert alternate.attrib["href"] == "https://www.kylewisniewski.com/lab/"

    dossier_id = "https://www.kylewisniewski.com/lab/stock-bond-regime-dossier.html"
    entries = root.findall("atom:entry", namespace)
    matching = [entry for entry in entries if entry.findtext("atom:id", namespaces=namespace) == dossier_id]
    assert len(matching) == 1
    dossier_updated = matching[0].findtext("atom:updated", namespaces=namespace)
    feed_updated = root.findtext("atom:updated", namespaces=namespace)
    assert dossier_updated is not None
    assert feed_updated is not None
    assert feed_updated >= dossier_updated


def test_research_feed_discovery_and_revision_dates_are_consistent() -> None:
    for page in PUBLIC_PAGES:
        source = page.read_text(encoding="utf-8")
        href = "/lab/feed.xml" if page.name == "404.html" else "feed.xml"
        assert f'title="{BRAND_NAME_HTML} research" href="{href}"' in source

    revised_pages = (
        "index.html",
        "findings.html",
        "finding-diversification-regimes.html",
        "finding-portfolio-estimation.html",
        "finding-var-backtest.html",
        "research-log.html",
    )
    for page_name in revised_pages:
        source = (FRONTEND / page_name).read_text(encoding="utf-8")
        assert '"dateModified": "2026-08-09"' in source, page_name

    root = ET.parse(FRONTEND / "feed.xml").getroot()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    revised_urls = {
        f"https://www.kylewisniewski.com/lab/{page_name}"
        for page_name in (
            "finding-diversification-regimes.html",
            "finding-portfolio-estimation.html",
            "finding-var-backtest.html",
        )
    }
    for entry in root.findall("atom:entry", namespace):
        if entry.findtext("atom:id", namespaces=namespace) in revised_urls:
            updated = entry.findtext("atom:updated", namespaces=namespace)
            assert updated is not None
            assert updated.startswith("2026-08-09")


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


def test_citation_tool_uses_the_declared_canonical_url() -> None:
    source = (FRONTEND / "js" / "reading.js").read_text(encoding="utf-8")
    assert "document.querySelector('link[rel=\"canonical\"]')" in source
    assert 'window.location.href.split("#")' not in source


def test_internal_frontend_links_resolve() -> None:
    for page in PUBLIC_PAGES:
        source = page.read_text(encoding="utf-8")
        for href in re.findall(r'href="([^"]+)"', source):
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or href.startswith(("mailto:", "data:")):
                continue
            path = unquote(parsed.path)
            if path.startswith("/lab/"):
                target = (FRONTEND / path.removeprefix("/lab/")).resolve()
            elif path == "/lab":
                target = (FRONTEND / "index.html").resolve()
            else:
                target = page if not path else (page.parent / path).resolve()
            assert target.exists(), f"{page.name}: missing {href}"
            if parsed.fragment and target.suffix == ".html":
                target_source = target.read_text(encoding="utf-8")
                fragment = re.escape(unquote(parsed.fragment))
                assert re.search(rf'\bid=["\']{fragment}["\']', target_source), (
                    f"{page.name}: missing fragment {href}"
                )
