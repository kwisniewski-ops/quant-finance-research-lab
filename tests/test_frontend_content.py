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
ATLAS = FRONTEND / "regime-atlas.html"
BRAND_NAME = "Quantitative Markets & Institutions Lab"
BRAND_NAME_HTML = "Quantitative Markets &amp; Institutions Lab"
OLD_BRAND_NAME = "Quantitative Markets Research Lab"
SOCIAL_IMAGE_URL = "https://www.kylewisniewski.com/lab/og-lab-v5.png"
SOCIAL_IMAGE_ALT_HTML = (
    "Quantitative Markets &amp; Institutions Lab: a seeded Monte Carlo path fan "
    "from the lab&#x27;s simulation research beside the lab wordmark."
)
REGIME_LENSES = {
    "finding-diversification-regimes.html": "#dossier-diversification",
    "finding-portfolio-estimation.html": "#dossier-allocation",
    "finding-var-backtest.html": "#dossier-risk",
}
REGIME_LENS_CONTEXT_LABELS = {
    "finding-diversification-regimes.html": "External empirical evidence",
    "finding-portfolio-estimation.html": "Institutional record",
    "finding-var-backtest.html": "Institutional record",
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
    return " ".join(" ".join(parser.parts).split())


def test_global_navigation_exposes_the_research_program_and_record() -> None:
    expected = ["Overview", "Research", "Models", "Methods", "Record"]
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

    image = FRONTEND / "og-lab-v5.png"
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


def test_research_hub_maps_seven_results_to_six_investigations() -> None:
    source = (FRONTEND / "findings.html").read_text(encoding="utf-8")
    assert source.count('class="finding-answer"') == 7
    assert "Quantitative investigations of model risk, market regimes, and institutional transmission" in source
    assert source.count("Empirical finding") == 5
    assert source.count("Numerical validation") >= 2
    assert "Seven published results from six executed investigations" in source
    assert "market-risk investigation supports separate public findings" in source
    assert "Six questions investors should ask quantitative models" not in source


def test_public_titles_are_declarative_and_consumer_wrappers_are_absent() -> None:
    prohibited = (
        "Questions before models",
        "plain-language",
        "Simple first",
        "Short answer",
        "Start with a question",
        "Start simpler",
        "Start narrower",
        "Continue exploring",
        "Follow the answer",
        "Stop when the answer is sufficient",
        "Read only as far as the decision requires",
    )
    for page in PUBLIC_PAGES:
        source = page.read_text(encoding="utf-8")
        text = visible_text(source)
        assert not any(phrase.casefold() in text.casefold() for phrase in prohibited), page.name
        assert "Decision brief" not in text, page.name
        assert '"proficiencyLevel"' not in source, page.name
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", source, flags=re.DOTALL)
        assert h1, page.name
        assert "?" not in visible_text(h1.group(0)), page.name
        if page.name != "404.html":
            title = re.search(r"<title>(.*?)</title>", source, flags=re.DOTALL)
            social_title = re.search(r'<meta property="og:title" content="([^"]+)">', source)
            assert title and "?" not in visible_text(title.group(0)), page.name
            assert social_title and "?" not in social_title.group(1), page.name


def test_feed_and_breadcrumbs_preserve_the_declarative_taxonomy() -> None:
    root = ET.parse(FRONTEND / "feed.xml").getroot()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", namespace):
        title = entry.findtext("atom:title", namespaces=namespace)
        assert title and "?" not in title

    for page in (page for page in PUBLIC_PAGES if page.name not in {"404.html", "index.html"}):
        source = page.read_text(encoding="utf-8")
        breadcrumb_match = re.search(
            r'<script id="breadcrumb-schema" type="application/ld\+json">(.*?)</script>',
            source,
            flags=re.DOTALL,
        )
        assert breadcrumb_match, page.name
        breadcrumb = json.loads(breadcrumb_match.group(1))
        names = [item["name"] for item in breadcrumb["itemListElement"]]
        assert "Questions" not in names, page.name
        assert not any("?" in name for name in names), page.name


def test_every_finding_uses_the_same_progressive_depth_path() -> None:
    anchors = ["#bottom-line", "#evidence", "#practice", "#limitations", "#explore"]
    labels = ["Conclusion", "Evidence", "Decision consequences", "Identification and limits", "Method, mathematics, and code"]
    assert len(FINDING_PAGES) == 6
    for page in FINDING_PAGES:
        source = page.read_text(encoding="utf-8")
        depth_nav = re.search(r'<nav class="depth-nav".*?</nav>', source, flags=re.DOTALL)
        assert depth_nav, page.name
        for anchor, label in zip(anchors, labels, strict=True):
            assert f'href="{anchor}"' in depth_nav.group(0), page.name
            assert f"<strong>{label}</strong>" in depth_nav.group(0), page.name


def test_regime_dossier_is_a_research_synthesis_not_a_seventh_brief() -> None:
    source = DOSSIER.read_text(encoding="utf-8")
    assert '<nav class="site-nav" aria-label="Primary navigation"><a href="index.html">Overview</a><a href="findings.html" aria-current="page">Research</a>' in source
    assert 'class="depth-nav"' not in source
    assert '<link rel="canonical" href="https://www.kylewisniewski.com/lab/stock-bond-regime-dossier.html">' in source
    assert '"@type": "TechArticle"' in source
    assert '"author": {"@id": "https://www.kylewisniewski.com/lab#organization"}' in source
    assert '"mainEntityOfPage": "https://www.kylewisniewski.com/lab/stock-bond-regime-dossier.html"' in source
    assert '"datePublished": "2026-08-09"' in source
    assert '"dateModified": "2026-08-10"' in source
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
            assert "Lab measurement" in source[lens_index:]
            assert REGIME_LENS_CONTEXT_LABELS[page.name] in source[lens_index:]
            assert "Interpretive synthesis" in source[lens_index:]
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
    record_main = re.search(r"<main.*?</main>", record, flags=re.DOTALL)
    assert record_main
    assert record_main.group(0).count("stock-bond-regime-dossier.html") == 2


def test_regime_atlas_is_a_bounded_research_framework() -> None:
    source = ATLAS.read_text(encoding="utf-8")
    assert '<link rel="canonical" href="https://www.kylewisniewski.com/lab/regime-atlas.html">' in source
    assert '<nav class="site-nav" aria-label="Primary navigation"><a href="index.html">Overview</a><a href="findings.html" aria-current="page">Research</a>' in source
    assert 'class="depth-nav"' not in source
    assert ATLAS not in FINDING_PAGES

    schemas = re.findall(
        r'<script(?: id="[^"]+")? type="application/ld\+json">(.*?)</script>',
        source,
        flags=re.DOTALL,
    )
    parsed = [json.loads(schema) for schema in schemas]
    article = next(schema for schema in parsed if schema.get("@type") == "TechArticle")
    assert article["author"] == {"@id": "https://www.kylewisniewski.com/lab#organization"}
    assert article["datePublished"] == "2026-08-09"
    assert article["dateModified"] == "2026-08-10"
    breadcrumb = next(schema for schema in parsed if schema.get("@type") == "BreadcrumbList")
    assert [item["name"] for item in breadcrumb["itemListElement"][:2]] == [
        BRAND_NAME,
        "Research",
    ]

    for phrase in (
        "without classifying the present",
        "Scenario, not forecast",
        "No live data",
        "Forecast status",
        "No present-state classification",
    ):
        assert phrase in source
    prohibited = re.compile(r"(?i)\b(?:the )?(?:current|present) regime is\b")
    assert not prohibited.search(visible_text(source))


def test_regime_atlas_progressively_enhances_static_scenarios() -> None:
    source = ATLAS.read_text(encoding="utf-8")
    state_controls = re.findall(
        r'<button[^>]+data-atlas-state="([^"]+)"[^>]+aria-controls="([^"]+)"[^>]+aria-pressed="(?:true|false)"',
        source,
    )
    expected_states = {
        "unclassified",
        "weak-build",
        "strong-build",
        "weak-ease",
        "strong-ease",
    }
    assert {state for state, _ in state_controls} == expected_states
    assert len(state_controls) == 5
    assert len(re.findall(r'<button[^>]+data-atlas-state="[^"]+"[^>]+disabled>', source)) == 5

    for state, controlled_id in state_controls:
        assert controlled_id == f"atlas-state-{state}"
        panel = re.search(
            rf'<section class="atlas-output" id="{re.escape(controlled_id)}" data-atlas-panel="{re.escape(state)}">(.*?)</section>',
            source,
            flags=re.DOTALL,
        )
        assert panel, state
        assert "What becomes fragile" in panel.group(1)
        assert "Evidence to inspect" in panel.group(1)

    assert source.count('<details class="episode-card">') == 6
    assert source.count("<strong>What this does not establish:</strong>") == 6
    for label in (
        "Lab measurement",
        "Institutional record",
        "External empirical evidence",
        "Interpretive synthesis",
        "Scenario, not forecast",
        "Proposed research",
    ):
        assert label in source

    for select_id, values in {
        "atlas-liquidity": {"unclassified", "functioning", "tightening", "impaired", "normalizing"},
        "atlas-response": {"unclassified", "routine", "constrained", "backstop", "fragmented"},
    }.items():
        select = re.search(rf'<select id="{select_id}".*?</select>', source, flags=re.DOTALL)
        assert select
        assert select.group(0).split(">", 1)[0].endswith(" disabled")
        assert set(re.findall(r'<option value="([^"]+)">', select.group(0))) == values

    definitions = re.findall(
        r'data-atlas-definition="([^"]+)"[^>]*>.*?data-atlas-definition-text>(.*?)</td>',
        source,
        flags=re.DOTALL,
    )
    assert len(definitions) == 10
    assert {key for key, _ in definitions} == {
        "liquidity:unclassified",
        "liquidity:functioning",
        "liquidity:tightening",
        "liquidity:impaired",
        "liquidity:normalizing",
        "response:unclassified",
        "response:routine",
        "response:constrained",
        "response:backstop",
        "response:fragmented",
    }
    assert all(visible_text(copy).strip() for _, copy in definitions)

    script_order = [
        source.index('src="js/common.js"'),
        source.index('src="js/reading.js"'),
        source.index('src="js/regime-atlas.js"'),
        source.index('src="js/engagement.js"'),
    ]
    assert script_order == sorted(script_order)

    javascript = (FRONTEND / "js" / "regime-atlas.js").read_text(encoding="utf-8")
    assert "history.pushState" in javascript
    assert 'const scenarioHashPrefix = "#scenario-"' in javascript
    assert 'const liquidity =' not in javascript
    assert 'const responses =' not in javascript
    assert "overlayDefinition" in javascript
    assert "button.disabled = false" in javascript
    assert "liquiditySelect.disabled = false" in javascript
    assert "responseSelect.disabled = false" in javascript
    assert "if (!window.location.hash.startsWith(scenarioHashPrefix)) return;" in javascript
    assert 'addEventListener("popstate"' in javascript
    assert "scrollIntoView" not in javascript
    assert ".focus(" not in javascript
    for title in (
        "Activity weakens while inflation pressure builds",
        "Activity strengthens while inflation pressure builds",
        "Activity weakens while inflation pressure eases",
        "Activity strengthens while inflation pressure eases",
    ):
        assert title in source
        assert title not in javascript

    stylesheet = (FRONTEND / "css" / "main.css").read_text(encoding="utf-8")
    assert '[data-atlas-panel][hidden] { display: none; }' in stylesheet


def test_regime_atlas_is_discoverable_across_the_lab() -> None:
    for page_name in (
        "index.html",
        "findings.html",
        "stock-bond-regime-dossier.html",
        "about.html",
        "method.html",
        "research-log.html",
    ):
        source = (FRONTEND / page_name).read_text(encoding="utf-8")
        assert 'href="regime-atlas.html' in source, page_name

    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    findings = (FRONTEND / "findings.html").read_text(encoding="utf-8")
    for source in (index, findings):
        assert 'href="regime-atlas.html"' in source
        assert 'href="stock-bond-regime-dossier.html"' in source
        assert "Historical framework · Scenario design" in source
        assert "Empirical dossier · Measured relationship" in source


def test_regime_frameworks_publish_ledgers_without_inflating_investigations() -> None:
    atlas = ATLAS.read_text(encoding="utf-8")
    dossier = DOSSIER.read_text(encoding="utf-8")
    record = (FRONTEND / "research-log.html").read_text(encoding="utf-8")
    assert 'class="ledger-record atlas-ledger"' in atlas
    assert 'class="ledger-record dossier-ledger"' in dossier
    for field in ("Classification", "Data status", "Forecast status"):
        assert field in atlas
    for field in ("Classification", "Data vintage", "Forecast status"):
        assert field in dossier
    assert 'id="research-pipeline"' in record
    assert record.count('class="status status-proposed">Proposed</span>') >= 5
    assert "Registered program on regime transmission and model failure" in record
    assert "<dt>Investigations</dt><dd>06</dd>" in record
    assert (ROOT / "docs" / "research_pipeline.md").exists()


def test_regime_atlas_has_one_research_feed_entry() -> None:
    root = ET.parse(FRONTEND / "feed.xml").getroot()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    atlas_id = "https://www.kylewisniewski.com/lab/regime-atlas.html"
    matching = [
        entry
        for entry in root.findall("atom:entry", namespace)
        if entry.findtext("atom:id", namespaces=namespace) == atlas_id
    ]
    assert len(matching) == 1
    assert matching[0].findtext("atom:published", namespaces=namespace).startswith("2026-08-09")
    assert matching[0].findtext("atom:updated", namespaces=namespace).startswith("2026-08-10")


def test_measured_periods_are_not_labeled_as_formal_regimes() -> None:
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    findings = (FRONTEND / "findings.html").read_text(encoding="utf-8")
    diversification = (FRONTEND / "finding-diversification-regimes.html").read_text(encoding="utf-8")
    assert "studied regimes" not in index
    assert "studied regimes" not in findings
    assert "2022–2023 inflation shock" not in diversification
    assert "2022–2023 interval" in diversification


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

    for page in (page for page in PUBLIC_PAGES if page.name != "404.html"):
        source = page.read_text(encoding="utf-8")
        assert re.search(r'"dateModified"\s*:\s*"2026-08-10"', source), page.name

    root = ET.parse(FRONTEND / "feed.xml").getroot()
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", namespace):
        updated = entry.findtext("atom:updated", namespaces=namespace)
        assert updated is not None
        assert updated.startswith("2026-08-10")


def test_evidence_classes_statuses_and_corpus_boundaries_are_explicit() -> None:
    method = (FRONTEND / "method.html").read_text(encoding="utf-8")
    for label in (
        "Lab measurement",
        "Numerical validation",
        "External empirical evidence",
        "Institutional record",
        "Interpretive synthesis",
        "Proposed research",
    ):
        assert label in method

    record = (FRONTEND / "research-log.html").read_text(encoding="utf-8")
    assert "Replicated" not in visible_text(record)
    assert "Seeded 05" not in visible_text(record)
    assert '<span class="status status-validated">Validated</span>' in record
    assert "<dt>Numerical</dt><dd>02</dd>" in record

    about = visible_text((FRONTEND / "about.html").read_text(encoding="utf-8"))
    atlas = visible_text(ATLAS.read_text(encoding="utf-8"))
    dossier = visible_text(DOSSIER.read_text(encoding="utf-8"))
    assert "not a live classifier, probability model, or forecast" in about
    assert "without classifying the present or forecasting the next state" in atlas
    assert "No present-state classification, regime probability, or return forecast" in dossier


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
