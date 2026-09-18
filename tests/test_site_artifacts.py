"""Schema regressions for the site artifacts in ``app/frontend/site-artifacts/``.

The artifacts are produced by ``scripts/export_site_artifacts.py`` and consumed
by the personal site (see ``docs/site_artifact_schema.md``). These tests check
the contract, not the numbers; ``scripts/verify_site_artifacts.py`` checks the
numbers against the published pages.
"""

from __future__ import annotations

import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "app" / "frontend" / "site-artifacts"
SCHEMA = "site-artifact/1"
SLUGS = ["portfolio-estimation", "var-backtest", "factor-etfs", "volatility", "monte-carlo", "pricing-validation", "stock-bond-regimes"]
COLOURS = {f"c{i}" for i in range(1, 9)}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_TAGS = {"p", "h3", "ul", "ol", "li", "em", "strong", "a", "code", "table", "thead", "tbody", "tr", "th", "td", "sup", "sub", "span", "br"}
FIGURE_TYPES = {"line", "columns", "bars", "histogram", "dots", "scatter", "heatmap"}
HARD_CAP = 3000

pytestmark = pytest.mark.skipif(not ARTIFACTS.exists(), reason="run scripts/export_site_artifacts.py first")


def _load(name: str) -> dict:
    text = (ARTIFACTS / name).read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text, f"{name} contains NaN/Infinity"
    return json.loads(text)


@pytest.fixture(scope="module")
def artifacts() -> dict[str, dict]:
    return {slug: _load(f"{slug}.json") for slug in SLUGS}


def _walk_numbers(obj, path="$"):
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        assert math.isfinite(obj), f"non-finite number at {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_numbers(v, f"{path}[{i}]")


class _TagAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()
        self.attrs: set[str] = set()
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        self.attrs.update(k for k, _ in attrs)
        if tag != "br":
            self.depth += 1

    def handle_endtag(self, tag):
        if tag != "br":
            self.depth -= 1


def _audit_html(fragment: str) -> None:
    parser = _TagAudit()
    parser.feed(fragment)
    parser.close()
    assert parser.tags <= ALLOWED_TAGS, parser.tags - ALLOWED_TAGS
    assert parser.attrs <= {"href", "class", "colspan"}, parser.attrs
    assert parser.depth == 0, "unbalanced HTML"
    assert ".html" not in re.sub(r"https?://[^\"']+", "", fragment), "internal link not rewritten"


def test_manifest_lists_every_artifact():
    manifest = _load("manifest.json")
    assert manifest["schema_version"] == SCHEMA
    assert ISO_STAMP.match(manifest["generated_at"])
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"])
    assert manifest["artifacts"] == [f"{s}.json" for s in SLUGS]
    for name in manifest["artifacts"]:
        assert (ARTIFACTS / name).exists()


def test_envelope(artifacts):
    for slug, a in artifacts.items():
        assert a["schema_version"] == SCHEMA
        assert a["investigation"] == slug
        for key in ("title", "kicker", "result", "evidence_class"):
            assert isinstance(a[key], str) and a[key], (slug, key)
        prov = a["provenance"]
        assert re.fullmatch(r"[0-9a-f]{40}", prov["source_commit"])
        assert ISO_STAMP.match(prov["generated_at"])
        assert prov["notebook"].startswith("notebooks/") and prov["notebook"].endswith(".ipynb")
        assert prov["notebook_url"].endswith(prov["notebook"]) and prov["source_commit"] in prov["notebook_url"]
        assert len(prov["universe"]) == 15
        for entry in prov["data"]:
            assert SHA256.match(entry["sha256"]) and entry["rows"] > 0
            assert ISO_DATE.match(entry["start"]) and ISO_DATE.match(entry["end"])
        tests = prov["tests"]
        assert tests["suite"] == "tests"
        if tests.get("count") is not None:
            assert tests["passed"] == tests["count"] > 0
        for key in ("key_numbers", "figures", "tables", "math_notes"):
            assert isinstance(a[key], list), (slug, key)
        _walk_numbers(a, slug)


def test_key_numbers(artifacts):
    for slug, a in artifacts.items():
        ids = [k["id"] for k in a["key_numbers"]]
        assert len(ids) == len(set(ids)), f"duplicate key ids in {slug}"
        for k in a["key_numbers"]:
            assert isinstance(k["value"], str) and k["value"], (slug, k["id"])
            assert isinstance(k["label"], str) and k["label"]
            assert isinstance(k["decimals"], int) and k["decimals"] >= 0
            assert isinstance(k["prefix"], str) and isinstance(k["suffix"], str)
            if "count" in k:
                assert isinstance(k["count"], (int, float)) and not isinstance(k["count"], bool)
            if k.get("unverified"):
                assert k.get("note"), f"unverified {slug}:{k['id']} needs a note"


def test_figures(artifacts):
    for slug, a in artifacts.items():
        ids = [f["id"] for f in a["figures"]]
        assert len(ids) == len(set(ids)), f"duplicate figure ids in {slug}"
        for fig in a["figures"]:
            assert fig["type"] in FIGURE_TYPES, (slug, fig["id"], fig["type"])
            for key in ("title", "sub", "alt"):
                assert isinstance(fig[key], str) and fig[key], (slug, fig["id"], key)
            if fig["type"] == "scatter":
                assert 0 < len(fig["points"]) <= 400
                for pt in fig["points"]:
                    assert isinstance(pt["x"], (int, float)) and isinstance(pt["y"], (int, float))
                    if "color" in pt:
                        assert pt["color"] in COLOURS
                assert fig["x"]["label"] and fig["y"]["label"]
            elif fig["type"] == "heatmap":
                grid = fig["grid"]
                assert len(grid["values"]) == len(grid["rows"])
                assert all(len(row) == len(grid["cols"]) for row in grid["values"])
            elif fig["type"] == "bars":
                for item in fig["items"]:
                    assert item["color"] in COLOURS
            else:
                n = len(fig["x"]["values"])
                assert fig["x"]["label"]
                assert 0 < n <= HARD_CAP, (slug, fig["id"], n)
                for s in fig["series"]:
                    assert s["color"] in COLOURS, (slug, fig["id"], s["key"])
                    assert len(s["values"]) == n, (slug, fig["id"], s["key"])
                    if "band" in s:
                        assert len(s["band"]["lo"]) == n and len(s["band"]["hi"]) == n
                if "overlay" in fig:
                    assert fig["overlay"]["color"] in COLOURS and len(fig["overlay"]["values"]) == n
                if fig["type"] == "line" and fig["x"]["values"] and isinstance(fig["x"]["values"][0], str):
                    assert all(ISO_DATE.match(v) for v in fig["x"]["values"]), (slug, fig["id"])
            for rule in fig.get("rules", []):
                assert rule["axis"] in {"x", "y"} and rule["color"] in COLOURS and rule["label"]
                assert isinstance(rule["at"], (int, float)) or ISO_DATE.match(rule["at"])
            for ann in fig.get("annotations", []):
                assert ann["text"] and ann["x"] is not None and ann["y"] is not None
            if "table" in fig:
                assert fig["table"]["columns"] and all(len(r) == len(fig["table"]["columns"]) for r in fig["table"]["rows"])


def test_tables(artifacts):
    for slug, a in artifacts.items():
        ids = [t["id"] for t in a["tables"]]
        assert len(ids) == len(set(ids)), f"duplicate table ids in {slug}"
        for t in a["tables"]:
            assert t["title"] and t["columns"] and t["rows"], (slug, t["id"])
            for row in t["rows"]:
                assert len(row) == len(t["columns"]), (slug, t["id"], row)


def test_prose_is_sanitised(artifacts):
    for slug, a in artifacts.items():
        prose = a["prose"]
        assert prose["decision_summary_html"].startswith("<")
        _audit_html(prose["decision_summary_html"])
        _audit_html(prose["limitations_html"])
        assert prose["sections"], slug
        for section in prose["sections"]:
            assert section["id"] and section["heading"] and section["html"], (slug, section.get("id"))
            _audit_html(section["html"])
        for n, source in enumerate(prose["sources"], start=1):
            assert source["n"] == n and source["title"], (slug, source)
            assert source["url"] is None or source["url"].startswith("http")


def test_math_notes_and_attachment(artifacts):
    attached = {slug: [n["id"] for n in a["math_notes"]] for slug, a in artifacts.items()}
    assert attached["volatility"] == ["ito", "gbm", "risk-neutral-pricing", "feynman-kac"]
    assert attached["monte-carlo"] == ["monte-carlo-error"]
    assert attached["var-backtest"] == ["fat-tail-var"]
    for a in artifacts.values():
        for note in a["math_notes"]:
            assert note["heading"] and "$" in note["markdown"], note["id"]
            assert note["tex_source"] == "katex"
            assert "<" not in re.sub(r"\$[^$]*\$", "", note["markdown"].replace("$$", "$")), "HTML leaked into markdown"


def test_ledger():
    ledger = _load("ledger.json")
    assert ledger["schema_version"] == SCHEMA
    entries = ledger["entries"]
    assert [e["investigation"] for e in entries] == ["pricing-validation", "portfolio-estimation", "volatility", "monte-carlo", "factor-etfs", "var-backtest"]
    for e in entries:
        for key in ("question", "claim", "method", "falsifier", "result", "commit", "notebook_url", "updated"):
            assert isinstance(e[key], str) and e[key], (e["investigation"], key)
        assert isinstance(e["assumptions"], list) and e["assumptions"]
        assert isinstance(e["failures"], list) and e["failures"]
        assert re.fullmatch(r"[0-9a-f]{40}", e["commit"])
        assert ledger["source_commit"] in e["notebook_url"]
        assert ISO_DATE.match(e["updated"])
        for rev in e["revisions"]:
            assert ISO_DATE.match(rev["date"]) and rev["note"]
