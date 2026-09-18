"""Export provenance-bearing site artifacts (``site-artifact/1``).

One JSON per investigation plus ``ledger.json`` and ``manifest.json`` are
written to ``app/frontend/site-artifacts/`` in the shape defined by
``docs/site_artifact_schema.md``. The personal site renders these natively
and never restates a number, so:

* every number a published lab page states is computed here (or, where the
  page carries it only as prose, lifted verbatim from the page);
* prose is lifted verbatim from the published pages, sanitised to the
  schema's element set, with internal links rewritten to ``/lab/<slug>``;
* the run is deterministic: fixed seeds, byte-identical output except
  ``generated_at`` (override with ``QMI_GENERATED_AT`` for reproducible
  builds; set ``QMI_SKIP_TESTS=1`` to skip the pytest provenance run).

Run from the repository root::

    python scripts/export_site_artifacts.py
    python scripts/verify_site_artifacts.py   # asserts the page-stated numbers
"""

from __future__ import annotations

import html as html_mod
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_frontend_research_data as bfrd  # noqa: E402

from src.backtesting.performance_metrics import summary  # noqa: E402
from src.data.market_data_loader import DEFAULT_TICKERS, load_prices, to_returns  # noqa: E402
from src.math.monte_carlo_methods import (  # noqa: E402
    mc_price,
    simulate_cir,
    simulate_gbm,
    simulate_ou,
    simulate_regime_switching,
)
from src.math.pde_solvers import cn_bs_price  # noqa: E402
from src.models.binomial_tree import binomial_price  # noqa: E402
from src.models.black_scholes import bs_price, implied_vol  # noqa: E402
from src.models.heston_model import HestonParams, heston_price  # noqa: E402
from src.portfolio.hierarchical_risk_parity import hrp_weights  # noqa: E402
from src.portfolio.mean_variance import PortfolioResult, efficient_frontier, max_sharpe  # noqa: E402
from src.portfolio.risk_parity import risk_contributions, risk_parity_weights  # noqa: E402
from src.portfolio.robust_optimization import robust_max_sharpe  # noqa: E402
from src.risk.drawdown_analysis import drawdown_stats  # noqa: E402
from src.risk.expected_shortfall import historical_es, parametric_es  # noqa: E402
from src.risk.stress_testing import HISTORICAL_SCENARIOS, correlation_stress, run_scenarios  # noqa: E402
from src.risk.value_at_risk import (  # noqa: E402
    cornish_fisher_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)

FRONTEND = ROOT / "app" / "frontend"
OUT = FRONTEND / "site-artifacts"
DATA = ROOT / "data" / "snapshots"
SCHEMA = "site-artifact/1"
REPO = bfrd.REPO
EVIDENCE_COMMIT = bfrd.SOURCE_COMMIT  # the commit the published pages cite
SEED = 42
SIG_DIGITS = 6
MAX_POINTS = 1500  # soft cap per series; daily correlation/equity keep every day (< 3000)
HARD_CAP = 3000

ARTIFACT_SLUGS = [
    "portfolio-estimation",
    "var-backtest",
    "factor-etfs",
    "volatility",
    "monte-carlo",
    "pricing-validation",
    "stock-bond-regimes",
]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def sig(x, n: int = SIG_DIGITS):
    """Round a float to ``n`` significant digits; NaN/inf become ``None``."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.{n}g}")


def sigs(values, n: int = SIG_DIGITS) -> list:
    return [sig(v, n) for v in values]


def iso(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def thin_positions(n: int, max_points: int = MAX_POINTS, keep: tuple[int, ...] = ()) -> list[int]:
    """Evenly spaced positions (first and last kept) plus any ``keep`` indices."""
    if n <= max_points:
        return list(range(n))
    pos = set(np.round(np.linspace(0, n - 1, max_points)).astype(int).tolist())
    pos.update(int(k) for k in keep if 0 <= int(k) < n)
    return sorted(pos)


def thin_series(index, values, max_points: int = MAX_POINTS, keep_extrema: bool = True):
    """Return (x labels, y values) thinned to ``max_points``; extrema are kept."""
    values = np.asarray(values, dtype=float)
    keep: tuple[int, ...] = ()
    if keep_extrema and len(values):
        keep = (int(np.nanargmin(values)), int(np.nanargmax(values)))
    pos = thin_positions(len(values), max_points, keep)
    x = [iso(index[i]) if isinstance(index, pd.DatetimeIndex) else sig(index[i]) for i in pos]
    return x, sigs(values[pos])


def fmt(value: float, decimals: int, prefix: str = "", suffix: str = "", sign: bool = False) -> str:
    s = f"{value:+,.{decimals}f}" if sign else f"{value:,.{decimals}f}"
    s = s.replace("-", "−")
    return f"{prefix}{s}{suffix}"


def key_number(
    id_: str,
    value: str,
    label: str,
    count=None,
    decimals: int = 0,
    prefix: str = "",
    suffix: str = "",
    page: str | None = None,
    stated: str | None = None,
    unverified: bool = False,
    note: str | None = None,
) -> dict:
    entry: dict = {"id": id_, "value": value, "label": label}
    if count is not None:
        entry["count"] = sig(count, 8)
    entry.update({"decimals": decimals, "prefix": prefix, "suffix": suffix})
    if page:
        entry["page"] = page
    if stated is not None:
        entry["stated"] = stated
    if unverified:
        entry["unverified"] = True
    if note:
        entry["note"] = note
    return entry


def figure(id_: str, type_: str, title: str, sub: str, alt: str, **kwargs) -> dict:
    fig = {"id": id_, "type": type_, "title": title, "sub": sub, "alt": alt}
    for key, value in kwargs.items():
        if value is not None:
            fig[key] = value
    return fig


def series(key: str, color: str, values, band=None, dash: bool = False) -> dict:
    s = {"key": key, "color": color, "values": list(values)}
    if band is not None:
        s["band"] = {"lo": list(band[0]), "hi": list(band[1])}
    if dash:
        s["dash"] = True
    return s


def table(id_: str, title: str, columns: list[str], rows: list[list], note: str | None = None) -> dict:
    t = {"id": id_, "title": title, "columns": columns, "rows": rows}
    if note:
        t["note"] = note
    return t


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generated_at() -> str:
    forced = os.environ.get("QMI_GENERATED_AT")
    if forced:
        return forced
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot_entry(path: Path) -> dict:
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "sha256": bfrd.sha256(path),
        "rows": int(len(frame)),
        "start": iso(frame.index[0]),
        "end": iso(frame.index[-1]),
    }


def run_tests() -> dict:
    if os.environ.get("QMI_SKIP_TESTS"):
        return {"suite": "tests", "count": None, "passed": None, "skipped_by_exporter": True}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", proc.stdout)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", proc.stdout)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", proc.stdout)) else 0
    return {"suite": "tests", "count": passed + failed + errors, "passed": passed}


class Common:
    """Objects shared by every artifact (computed once)."""

    def __init__(self) -> None:
        self.source_commit = git_head()
        self.generated_at = generated_at()
        self.data = [snapshot_entry(DATA / "prices.csv"), snapshot_entry(DATA / "ff_factors.csv")]
        header = (DATA / "prices.csv").read_text(encoding="utf-8").splitlines()[0]
        self.universe = header.split(",")[1:]
        self.tests = run_tests()
        self.cards = index_cards()
        self.ledger = ledger_entries()
        self.notes = math_notes()

    def provenance(self, notebook: str, **extra) -> dict:
        prov = {
            "source_commit": self.source_commit,
            "evidence_commit": EVIDENCE_COMMIT,
            "generated_at": self.generated_at,
            "data": self.data,
            "universe": self.universe,
            "notebook": notebook,
            "notebook_url": f"{REPO}/blob/{self.source_commit}/{notebook}",
            "tests": self.tests,
        }
        prov.update(extra)
        return prov


# ---------------------------------------------------------------------------
# HTML utilities
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*?)(/?)>", re.S)
_VOID = {"br", "img", "meta", "link", "input", "hr", "source", "path", "circle", "rect", "line", "polyline"}


def read_page(name: str) -> str:
    return (FRONTEND / name).read_text(encoding="utf-8")


def parse_attrs(attrs: str) -> dict[str, str]:
    return {k.lower(): html_mod.unescape(v) for k, v in re.findall(r'([a-zA-Z-]+)="([^"]*)"', attrs)}


def find_elements(html: str, tags, pred=None) -> list[tuple[str, dict[str, str], str]]:
    """Return ``(tag, attrs, inner_html)`` for matching elements.

    Nesting-aware: once an element matches, its descendants are skipped; when
    it does not match, the search continues inside it.
    """
    if isinstance(tags, str):
        tags = {tags}
    tags = {t.lower() for t in tags}
    alternation = "|".join(sorted(tags))
    start_re = re.compile(rf"<({alternation})\b([^>]*)>", re.S | re.I)
    out = []
    i = 0
    while True:
        m = start_re.search(html, i)
        if not m:
            break
        tag = m.group(1).lower()
        attrs = parse_attrs(m.group(2))
        after = m.end()
        depth = 1
        end_pos = None
        for t in _TAG_RE.finditer(html, after):
            if t.group(2).lower() != tag:
                continue
            if t.group(1) == "/":
                depth -= 1
            elif not t.group(4) and tag not in _VOID:
                depth += 1
            if depth == 0:
                inner_end, end_pos = t.start(), t.end()
                break
        if end_pos is None:
            raise ValueError(f"unbalanced <{tag}> in page")
        if pred is None or pred(tag, attrs):
            out.append((tag, attrs, html[after:inner_end]))
            i = end_pos
        else:
            i = after
    return out


def has_class(attrs: dict[str, str], cls: str) -> bool:
    return cls in attrs.get("class", "").split()


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def first_text(fragment: str, tag: str, cls: str | None = None) -> str | None:
    for _, attrs, inner in find_elements(fragment, tag, (lambda t, a: cls is None or has_class(a, cls))):
        return strip_tags(remove_elements(inner, "a", lambda t, a: has_class(a, "heading-link")))
    return None


def remove_elements(fragment: str, tags, pred=None) -> str:
    """Delete matching elements (outer HTML) from the fragment."""
    if isinstance(tags, str):
        tags = {tags}
    alternation = "|".join(sorted(t.lower() for t in tags))
    start_re = re.compile(rf"<({alternation})\b([^>]*)>", re.S | re.I)
    out, i = [], 0
    while True:
        m = start_re.search(fragment, i)
        if not m:
            break
        tag = m.group(1).lower()
        attrs = parse_attrs(m.group(2))
        depth, end_pos = 1, None
        for t in _TAG_RE.finditer(fragment, m.end()):
            if t.group(2).lower() != tag:
                continue
            if t.group(1) == "/":
                depth -= 1
            elif not t.group(4) and tag not in _VOID:
                depth += 1
            if depth == 0:
                end_pos = t.end()
                break
        if end_pos is None:
            raise ValueError(f"unbalanced <{tag}>")
        if pred is None or pred(tag, attrs):
            out.append(fragment[i:m.start()])
            i = end_pos
        else:
            out.append(fragment[i:m.end()])
            i = m.end()
    out.append(fragment[i:])
    return "".join(out)


def rewrite_href(href: str) -> str:
    """Internal lab links become clean ``/lab/<slug>`` paths."""
    if href.startswith(("http://", "https://", "mailto:", "#")):
        return href
    m = re.match(r"^(?:\./)?intro/index\.html(#.*)?$", href)
    if m:
        return "/lab/intro" + (m.group(1) or "")
    m = re.match(r"^(?:\./)?([a-z0-9-]+)\.html(#.*)?$", href)
    if m:
        slug, frag = m.group(1), m.group(2) or ""
        return ("/lab" if slug == "index" else f"/lab/{slug}") + frag
    return href


ALLOWED_TAGS = {
    "p", "h3", "ul", "ol", "li", "em", "strong", "a", "code", "table", "thead", "tbody",
    "tr", "th", "td", "sup", "sub", "span", "br",
}
RENAME_TAGS = {"h2": "h3", "i": "em", "b": "strong", "h4": "strong", "h5": "strong", "tfoot": "tbody"}
DROP_TAGS = {
    "script", "style", "svg", "figure", "figcaption", "nav", "button", "output", "summary",
    "iframe", "noscript", "canvas", "form", "input", "select", "label", "option", "aside", "head",
}
DROP_CLASSES = {
    "heading-link", "reading-actions", "section-nav", "local-toc", "sr-only", "cite-button",
    "proof-strip", "chart-figure", "chart", "panel", "atlas-map", "atlas-overlays", "atlas-workbench__head",
}
BLOCK_TAGS = {"p", "h3", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td"}
_PRE_TAGS = {"pre"}


class Sanitizer(HTMLParser):
    """Rebuild a fragment with the schema's element set only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[tuple[str, str]] = []  # (original tag, mode)
        self.drop_depth = 0
        self.captures: list[list[str]] = []  # caption capture buffers
        self.table_buffers: list[tuple[list[str], list[str]]] = []
        self.auto_p = False
        self.block_depth = 0
        self.in_pre = 0

    # -- helpers -----------------------------------------------------------
    def emit(self, s: str) -> None:
        if self.table_buffers:
            self.table_buffers[-1][0].append(s)
        else:
            self.out.append(s)

    def open_auto_p(self) -> None:
        if self.block_depth == 0 and not self.auto_p and not self.table_buffers:
            self.emit("<p>")
            self.auto_p = True

    def close_auto_p(self) -> None:
        if self.auto_p:
            self.emit("</p>")
            self.auto_p = False

    # -- parser callbacks --------------------------------------------------
    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        if self.drop_depth:
            if tag not in _VOID:
                self.stack.append((tag, "drop"))
                self.drop_depth += 1
            return
        classes = set(attrs.get("class", "").split())
        if tag in DROP_TAGS or classes & DROP_CLASSES:
            if tag not in _VOID:
                self.stack.append((tag, "drop"))
                self.drop_depth = 1
            return
        if tag == "a" and attrs.get("aria-label", "").startswith("Return to citation"):
            self.stack.append((tag, "drop"))
            self.drop_depth = 1
            return
        if tag == "p" and classes:
            self.close_auto_p()
            self.emit(f'<p><span class="{html_mod.escape(" ".join(sorted(classes)), quote=True)}">')
            self.stack.append((tag, "kicker-p"))
            self.block_depth += 1
            return
        if tag == "code" and self.in_pre:
            self.stack.append((tag, "unwrap"))
            return
        if tag == "caption":
            self.captures.append([])
            self.stack.append((tag, "caption"))
            self.drop_depth = 1
            return
        if tag == "table":
            self.close_auto_p()
            self.table_buffers.append(([], []))
            self.stack.append((tag, "table"))
            self.block_depth += 1
            return
        if tag == "dt":
            self.close_auto_p()
            self.emit("<p><strong>")
            self.stack.append((tag, "dt"))
            self.block_depth += 1
            return
        if tag == "dd":
            self.stack.append((tag, "dd"))
            self.block_depth += 1
            return
        if tag in _PRE_TAGS:
            self.close_auto_p()
            self.emit("<p><code>")
            self.stack.append((tag, "pre"))
            self.block_depth += 1
            self.in_pre += 1
            return
        if tag == "br":
            self.open_auto_p()
            self.emit("<br>")
            return
        name = RENAME_TAGS.get(tag, tag)
        if name in ALLOWED_TAGS:
            if name in BLOCK_TAGS:
                self.close_auto_p()
                self.block_depth += 1
            else:
                self.open_auto_p()
            attr_text = ""
            if name == "a" and attrs.get("href"):
                attr_text = f' href="{html_mod.escape(rewrite_href(attrs["href"]), quote=True)}"'
            elif name == "span" and attrs.get("class"):
                attr_text = f' class="{html_mod.escape(attrs["class"], quote=True)}"'
            elif name in {"th", "td"} and attrs.get("colspan"):
                attr_text = f' colspan="{html_mod.escape(attrs["colspan"], quote=True)}"'
            self.emit(f"<{name}{attr_text}>")
            self.stack.append((tag, name))
            return
        # anything else is unwrapped (children kept)
        if tag not in _VOID:
            self.stack.append((tag, "unwrap"))

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _VOID:
            return
        # pop to the matching open tag (tolerates stray end tags)
        idx = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                idx = i
                break
        if idx is None:
            return
        while len(self.stack) > idx:
            _, mode = self.stack.pop()
            if mode == "drop":
                self.drop_depth -= 1
            elif mode == "caption":
                self.drop_depth -= 1
                text = re.sub(r"\s+", " ", "".join(self.captures.pop())).strip()
                if text and self.table_buffers:
                    self.table_buffers[-1][1].append(text)
            elif mode == "table":
                body, captions = self.table_buffers.pop()
                self.block_depth -= 1
                for cap in captions:
                    self.emit(f"<p><em>{html_mod.escape(cap, quote=False)}</em></p>")
                self.emit("<table>" + "".join(body) + "</table>")
            elif mode == "dt":
                self.emit("</strong> ")
                self.block_depth -= 1
            elif mode == "kicker-p":
                self.emit("</span></p>")
                self.block_depth -= 1
            elif mode == "dd":
                self.emit("</p>")
                self.block_depth -= 1
            elif mode == "pre":
                self.emit("</code></p>")
                self.block_depth -= 1
                self.in_pre -= 1
            elif mode == "unwrap":
                pass
            else:
                self.emit(f"</{mode}>")
                if mode in BLOCK_TAGS:
                    self.block_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.drop_depth:
            if self.captures and self.stack and self.stack[-1][1] == "caption":
                self.captures[-1].append(data)
            return
        text = data if self.in_pre else re.sub(r"\s+", " ", data)
        if not text.strip():
            if self.block_depth or self.auto_p:
                self.emit(html_mod.escape(text, quote=False) if self.in_pre else " ")
            return
        self.open_auto_p()
        self.emit(html_mod.escape(text, quote=False))

    def result(self) -> str:
        self.close()
        self.close_auto_p()
        out = "".join(self.out)
        out = re.sub(r"\s*(</?(?:p|h3|ul|ol|li|table|thead|tbody|tr|th|td)\b[^>]*>)\s*", r"\1", out)
        out = re.sub(r"<p>\s*</p>", "", out)
        out = re.sub(r" {2,}", " ", out)
        return out.strip()


def sanitize(fragment: str) -> str:
    parser = Sanitizer()
    parser.feed(fragment)
    return parser.result()


def remove_first(fragment: str, tag: str, pred=None) -> str:
    seen = [False]

    def once(t, a):
        if seen[0] or (pred is not None and not pred(t, a)):
            return False
        seen[0] = True
        return True

    return remove_elements(fragment, tag, once)


def section_body(inner: str, keep_heading: bool = False, strip_kicker: bool = True) -> str:
    """Drop the leading kicker paragraph, the revision line and (unless kept) the heading."""
    body = inner
    if strip_kicker:
        body = remove_first(body, "p", lambda t, a: has_class(a, "kicker"))
    if not keep_heading:
        body = remove_first(body, "h2")
    body = re.sub(r"<p>\s*<small>.*?</small>\s*</p>", "", body, flags=re.S)
    return body


# ---------------------------------------------------------------------------
# Page extraction (verbatim lifts)
# ---------------------------------------------------------------------------
def page_dates(html: str) -> dict:
    out = {}
    for m in re.finditer(r'"(datePublished|dateModified)":\s*"(\d{4}-\d{2}-\d{2})"', html):
        out[{"datePublished": "published", "dateModified": "modified"}[m.group(1)]] = m.group(2)
    return out


def index_cards() -> dict[str, dict]:
    """Card metadata (kicker, result, evidence class, metric) keyed by href."""
    html = read_page("index.html")
    cards: dict[str, dict] = {}
    for _, attrs, inner in find_elements(html, "a", lambda t, a: has_class(a, "finding-card")):
        cards[attrs["href"]] = {
            "kicker": first_text(inner, "p", "finding-question"),
            "title": first_text(inner, "h3"),
            "evidence_class": first_text(inner, "p", "finding-answer-label"),
            "result": first_text(inner, "p", "finding-answer"),
            "metric": first_text(inner, "p", "finding-metric"),
        }
    return cards


def bibliography(html: str) -> list[dict]:
    """``ol.bibliography`` entries as ``{n, title, url, publisher, text}``."""
    entries = []
    for _, _, ol_inner in find_elements(html, "ol", lambda t, a: has_class(a, "bibliography")):
        for n, (_, _, li) in enumerate(find_elements(ol_inner, "li"), start=1):
            li = remove_elements(li, "a", lambda t, a: a.get("href", "").startswith("#"))
            text = strip_tags(li)
            quoted = re.search(r"[“\"](.+?)[”\"]", li, re.S)
            italic = re.search(r"<i>(.+?)</i>", li, re.S)
            urls = re.findall(r'href="(https?://[^"]+)"', li)
            if quoted:
                title = strip_tags(quoted.group(1)).rstrip(",")
                publisher = strip_tags(italic.group(1)) if italic else None
            elif italic:
                title = strip_tags(italic.group(1))
                tail = li[italic.end():]
                tail = re.split(r"<a\b", tail, maxsplit=1)[0]
                publisher = strip_tags(tail).strip(" ,.") or None
            else:
                title, publisher = text, None
            entries.append({"n": n, "title": title, "url": urls[0] if urls else None,
                            "publisher": publisher, "text": text})
    return entries


def _sections_of(fragment: str, tags=("section", "aside")) -> list[dict]:
    out = []
    for tag, attrs, inner in find_elements(fragment, set(tags), lambda t, a: True):
        classes = attrs.get("class", "")
        if tag == "section" and "section" not in classes.split() and "decision-summary" not in classes:
            continue
        has_h2 = first_text(inner, "h2") is not None
        heading = first_text(inner, "h2") or attrs.get("aria-label")
        fallback = re.sub(r"[^a-z0-9]+", "-", (attrs.get("aria-label") or heading or "section").lower()).strip("-")
        sid = attrs.get("aria-labelledby") or attrs.get("id") or fallback
        kicker = first_text(inner, "p", "kicker") if has_h2 else None
        body = sanitize(section_body(inner, keep_heading="decision-summary" in classes, strip_kicker=has_h2))
        if not body:
            continue
        entry = {"id": sid, "heading": heading, "html": body}
        if kicker:
            entry["kicker"] = kicker
        if "decision-summary" in classes:
            entry["role"] = "decision-summary"
        if tag == "aside":
            entry["role"] = "regime-lens"
        out.append(entry)
    return out


def finding_prose(page: str, extra_source_pages: tuple[str, ...] = (), extra_sections: list[dict] | None = None) -> tuple[dict, dict]:
    """Prose block for a ``finding-*.html`` page plus its title/dates."""
    html = read_page(page)
    article = find_elements(html, "article")[0][2]
    title = first_text(article, "h1")
    lede = first_text(article, "p", "lede")
    blocks = _sections_of(article)
    decision = next(b for b in blocks if b.get("role") == "decision-summary")
    sections, limitations = [], None
    for block in blocks:
        if block is decision:
            continue
        if block["id"] == "limitations":
            limitations = block["html"]
            continue
        sections.append({k: v for k, v in block.items() if k != "role"} | ({"role": block["role"]} if block.get("role") else {}))
    if extra_sections:
        sections.extend(extra_sections)
    sources = bibliography(html)
    for extra in extra_source_pages:
        offset = len(sources)
        for entry in bibliography(read_page(extra)):
            entry["n"] += offset
            entry["page"] = f"/lab/{extra[:-5]}"
            sources.append(entry)
    small = re.search(r"<p><small>(.*?)</small></p>", article, re.S)
    revisions = []
    if small:
        for part in strip_tags(small.group(1)).split(" · "):
            m = re.match(r"(.*?)\s*(\d{1,2} \w+ \d{4})$", part.strip())
            if m:
                revisions.append({"date": datetime.strptime(m.group(2), "%d %B %Y").date().isoformat(),
                                  "note": m.group(1).strip()})
    prose = {
        "decision_summary_html": decision["html"],
        "sections": sections,
        "limitations_html": limitations or "",
        "sources": sources,
    }
    meta = {"title": title, "lede": lede, "dates": page_dates(html), "revisions": revisions,
            "decision_heading": decision.get("heading")}
    return prose, meta


def page_sections(page: str, id_prefix: str = "", only_ids: set[str] | None = None) -> list[dict]:
    """Prose sections of a non-finding page (dossier, case study, model pages)."""
    html = read_page(page)
    main = find_elements(html, "main")[0][2]
    out = []
    for block in _sections_of(main, tags=("section",)):
        if only_ids is not None and block["id"] not in only_ids:
            continue
        block = {k: v for k, v in block.items() if k != "role"} | ({"role": block["role"]} if block.get("role") else {})
        block["page"] = f"/lab/{page[:-5]}"
        if id_prefix and not block["id"].startswith(id_prefix):
            block["id"] = id_prefix + block["id"]
        out.append(block)
    return out


def ledger_entries() -> dict[str, dict]:
    """Research-log entries keyed by their page id, fields lifted verbatim."""
    html = read_page("research-log.html")
    entries: dict[str, dict] = {}
    for _, attrs, inner in find_elements(html, "article", lambda t, a: has_class(a, "ledger-entry")):
        fields: dict[str, str] = {}
        for _, _, div in find_elements(inner, "div"):
            dt = first_text(div, "dt")
            dd = find_elements(div, "dd")
            if dt and dd:
                fields[dt] = dd[0][2]
        repro = fields.get("Reproduction", "")
        commit = re.search(r"/commit/([0-9a-f]{40})", repro)
        updated = re.search(r'datetime="(\d{4}-\d{2}-\d{2})"', fields.get("Last updated", ""))
        entries[attrs["id"]] = {
            "id": attrs["id"],
            "number": first_text(inner, "span", "ledger-number"),
            "title": first_text(inner, "h2"),
            "status": first_text(inner, "span", "status"),
            "fields": {k: strip_tags(v) for k, v in fields.items()},
            "fields_html": {k: sanitize(v) for k, v in fields.items()},
            "commit": commit.group(1) if commit else None,
            "updated": updated.group(1) if updated else None,
            "links": re.findall(r'href="([^"]+)"', inner),
        }
    return entries


def split_list(text: str) -> list[str]:
    if "; " in text:
        return [p.strip() for p in text.split("; ") if p.strip()]
    parts = re.split(r"(?<=\.)\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


class MarkdownConverter(HTMLParser):
    """HTML → Markdown with the KaTeX source preserved (``$…$`` / ``$$…$$``)."""

    DROP = {"button", "nav", "script", "style", "svg"}
    DROP_CLASSES = {"reading-actions", "section-nav", "heading-link"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[tuple[str, str, str]] = []  # (tag, mode, payload)
        self.drop = 0
        self.in_math = 0
        self.cited: list[int] = []

    def handle_starttag(self, tag, attrs) -> None:
        attrs = {k: (v or "") for k, v in attrs}
        classes = set(attrs.get("class", "").split())
        if tag == "br":
            if not self.drop:
                self.out.append("  \n")
            return
        if self.drop or tag in self.DROP or classes & self.DROP_CLASSES:
            self.drop += 1
            self.stack.append((tag, "drop", ""))
            return
        if tag == "a":
            m = re.match(r"#notes-ref-(\d+)$", attrs.get("href", ""))
            if m:
                self.cited.append(int(m.group(1)))
                self.out.append(f"[^{m.group(1)}]")
                self.drop += 1
                self.stack.append((tag, "drop", ""))
                return
            self.out.append("[")
            self.stack.append((tag, "a", rewrite_href(attrs.get("href", ""))))
            return
        mode = "other"
        if tag == "div" and "math-block" in classes:
            self.in_math += 1
            self.out.append("\n\n")
            mode = "math"
        elif tag == "p":
            self.out.append("\n\n")
            if "note-meta" in classes:
                self.out.append("*")
                mode = "p-meta"
            else:
                mode = "p"
        elif tag in {"em", "i"}:
            self.out.append("*")
            mode = "em"
        elif tag in {"strong", "b"}:
            self.out.append("**")
            mode = "strong"
        elif tag == "code":
            self.out.append("`")
            mode = "code"
        self.stack.append((tag, mode, ""))

    def handle_endtag(self, tag) -> None:
        idx = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if idx is None:
            return
        while len(self.stack) > idx:
            _, mode, payload = self.stack.pop()
            if mode == "drop":
                self.drop -= 1
            elif mode == "math":
                self.in_math -= 1
                self.out.append("\n\n")
            elif mode == "p-meta":
                self.out.append("*\n\n")
            elif mode == "p":
                self.out.append("\n\n")
            elif mode == "em":
                self.out.append("*")
            elif mode == "strong":
                self.out.append("**")
            elif mode == "code":
                self.out.append("`")
            elif mode == "a":
                self.out.append(f"]({payload})")

    def handle_data(self, data) -> None:
        if self.drop:
            return
        self.out.append(data.strip() if self.in_math else re.sub(r"\s+", " ", data))

    def result(self) -> str:
        self.close()
        md = "".join(self.out)
        md = re.sub(r"\\\((.+?)\\\)", r"$\1$", md, flags=re.S)
        md = re.sub(r"[ \t]+\n", "\n", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = re.sub(r"(?m)^ +", "", md)
        md = re.sub(r"\*\s+\*", "", md)
        return md.strip()


def math_notes() -> list[dict]:
    """The six notes of ``math-notes.html`` as Markdown with KaTeX source."""
    html = read_page("math-notes.html")
    refs = {e["n"]: e["text"] for e in bibliography(html)}
    notes = []
    for n, (_, attrs, inner) in enumerate(find_elements(html, "section", lambda t, a: has_class(a, "note-section")), start=1):
        heading = first_text(inner, "h2")
        body = section_body(inner)
        conv = MarkdownConverter()
        conv.feed(body)
        md = conv.result()
        cited = sorted(set(conv.cited))
        if cited:
            md += "\n\n" + "\n".join(f"[^{k}]: {refs[k]}" for k in cited if k in refs)
        note_id = attrs.get("aria-labelledby", f"note-{n}").replace("note-", "", 1)
        has_tex = ("$" in md) or ("\\(" in body)
        notes.append({
            "id": note_id,
            "n": f"§{n:02d}",
            "heading": heading,
            "markdown": md,
            "source": f"/lab/math-notes#{attrs.get('aria-labelledby', '')}",
            "tex_source": "katex" if has_tex else "rendered-text",
        })
    return notes


def atlas_episodes() -> tuple[dict, dict]:
    """Six historical lenses (table) and the activity × inflation plane (heatmap)."""
    html = read_page("regime-atlas.html")
    rows = []
    for _, _, card in find_elements(html, "details", lambda t, a: has_class(a, "episode-card")):
        summary_html = find_elements(card, "summary")[0][2]
        period = first_text(summary_html, "small") or ""
        question = first_text(summary_html, "strong") or ""
        record_p = find_elements(card, "p", lambda t, a: not a.get("class"))
        record = strip_tags(remove_elements(record_p[0][2], "span")) if record_p else ""
        grid = {}
        for _, _, div in find_elements(card, "div", lambda t, a: not a.get("class")):
            dt, dd = first_text(div, "dt"), first_text(div, "dd")
            if dt and dd:
                grid[dt] = dd
        boundary = first_text(card, "p", "episode-boundary") or ""
        boundary = re.sub(r"^What this does not establish:\s*", "", boundary)
        sources = re.findall(r'href="(https?://[^"]+)"', find_elements(card, "p", lambda t, a: has_class(a, "episode-sources"))[0][2])
        region, _, dates = period.partition(" · ")
        rows.append([region, dates, question, grid.get("Axis path", ""), grid.get("Institutional record", ""),
                     grid.get("Possible transmission", ""), grid.get("Competing explanations", ""), record, boundary, "; ".join(sources)])
    episodes = table(
        "regime-episodes",
        "Historical lenses of the Market Regime Atlas",
        ["Episode", "Period", "Question", "Axis path", "Institutional record", "Possible transmission",
         "Competing explanations", "Documented record", "What this does not establish", "Sources"],
        rows,
        note="Each lens is dated and ex post; lifted verbatim from regime-atlas.html.",
    )
    # transmission plane: count schematic episode marks per quadrant
    quads = {}
    for m in re.finditer(r'<rect class="plane-quadrant" data-plane-state="([a-z-]+)" x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', html):
        quads[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    marks = []
    for m in re.finditer(r'<g class="plane-episode[^"]*"><circle cx="([\d.]+)" cy="([\d.]+)" r="[\d.]+"/><text[^>]*>([^<]+)</text></g>', html):
        cx, cy, label = float(m.group(1)), float(m.group(2)), html_mod.unescape(m.group(3))
        state = next((s for s, (x, y, w, h) in quads.items() if x <= cx < x + w and y <= cy < y + h), None)
        marks.append((label, state))
    counts = {s: sum(1 for _, st in marks if st == s) for s in quads}
    state_titles = {}
    for m in re.finditer(r'data-atlas-state="([a-z-]+)"[^>]*>\s*<span>([^<]+)</span>\s*<strong>([^<]+)</strong>', html):
        state_titles[m.group(1)] = (html_mod.unescape(m.group(2)), html_mod.unescape(m.group(3)))
    heatmap = figure(
        "regime-plane",
        "heatmap",
        "Transmission plane · episode marks per activity × inflation quadrant",
        "count of marks · six dated lenses and the 2020–23 trajectory · regime-atlas.html",
        "A two-by-two grid of the atlas transmission plane counting how many documented episode marks the atlas places in each activity and inflation quadrant; placements are schematic readings of each episode's documented axis path, not measured coordinates.",
        grid={
            "rows": ["Inflation pressure building", "Inflation pressure easing"],
            "cols": ["Activity weakening", "Activity strengthening"],
            "values": [[counts.get("weak-build", 0), counts.get("strong-build", 0)],
                       [counts.get("weak-ease", 0), counts.get("strong-ease", 0)]],
        },
        scale={"min": 0, "max": max(counts.values()) if counts else 1},
        table={
            "columns": ["Quadrant", "Atlas scenario heading", "Marks", "Episode marks"],
            "rows": [[state_titles.get(s, (s, ""))[0], state_titles.get(s, (s, ""))[1], counts.get(s, 0),
                      "; ".join(l for l, st in marks if st == s)] for s in ("weak-build", "strong-build", "weak-ease", "strong-ease")],
        },
    )
    return episodes, heatmap


# ---------------------------------------------------------------------------
# Shared artifact assembly
# ---------------------------------------------------------------------------
def card_meta(common: Common, page: str) -> dict:
    card = common.cards.get(page) or {}
    return {"kicker": card.get("kicker"), "result": card.get("result"),
            "evidence_class": card.get("evidence_class"), "metric": card.get("metric")}


def assemble(common: Common, slug: str, page: str, prose: dict, meta: dict, notebook: str,
             key_numbers: list, figures: list, tables: list, math_note_ids: tuple[str, ...] = (),
             ledger_id: str | None = None, **extra) -> dict:
    card = card_meta(common, page)
    payload = {
        "schema_version": SCHEMA,
        "investigation": slug,
        "title": meta["title"],
        "kicker": card["kicker"],
        "result": card["result"],
        "evidence_class": card["evidence_class"],
        "lede": meta.get("lede"),
        "metric": card["metric"],
        "page": f"/lab/{page[:-5]}",
        "record": f"/lab/research-log#{ledger_id}" if ledger_id else None,
        "dates": meta.get("dates", {}),
        "provenance": common.provenance(notebook, **extra),
        "key_numbers": key_numbers,
        "figures": figures,
        "tables": tables,
        "prose": prose,
        "math_notes": [n for n in common.notes if n["id"] in math_note_ids],
        "revisions": meta.get("revisions", []),
    }
    return payload


def check_series_lengths(payload: dict) -> None:
    for fig in payload["figures"]:
        for s in fig.get("series", []):
            assert len(s["values"]) <= HARD_CAP, (payload["investigation"], fig["id"], len(s["values"]))
        if "points" in fig:
            assert len(fig["points"]) <= HARD_CAP


# ---------------------------------------------------------------------------
# 1. portfolio-estimation  (notebook 02)
# ---------------------------------------------------------------------------
def build_portfolio_estimation(common: Common) -> dict:
    page = "finding-portfolio-estimation.html"
    prose, meta = finding_prose(page)
    tickers = list(DEFAULT_TICKERS)
    prices = load_prices(tickers)
    rets = to_returns(prices)
    ins, oos = rets.loc[:"2021-12-31"], rets.loc["2022-01-01":]
    n_years = len(ins) / 252
    mu, cov, vol = ins.mean() * 252, ins.cov() * 252, ins.std() * np.sqrt(252)
    mu_se = vol / np.sqrt(n_years)
    rf = 0.02

    w_mvo = max_sharpe(mu, cov, rf=rf)
    w_rob = robust_max_sharpe(mu, cov, mu_uncertainty=mu_se, rf=rf)
    w_rp, w_hrp = risk_parity_weights(cov), hrp_weights(ins)

    def as_result(w: pd.Series) -> PortfolioResult:
        er, v = float(w @ mu), float(np.sqrt(w @ cov @ w))
        return PortfolioResult(weights=w, expected_return=er, volatility=v, sharpe=(er - rf) / v)

    results = {"MVO max Sharpe": w_mvo, "Robust MVO": w_rob, "Risk parity": as_result(w_rp), "HRP": as_result(w_hrp)}
    methods = list(results)
    W = pd.DataFrame({k: v.weights for k, v in results.items()}).reindex(tickers).fillna(0.0)
    eff_n = 1.0 / (W**2).sum()
    rc = pd.DataFrame({m: risk_contributions(W[m], cov) for m in methods})
    frontier = efficient_frontier(mu, cov, n_points=30, rf=rf)
    perf_in = {m: summary(ins @ W[m], rf=rf) for m in methods}
    perf_out = {m: summary(oos @ W[m], rf=rf) for m in methods}
    oos_curves = (1 + pd.DataFrame({m: oos @ W[m] for m in methods})).cumprod()
    y_oos = len(oos) / 252
    sr_mvo_out = float(perf_out["MVO max Sharpe"]["sharpe_ratio"])
    sharpe_se = math.sqrt((1 + sr_mvo_out**2 / 2) / y_oos)
    sr_in, sr_out = float(perf_in["MVO max Sharpe"]["sharpe_ratio"]), sr_mvo_out
    rob_out = float(perf_out["Robust MVO"]["sharpe_ratio"])

    key_numbers = [
        key_number("mvo-sharpe-fit", fmt(sr_in, 2), "MVO Sharpe · 2015–2021 fit", sr_in, 2, page=page, stated="1.12"),
        key_number("mvo-sharpe-eval", fmt(sr_out, 2), "MVO Sharpe · sealed 2022–2026 evaluation", sr_out, 2, page=page, stated="0.44"),
        key_number("sharpe-gap", fmt(round(sr_in, 2) - round(sr_out, 2), 2), "Gap between the two published Sharpe ratios",
                   round(sr_in, 2) - round(sr_out, 2), 2, page=page, stated="0.68",
                   note=f"Difference of the two published (rounded) Sharpe ratios; unrounded gap {sr_in - sr_out:.3f}."),
        key_number("sharpe-sampling-error", "≈" + fmt(sharpe_se, 1), "Estimated sampling error of a Sharpe ratio over this window",
                   sharpe_se, 1, page=page, stated="near 0.5", note="sqrt((1 + SR²/2) / Y) with Y the evaluation window in years."),
        key_number("robust-sharpe-eval", "≈" + fmt(rob_out, 1), "Robust MVO Sharpe · evaluation (best of the four)", rob_out, 1, page=page, stated="about 0.6"),
        key_number("robust-drawdown-eval", fmt(100 * float(perf_out["Robust MVO"]["max_drawdown"]), 1, suffix="%"),
                   "Robust MVO maximum drawdown · evaluation window", 100 * float(perf_out["Robust MVO"]["max_drawdown"]), 1, suffix="%", page=page, stated="more than 30% in 2022"),
        key_number("observations", f"{len(rets):,}", "Daily observations · 15 ETFs", len(rets), 0, page=page, stated="2,891"),
        key_number("assets", str(len(tickers)), "Liquid ETFs in the universe", len(tickers), 0, page=page, stated="15"),
        key_number("fit-window", "2015–2021", "Fit window", page=page),
        key_number("eval-window", "2022–2026", "Sealed evaluation window", page=page),
        key_number("sample-start", iso(rets.index[0]), "First daily observation", page="research-log.html", stated="2015-01-05"),
        key_number("sample-end", iso(rets.index[-1]), "Last daily observation", page="research-log.html", stated="2026-07-06"),
    ]

    colors = {"MVO max Sharpe": "c1", "Robust MVO": "c2", "Risk parity": "c3", "HRP": "c4"}
    points = [{"x": sig(100 * r.volatility), "y": sig(100 * r.expected_return), "label": "frontier", "color": "c5"} for r in frontier.itertuples()]
    points += [{"x": sig(100 * vol[t]), "y": sig(100 * mu[t]), "label": t, "color": "c6"} for t in tickers]
    points += [{"x": sig(100 * results[m].volatility), "y": sig(100 * results[m].expected_return), "label": m, "color": colors[m], "strong": True} for m in methods]
    figures = [
        figure("frontier", "scatter", "In-sample efficient frontier, 2015–2021 estimates",
               "% annualised · in-sample moments · notebook 02",
               "Scatter of annualised volatility against expected return: the constrained efficient frontier, the fifteen ETFs, and the four construction methods placed by their in-sample moments.",
               points=points, x={"label": "annualised volatility (%)"}, y={"label": "annualised expected return (%)"}),
        figure("sharpe-in-out", "dots", "Sharpe ratio by construction method · fit window and sealed evaluation",
               "Sharpe (rf 2%) · 2015–2021 vs 2022–2026 · notebook 02",
               "Dots comparing each method's in-sample Sharpe ratio with its out-of-sample Sharpe ratio; every method degrades and the ranking reshuffles.",
               x={"label": "construction method", "values": methods},
               series=[series("in-sample 2015–2021", "c1", [sig(float(perf_in[m]["sharpe_ratio"])) for m in methods]),
                       series("out-of-sample 2022–2026", "c2", [sig(float(perf_out[m]["sharpe_ratio"])) for m in methods])],
               yMin=0),
        figure("oos-equity", "line", "Out-of-sample equity curves, frozen 2021 weights",
               "growth of $1 · 2022-01 = 1 · daily · notebook 02",
               "Line chart of the growth of one dollar for the four frozen weight vectors across the sealed 2022–2026 window.",
               x={"label": "date", "values": [iso(d) for d in oos_curves.index]},
               series=[series(m, colors[m], sigs(oos_curves[m].to_numpy())) for m in methods]),
        figure("weights", "columns", "Weights fitted on 2015–2021",
               "weight (%) · long-only, fully invested · notebook 02",
               "Grouped columns of the portfolio weight each method assigns to each of the fifteen ETFs.",
               x={"label": "ETF", "values": tickers},
               series=[series(m, colors[m], sigs(100 * W[m].to_numpy())) for m in methods], yMin=0),
        figure("risk-contributions", "columns", "Fractional risk contributions by construction method",
               "fraction of portfolio variance (%) · in-sample covariance · notebook 02",
               "Grouped columns of each asset's fractional contribution to portfolio variance under the four methods, against the one-over-N line.",
               x={"label": "ETF", "values": tickers},
               series=[series(m, colors[m], sigs(100 * rc[m].to_numpy())) for m in methods],
               rules=[{"axis": "y", "at": sig(100 / len(tickers)), "label": "1/N", "color": "c7"}], yMin=0),
    ]
    assert len(figures[0]["points"]) <= 400
    perf_keys = [("annualized_return", "Annualised return"), ("annualized_vol", "Annualised volatility"),
                 ("sharpe_ratio", "Sharpe (rf 2%)"), ("sortino_ratio", "Sortino"), ("max_drawdown", "Max drawdown"),
                 ("hit_rate", "Hit rate"), ("skew", "Skew"), ("kurtosis", "Excess kurtosis"), ("VaR95", "VaR 95 (1d)"), ("ES95", "ES 95 (1d)")]
    tables = [
        table("performance", "Performance by method · in-sample 2015–2021 and out-of-sample 2022–2026",
              ["Window", "Method"] + [label for _, label in perf_keys],
              [[w, m] + [sig(float(p[m][k])) for k, _ in perf_keys] for w, p in (("in-sample 2015–2021", perf_in), ("out-of-sample 2022–2026", perf_out)) for m in methods],
              note="Returns, volatility, drawdown and VaR/ES are decimals (0.146 = 14.6%)."),
        table("weights", "Portfolio weights fitted on 2015–2021", ["ETF"] + methods,
              [[t] + [sig(float(W.loc[t, m])) for m in methods] for t in tickers]),
        table("effective-n", "Effective number of holdings (1 / Σ w²)", ["Method", "Effective N"],
              [[m, sig(float(eff_n[m]))] for m in methods]),
        table("risk-contributions", "Fractional risk contributions", ["ETF"] + methods,
              [[t] + [sig(float(rc.loc[t, m])) for m in methods] for t in tickers]),
        table("estimation", "In-sample moments and the standard error of the annualised mean",
              ["ETF", "Annualised mean", "Annualised vol", "SE(mean)", "mean / SE"],
              [[t, sig(float(mu[t])), sig(float(vol[t])), sig(float(mu_se[t])), sig(float(mu[t] / mu_se[t]))] for t in mu.sort_values(ascending=False).index],
              note=f"SE(mean) = σ / √Y with Y = {n_years:.2f} years of in-sample data."),
        table("frontier", "Efficient frontier points (in-sample)", ["Expected return", "Volatility", "Sharpe"] + tickers,
              [[sig(r.expected_return), sig(r.volatility), sig(r.sharpe)] + [sig(float(getattr(r, t))) for t in tickers] for r in frontier.itertuples()]),
    ]
    payload = assemble(common, "portfolio-estimation", page, prose, meta, "notebooks/02_portfolio_optimization.ipynb",
                       key_numbers, figures, tables, ledger_id="portfolio-construction",
                       parameters={"risk_free_rate": rf, "fit_window": [iso(ins.index[0]), iso(ins.index[-1])],
                                   "evaluation_window": [iso(oos.index[0]), iso(oos.index[-1])], "frontier_points": 30,
                                   "robust_uncertainty": "one standard error of the annualised mean per asset", "seed": SEED})
    return payload


# ---------------------------------------------------------------------------
# 2. var-backtest  (risk_study + notebook 06)
# ---------------------------------------------------------------------------
def build_var_backtest(common: Common, risk: dict) -> dict:
    page = "finding-var-backtest.html"
    prose, meta = finding_prose(page, extra_source_pages=("risk.html",))
    r, equity, dd, roll_vol = risk["returns"], risk["equity"], risk["drawdown"], risk["roll_vol"]
    tests, stresses, forecasts = risk["tests"], risk["stresses"], risk["forecasts"]
    target, prices = risk["target"], risk["prices"]
    var95, es95 = risk["var95"], risk["es95"]
    var_ci = np.quantile(risk["boot_var"], [0.025, 0.975])
    es_ci = np.quantile(risk["boot_es"], [0.025, 0.975])
    n_days, n_fc = len(r), tests["forecast_days"]
    breaches = risk["breach_values"].astype(int)
    # run lengths of consecutive breaches
    runs: dict[int, int] = {}
    k = 0
    for b in np.append(breaches, 0):
        if b:
            k += 1
        elif k:
            runs[k] = runs.get(k, 0) + 1
            k = 0
    tc = tests["transition_counts"]
    p_breach = tests["breaches"] / n_fc
    expected_n11 = (tc["n10"] + tc["n11"]) * p_breach
    perf2 = summary(r, rf=0.02)
    sharpe3 = (risk["mean_d"] * 252 - 0.03) / risk["annual_vol"]
    # VaR four ways (notebook 06, section 2) — rng consumed in the notebook's order
    asset_rets = to_returns(prices).loc[r.index]
    mu_d, cov_d = asset_rets.mean(), asset_rets.cov()
    rng = np.random.default_rng(SEED)
    var_tab = {}
    for alpha in (0.95, 0.99):
        var_tab[alpha] = {
            "historical VaR": historical_var(r, alpha=alpha),
            "parametric (normal) VaR": parametric_var(r, alpha=alpha),
            "Cornish-Fisher VaR": cornish_fisher_var(r, alpha=alpha),
            "Monte Carlo VaR": monte_carlo_var(mu_d, cov_d, target, alpha=alpha, n_sims=100_000, rng=rng),
            "historical ES": historical_es(r, alpha=alpha),
            "parametric ES": parametric_es(r, alpha=alpha),
        }
    top5 = drawdown_stats(r)
    scen = run_scenarios(target, HISTORICAL_SCENARIOS)
    stress_rows = risk["stresses"]

    key_numbers = [
        key_number("breaches", str(tests["breaches"]), "breaches at 95% VaR", tests["breaches"], 0, page=page, stated="135"),
        key_number("expected-breaches", f"{tests['expected_breaches']:.0f}", "expected breaches at 5%", tests["expected_breaches"], 0, page=page, stated="107"),
        key_number("breach-rate", fmt(100 * tests["breach_rate"], 2, suffix="%"), "observed breach rate", 100 * tests["breach_rate"], 2, suffix="%", page=page, stated="6.31%"),
        key_number("forecast-days", f"{n_fc:,}", "one-step forecasts", n_fc, 0, page=page, stated="2,140"),
        key_number("net-returns", f"{n_days:,}", "net daily returns after warm-up", n_days, 0, page=page, stated="2,640"),
        key_number("kupiec-p", f"{tests['kupiec_p']:.4f}", "Kupiec unconditional-coverage p-value", tests["kupiec_p"], 4, page=page, stated="0.0075"),
        key_number("independence-p", f"{tests['christoffersen_independence_p']:.5f}", "Christoffersen independence p-value", tests["christoffersen_independence_p"], 5, page=page, stated="0.00025"),
        key_number("conditional-coverage-p", f"{tests['christoffersen_conditional_coverage_p']:.6f}", "Conditional-coverage p-value", tests["christoffersen_conditional_coverage_p"], 6, page=page, stated="0.000035"),
        key_number("var95", fmt(100 * var95, 3, suffix="%"), "Historical VaR 95 (one day)", 100 * var95, 3, suffix="%", page=page, stated="1.079%"),
        key_number("var95-ci-low", fmt(100 * var_ci[0], 3, suffix="%"), "VaR 95 bootstrap interval · lower", 100 * var_ci[0], 3, suffix="%", page=page, stated="0.930%"),
        key_number("var95-ci-high", fmt(100 * var_ci[1], 3, suffix="%"), "VaR 95 bootstrap interval · upper", 100 * var_ci[1], 3, suffix="%", page=page, stated="1.219%"),
        key_number("es95", fmt(100 * es95, 3, suffix="%"), "Historical ES 95 (one day)", 100 * es95, 3, suffix="%", page=page, stated="1.718%"),
        key_number("es95-ci-low", fmt(100 * es_ci[0], 3, suffix="%"), "ES 95 bootstrap interval · lower", 100 * es_ci[0], 3, suffix="%", page=page, stated="1.421%"),
        key_number("es95-ci-high", fmt(100 * es_ci[1], 3, suffix="%"), "ES 95 bootstrap interval · upper", 100 * es_ci[1], 3, suffix="%", page=page, stated="2.126%"),
        key_number("n11", str(tc["n11"]), "breach-follows-breach days observed", tc["n11"], 0, page=page, stated="20"),
        key_number("n11-expected", "≈" + fmt(expected_n11, 1), "breach-follows-breach days expected under independence", expected_n11, 1, page=page, stated="≈8.5"),
        *[key_number(f"runs-{name}", str(runs.get(k, 0)), f"{label} exception runs (from the breach dates)", runs.get(k, 0), 0, page=page, stated=stated,
                     unverified=runs.get(k, 0) != int(stated),
                     note="The page's run structure was derived from the transition counts alone; the recorded breach dates give "
                          + ", ".join(f"{v} run(s) of {k} day(s)" for k, v in sorted(runs.items())) + ".")
          for k, name, label, stated in ((1, "single", "isolated single-day", "99"), (2, "two", "two-day", "12"), (3, "three", "three-day", "4"))],
        key_number("runs-four", str(runs.get(4, 0)), "four-day exception runs (from the breach dates)", runs.get(4, 0), 0, page="computed"),
        key_number("annual-return", fmt(100 * risk["annual_return"], 2, suffix="%"), "Annualised return", 100 * risk["annual_return"], 2, suffix="%", page="risk.html", stated="10.14%"),
        key_number("annual-vol", fmt(100 * risk["annual_vol"], 2, suffix="%"), "Annualised volatility", 100 * risk["annual_vol"], 2, suffix="%", page="risk.html", stated="11.50%"),
        key_number("sharpe-rf3", fmt(sharpe3, 2), "Sharpe (rf 3%)", sharpe3, 2, page="risk.html", stated="0.64"),
        key_number("sharpe-rf2", fmt(float(perf2["sharpe_ratio"]), 2), "Sharpe (rf 2%)", float(perf2["sharpe_ratio"]), 2, page="notebooks/06", stated="≈0.72"),
        key_number("max-drawdown", fmt(100 * float(dd.min()), 1, suffix="%"), "Maximum drawdown", 100 * float(dd.min()), 1, suffix="%", page="risk.html", stated="-24.3%"),
        key_number("skewness", fmt(float(r.skew()), 2), "Skewness of daily returns", float(r.skew()), 2, page="risk.html", stated="-0.54"),
        key_number("excess-kurtosis", fmt(float(r.kurt()), 2), "Excess kurtosis of daily returns", float(r.kurt()), 2, page="risk.html", stated="12.98"),
        key_number("ending-value", fmt(float(equity.iloc[-1]), 3, prefix="$"), "Ending value of $1", float(equity.iloc[-1]), 3, prefix="$", page="risk.html", stated="$2.750"),
        key_number("total-cost", fmt(100 * float(risk["result"].costs.sum()), 3, suffix="%"), "Total modeled costs (% of NAV)", 100 * float(risk["result"].costs.sum()), 3, suffix="%", page="risk.html", stated="0.183%"),
        key_number("hit-rate", fmt(100 * float(perf2["hit_rate"]), 0, suffix="%"), "Hit rate (days with positive return)", 100 * float(perf2["hit_rate"]), 0, suffix="%", page="notebooks/06", stated="57%"),
        key_number("rebalances", str(len(risk["result"].turnover)), "Monthly rebalances", len(risk["result"].turnover), 0, page="notebooks/06", stated="127"),
        key_number("backtest-start", iso(r.index[0]), "First net return", page="risk.html", stated="2016-01-04"),
        key_number("backtest-end", iso(r.index[-1]), "Last net return", page="risk.html", stated="2026-07-06"),
    ]

    fc_dates = [f["date"] for f in forecasts]
    fc_breach = [1 if f["breach"] else 0 for f in forecasts]
    fc_ret_on_breach = [sig(100 * f["return"]) if f["breach"] else None for f in forecasts]
    dd_x, dd_y = thin_series(dd.index, 100 * dd.to_numpy())
    rv = roll_vol.dropna()
    rv_x, rv_y = thin_series(rv.index, 100 * rv.to_numpy())
    fc_x_pos = thin_positions(len(forecasts), MAX_POINTS)
    centers = risk["histogram_centers"]
    stress_rules = [{"axis": "x", "at": s["start"], "label": s["label"], "color": "c5"} for s in stresses]
    figures = [
        figure("equity", "line", "Growth of $1 after modeled transaction costs", "$ · daily · 2016-01-04 to 2026-07-06 · notebook 06",
               "Line chart of the cumulative growth of one dollar in the monthly-rebalanced 60/40-plus-alternatives portfolio, net of 10 basis point turnover costs.",
               x={"label": "date", "values": [iso(d) for d in equity.index]}, series=[series("equity", "c1", sigs(equity.to_numpy()))], rules=stress_rules),
        figure("drawdown", "line", "Drawdown from the running peak", "% · daily · notebook 06",
               "Line chart of the portfolio drawdown from its running peak; the 2020 spike and the 2022 trench are the two largest episodes.",
               x={"label": "date", "values": dd_x}, series=[series("drawdown", "c1", dd_y)], yMax=0, rules=stress_rules),
        figure("rolling-vol", "line", "Rolling 63-day annualised volatility", "% · daily, 63-observation window · notebook 06",
               "Line chart of rolling sixty-three day annualised volatility of the portfolio, showing clustering around the 2018, 2020, 2022 and 2025 episodes.",
               x={"label": "date", "values": rv_x}, series=[series("rolling 63d vol", "c1", rv_y)], yMin=0),
        figure("return-histogram", "histogram", "Daily return distribution with a matched Gaussian", "count · daily return (%) · 56 bins · notebook 06",
               "Histogram of daily portfolio returns in percent with the expected counts of a Gaussian of the same mean and variance overlaid, and vertical rules at the historical VaR 95 and ES 95 thresholds.",
               x={"label": "daily return (%)", "values": sigs(centers)},
               series=[series("observed", "c1", [int(c) for c in risk["histogram_counts"]])],
               overlay={"key": "Gaussian, same mean and variance", "color": "c2", "values": sigs(risk["normal_counts"])},
               rules=[{"axis": "x", "at": sig(-100 * var95), "label": "VaR 95", "color": "c5"},
                      {"axis": "x", "at": sig(-100 * es95), "label": "ES 95", "color": "c3"}]),
        figure("breach-strip", "line", "Exception record · one mark per day beyond forecast VaR", "1 = breach · 2,140 forecast days · notebook 06",
               "Timeline strip of the 2,140 one-step forecasts marking each day the realised return fell below the rolling 500-observation historical VaR forecast; clusters of consecutive marks are visible in 2018, 2020 and 2022.",
               x={"label": "date", "values": fc_dates},
               series=[series("breach (1 = yes)", "c1", fc_breach), series("return on breach day (%)", "c3", fc_ret_on_breach)],
               rules=stress_rules,
               table={"columns": ["Date", "Return (%)", "VaR 95 forecast (%)"],
                      "rows": [[f["date"], sig(100 * f["return"]), sig(100 * f["var95"])] for f in forecasts if f["breach"]]}),
        figure("var-forecast", "line", "Rolling 500-observation historical VaR 95 forecast", "% loss · one-day · notebook 06",
               "Line chart of the rolling historical VaR 95 forecast across the 2,140 forecast days; the forecast adapts only as new returns enter the 500-day window.",
               x={"label": "date", "values": [fc_dates[i] for i in fc_x_pos]},
               series=[series("VaR 95 forecast", "c1", [sig(100 * forecasts[i]["var95"]) for i in fc_x_pos])], yMin=0),
        figure("var-by-method", "columns", "The same portfolio, six risk numbers per confidence level", "% of NAV · one day · notebook 06",
               "Grouped columns of historical, parametric, Cornish–Fisher and Monte Carlo VaR plus historical and parametric ES at the 95 and 99 percent levels.",
               x={"label": "estimator", "values": list(var_tab[0.95])},
               series=[series("α = 95%", "c1", [sig(100 * v) for v in var_tab[0.95].values()]),
                       series("α = 99%", "c2", [sig(100 * v) for v in var_tab[0.99].values()])], yMin=0),
        figure("correlation-stress", "columns", "Diversification decay under correlation stress", "% annualised · λ from 0 (observed) to 1 (all correlations → 1) · notebook 06",
               "Columns of annualised portfolio volatility as the correlation matrix is blended toward perfect correlation.",
               x={"label": "λ", "values": sigs(np.linspace(0, 1, 21))},
               series=[series("portfolio volatility", "c1", [sig(100 * v) for v in correlation_stress_vols(risk)])],
               rules=[{"axis": "y", "at": sig(100 * undiversified_vol(risk)), "label": "weighted-average vol (no diversification)", "color": "c5"}], yMin=0),
    ]
    verdict = lambda p: "Reject at 5%" if p < 0.05 else "Do not reject at 5%"  # noqa: E731
    tables = [
        table("coverage-tests", "Coverage and independence tests · rolling 500-observation historical VaR 95",
              ["Test", "Statistic", "p-value", "Verdict"],
              [["Kupiec unconditional coverage", sig(tests["kupiec_lr"]), sig(tests["kupiec_p"]), verdict(tests["kupiec_p"])],
               ["Christoffersen independence", sig(tests["christoffersen_independence_lr"]), sig(tests["christoffersen_independence_p"]), verdict(tests["christoffersen_independence_p"])],
               ["Conditional coverage", sig(tests["christoffersen_conditional_coverage_lr"]), sig(tests["christoffersen_conditional_coverage_p"]), verdict(tests["christoffersen_conditional_coverage_p"])]],
              note=f"{tests['breaches']} breaches over {n_fc:,} forecasts against {tests['expected_breaches']:.0f} expected; likelihood-ratio statistics."),
        table("transitions", "Breach transition counts", ["n00", "n01", "n10", "n11", "Expected n11 under independence"],
              [[tc["n00"], tc["n01"], tc["n10"], tc["n11"], sig(expected_n11)]]),
        table("run-lengths", "Exception run lengths", ["Run length (days)", "Runs"], [[k, v] for k, v in sorted(runs.items())]),
        table("var-by-method", "VaR and ES by estimator (% of NAV, one day)", ["Estimator", "α = 95%", "α = 99%"],
              [[k, sig(100 * var_tab[0.95][k]), sig(100 * var_tab[0.99][k])] for k in var_tab[0.95]]),
        table("bootstrap", "Moving-block bootstrap uncertainty · 2,000 replications, 21-day blocks, seed 20260803",
              ["Measure", "Point estimate (%)", "Bootstrap 95% lower (%)", "Bootstrap 95% upper (%)"],
              [["Historical VaR 95", sig(100 * var95), sig(100 * var_ci[0]), sig(100 * var_ci[1])],
               ["Historical ES 95", sig(100 * es95), sig(100 * es_ci[0]), sig(100 * es_ci[1])]]),
        table("stress-windows", "Fixed historical stress windows within the frozen sample",
              ["Window", "Start", "End", "Cumulative return", "Max drawdown", "Worst day", "VaR breaches"],
              [[s["label"], s["start"], s["end"], sig(s["cumulative_return"]), sig(s["max_drawdown"]), sig(s["worst_day"]), s["var_breaches"]] for s in stress_rows]),
        table("drawdown-episodes", "Five deepest drawdown episodes", ["Depth", "Start", "Trough", "Recovery", "Duration (days)"],
              [[sig(float(row.depth)), iso(row.start), iso(row.trough), iso(row.recovery) if pd.notna(row.recovery) else None, int(row.duration_days)] for row in top5.itertuples()]),
        table("scenarios", "Replaying four historical episodes against today's weights", ["Scenario", "Portfolio P&L", "Assets shocked"],
              [[name, sig(float(row["pnl"])), int(row["n_assets_shocked"])] for name, row in scen.iterrows()]),
        table("weights", "Target weights", ["ETF", "Weight"], [[t, sig(float(w))] for t, w in target.items()]),
        table("summary", "Summary card (rf 2%)", ["Metric", "Value"],
              [[k, sig(float(v))] for k, v in perf2.items()] + [["Sharpe (rf 3%)", sig(sharpe3)], ["Total cost (fraction of NAV)", sig(float(risk["result"].costs.sum()))]]),
    ]
    payload = assemble(common, "var-backtest", page, prose, meta, "notebooks/06_market_risk_dashboard.ipynb",
                       key_numbers, figures, tables, math_note_ids=("fat-tail-var",), ledger_id="market-risk",
                       parameters={"rebalance": "monthly", "warm_up_days": 252, "cost_bps": 10.0, "var_window": risk["window"],
                                   "bootstrap": {"blocks": 21, "replications": 2000, "seed": 20260803}, "monte_carlo_var_seed": SEED})
    return payload


def correlation_stress_vols(risk: dict) -> list[float]:
    asset_rets = to_returns(risk["prices"]).loc[risk["returns"].index]
    cov_ann = asset_rets.cov() * 252
    return [float(np.sqrt(risk["target"] @ correlation_stress(cov_ann, float(l)) @ risk["target"])) for l in np.linspace(0, 1, 21)]


def undiversified_vol(risk: dict) -> float:
    asset_rets = to_returns(risk["prices"]).loc[risk["returns"].index]
    cov_ann = asset_rets.cov() * 252
    return float(risk["target"] @ np.sqrt(np.diag(cov_ann)))


# ---------------------------------------------------------------------------
# 3. factor-etfs  (factor_study + notebook 05)
# ---------------------------------------------------------------------------
def build_factor_etfs(common: Common, fs: dict) -> dict:
    page = "finding-factor-etfs.html"
    prose, meta = finding_prose(page, extra_source_pages=("factors.html",))
    aligned, X, fitted, regs, oos = fs["aligned"], fs["X"], fs["fitted"], fs["regressions"], fs["oos_rows"]
    etfs, factors = list(bfrd.ETF_NAMES), list(bfrd.FACTOR_NAMES)
    ols = {t: sm.OLS(fs["y_by_etf"][t], X).fit() for t in etfs}  # plain-SE t-statistics (notebook 05)
    by_ticker = {r["ticker"]: r for r in regs}
    exposure = {(r["ticker"], e["factor"]): e for r in regs for e in r["exposures"]}
    # rolling QQQ betas with HAC(5) bands at each month end
    roll_dates, roll_params, roll_se = [], {f: [] for f in ("Mkt-RF", "HML", "Mom")}, {f: [] for f in ("Mkt-RF", "HML", "Mom")}
    qqq = fs["y_by_etf"]["QQQ"]
    for date, sl in fs["rolling_windows"]:
        m = sm.OLS(qqq.iloc[sl], X.iloc[sl]).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        roll_dates.append(iso(date))
        for f in roll_params:
            roll_params[f].append(float(m.params[f]))
            roll_se[f].append(float(m.bse[f]))
    corr = aligned[factors].corr()
    decays = {r["ticker"]: -100 * r["decay"] for r in oos}
    survivors = sum(1 for r in regs if r["alpha_survives_5pct_bh"])
    mtum_mom, vlue_hml = exposure[("MTUM", "Mom")], exposure[("VLUE", "HML")]
    qqq = by_ticker["QQQ"]

    key_numbers = [
        key_number("mtum-momentum-beta", fmt(mtum_mom["beta"], 2), "MTUM momentum β", mtum_mom["beta"], 2, page=page, stated="0.34"),
        key_number("mtum-momentum-t", fmt(float(ols["MTUM"].tvalues["Mom"]), 0), "MTUM momentum t-statistic (OLS)", float(ols["MTUM"].tvalues["Mom"]), 0, page=page, stated="49",
                   note=f"Plain-OLS t as printed in notebook 05; HAC(5) t = {mtum_mom['t_hac']:.1f}."),
        key_number("vlue-value-beta", fmt(vlue_hml["beta"], 2), "VLUE value (HML) β", vlue_hml["beta"], 2, page=page, stated="0.31"),
        key_number("vlue-value-t", fmt(float(ols["VLUE"].tvalues["HML"]), 0), "VLUE value t-statistic (OLS)", float(ols["VLUE"].tvalues["HML"]), 0, page=page, stated="29",
                   note=f"Plain-OLS t as printed in notebook 05; HAC(5) t = {vlue_hml['t_hac']:.1f}."),
        key_number("qqq-alpha", fmt(100 * qqq["alpha_annualized"], 2, suffix="%", sign=True), "QQQ annualised alpha", 100 * qqq["alpha_annualized"], 2, suffix="%", page=page, stated="+3.29%"),
        key_number("qqq-alpha-p", fmt(qqq["alpha_p_raw"], 3), "QQQ raw HAC p-value", qqq["alpha_p_raw"], 3, page=page, stated="0.024"),
        key_number("qqq-alpha-q", fmt(qqq["alpha_q_bh"], 3), "QQQ Benjamini–Hochberg q-value", qqq["alpha_q_bh"], 3, page=page, stated="0.147"),
        key_number("alphas-surviving", str(survivors), "alphas surviving the 5% false-discovery threshold", survivors, 0, page=page, stated="0"),
        key_number("observations", f"{qqq['n']:,}", "aligned daily observations", qqq["n"], 0, page=page, stated="2,867"),
        key_number("sample-start", iso(aligned.index[0]), "First aligned observation", page=page, stated="5 January 2015"),
        key_number("sample-end", iso(aligned.index[-1]), "Last aligned observation", page=page, stated="29 May 2026"),
        key_number("holdout-decay-min", fmt(min(decays.values()), 1), "Smallest holdout R² decay (percentage points)", min(decays.values()), 1, page=page, stated="0.4"),
        key_number("holdout-decay-max", fmt(max(decays.values()), 1), "Largest holdout R² decay (percentage points)", max(decays.values()), 1, page=page, stated="29.6"),
        key_number("holdout-decay-max-etf", max(decays, key=decays.get), "ETF with the largest holdout decay", page=page, stated="USMV"),
        key_number("iwm-size-beta", fmt(exposure[("IWM", "SMB")]["beta"], 2), "IWM size (SMB) β", exposure[("IWM", "SMB")]["beta"], 2, page="notebooks/05", stated="0.85"),
        key_number("iwm-r2", fmt(by_ticker["IWM"]["r_squared"], 2), "IWM R²", by_ticker["IWM"]["r_squared"], 2, page="notebooks/05", stated="0.98"),
        key_number("usmv-market-beta", fmt(exposure[("USMV", "Mkt-RF")]["beta"], 2), "USMV market β", exposure[("USMV", "Mkt-RF")]["beta"], 2, page="notebooks/05", stated="0.74"),
        key_number("qqq-market-beta", fmt(exposure[("QQQ", "Mkt-RF")]["beta"], 2), "QQQ market β", exposure[("QQQ", "Mkt-RF")]["beta"], 2, page="notebooks/05", stated="1.11"),
        key_number("qqq-hml-beta", fmt(exposure[("QQQ", "HML")]["beta"], 2), "QQQ value (HML) β", exposure[("QQQ", "HML")]["beta"], 2, page="notebooks/05", stated="-0.31"),
        key_number("qual-rmw-beta", fmt(exposure[("QUAL", "RMW")]["beta"], 2), "QUAL profitability (RMW) β", exposure[("QUAL", "RMW")]["beta"], 2, page="notebooks/05", stated="0.15"),
    ]
    etf_colors = dict(zip(etfs, ("c1", "c2", "c3", "c4", "c5", "c6")))
    alpha_se = {t: float(fitted[t].bse["const"]) * 252 * 100 for t in etfs}
    figures = [
        figure("betas", "dots", "Factor loadings with 95% HAC intervals · six style ETFs", "β · daily excess returns on FF5 + momentum · Newey–West HAC(5) · notebook 05",
               "Dots with interval bands showing each ETF's loading on the market, size, value, profitability, investment and momentum factors.",
               x={"label": "factor", "values": factors},
               series=[series(t, etf_colors[t], [sig(exposure[(t, f)]["beta"]) for f in factors],
                              band=([sig(exposure[(t, f)]["ci_low"]) for f in factors], [sig(exposure[(t, f)]["ci_high"]) for f in factors])) for t in etfs], zero=True),
        figure("alphas", "dots", "Annualised alpha with HAC intervals · six style ETFs", "% per year · 95% HAC(5) interval · notebook 05",
               "Dots with interval bands of annualised alpha for the six ETFs against a zero line; QQQ carries the largest estimate, and no alpha survives Benjamini–Hochberg correction at 5%.",
               x={"label": "ETF", "values": etfs},
               series=[series("annualised alpha", "c1", [sig(100 * by_ticker[t]["alpha_annualized"]) for t in etfs],
                              band=([sig(100 * by_ticker[t]["alpha_annualized"] - 1.96 * alpha_se[t]) for t in etfs],
                                    [sig(100 * by_ticker[t]["alpha_annualized"] + 1.96 * alpha_se[t]) for t in etfs]))], zero=True,
               table={"columns": ["ETF", "Alpha (% p.a.)", "Raw p", "Holm p", "BH q", "Survives 5% FDR"],
                      "rows": [[t, sig(100 * by_ticker[t]["alpha_annualized"]), sig(by_ticker[t]["alpha_p_raw"]), sig(by_ticker[t]["alpha_p_holm"]), sig(by_ticker[t]["alpha_q_bh"]), by_ticker[t]["alpha_survives_5pct_bh"]] for t in etfs]}),
        figure("rolling-qqq", "line", "Rolling 252-day QQQ betas with HAC bands", "β · trailing 252 observations, month-end · 95% HAC(5) band · notebook 05",
               "Lines with bands of QQQ's rolling one-year market, value and momentum loadings; the momentum loading changes sign across the sample.",
               x={"label": "window end", "values": roll_dates},
               series=[series(f, c, sigs(roll_params[f]), band=(sigs(np.array(roll_params[f]) - 1.96 * np.array(roll_se[f])), sigs(np.array(roll_params[f]) + 1.96 * np.array(roll_se[f]))))
                       for f, c in (("Mkt-RF", "c1"), ("HML", "c2"), ("Mom", "c3"))],
               rules=[{"axis": "y", "at": sig(float(fs["qqq_full"].params[f])), "label": f"full-sample {f}", "color": c} for f, c in (("Mkt-RF", "c1"), ("HML", "c2"), ("Mom", "c3"))]),
        figure("r2-holdout", "columns", "In-sample versus holdout R² · chronological 50/50 split", "R² (%) · coefficients frozen on the first half · notebook 05",
               "Grouped columns comparing training R-squared with holdout R-squared for each ETF; USMV decays most.",
               x={"label": "ETF", "values": etfs},
               series=[series("training R²", "c1", [sig(100 * r["in_sample_r2"]) for r in oos]), series("holdout R²", "c2", [sig(100 * r["out_of_sample_r2"]) for r in oos])], yMin=0, yMax=100),
    ]
    tables = [
        table("exposures", "Factor exposures · OLS coefficients with HAC(5) intervals and plain-OLS t", ["ETF", "Factor", "β", "CI low", "CI high", "HAC t", "OLS t"],
              [[t, f, sig(exposure[(t, f)]["beta"]), sig(exposure[(t, f)]["ci_low"]), sig(exposure[(t, f)]["ci_high"]), sig(exposure[(t, f)]["t_hac"]), sig(float(ols[t].tvalues[f]))] for t in etfs for f in factors]),
        table("alpha-tests", "ETF alpha tests with family-wise and false-discovery corrections", ["ETF", "Alpha (annualised)", "Alpha HAC SE (annualised)", "Raw p", "Holm adjusted p", "BH q", "Survives BH 5%", "R²", "n"],
              [[t, sig(by_ticker[t]["alpha_annualized"]), sig(alpha_se[t] / 100), sig(by_ticker[t]["alpha_p_raw"]), sig(by_ticker[t]["alpha_p_holm"]), sig(by_ticker[t]["alpha_q_bh"]), by_ticker[t]["alpha_survives_5pct_bh"], sig(by_ticker[t]["r_squared"]), by_ticker[t]["n"]] for t in etfs]),
        table("holdout", "Chronological holdout results; coefficients fitted only on the first half", ["ETF", "Training R²", "Holdout R²", "Change", "Training window", "Holdout window", "Holdout RMSE (daily)"],
              [[r["ticker"], sig(r["in_sample_r2"]), sig(r["out_of_sample_r2"]), sig(r["decay"]), f"{r['train_start']} – {r['train_end']}", f"{r['test_start']} – {r['test_end']}", sig(r["test_rmse_daily"])] for r in oos]),
        table("factor-correlations", "Daily factor correlations, 2015–2026", ["Factor"] + factors,
              [[f] + [sig(float(corr.loc[f, g])) for g in factors] for f in factors]),
        table("rolling-ranges", "Rolling QQQ beta ranges across windows", ["Factor", "Min", "Max"],
              [[f, sig(min(roll_params[f])), sig(max(roll_params[f]))] for f in roll_params]),
    ]
    payload = assemble(common, "factor-etfs", page, prose, meta, "notebooks/05_factor_investing.ipynb", key_numbers, figures, tables,
                       ledger_id="factor-exposures",
                       parameters={"factors": factors, "hac_lags": 5, "rolling_window": 252, "holdout": "chronological 50/50 split",
                                   "multiple_testing": ["holm", "fdr_bh"], "factor_source": bfrd.FACTOR_URL})
    return payload


# ---------------------------------------------------------------------------
# 4. volatility  (notebook 03 + case-study.js)
# ---------------------------------------------------------------------------
def build_volatility(common: Common) -> dict:
    page = "finding-volatility-smile.html"
    case_sections = page_sections("case-study.html", only_ids={"case-model", "case-refutation", "case-repair", "case-comparison", "case-residue", "case-conclusion"})
    prose, meta = finding_prose(page, extra_source_pages=("case-study.html",), extra_sections=case_sections)
    spy = to_returns(load_prices(["SPY"]))["SPY"]
    ann_vol = float(spy.std() * np.sqrt(252))
    roll = (spy.rolling(21).std() * np.sqrt(252)).dropna()
    r_ = spy.dropna()
    skew, exkurt = float(stats.skew(r_)), float(stats.kurtosis(r_))
    worst_sigma = float(r_.min() / r_.std())
    p_gauss = float(stats.norm.sf(abs(worst_sigma)))
    lags = np.arange(1, 61)
    acf_abs = [float(r_.abs().autocorr(int(l))) for l in lags]
    acf_raw = [float(r_.autocorr(int(l))) for l in lags]
    ci = 1.96 / np.sqrt(len(r_))
    # Heston smile (notebook 03, section 3)
    S0, r = 100.0, 0.03
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.045, xi=0.35, rho=-0.70)
    strikes = np.linspace(70, 130, 25)
    mats = [0.083, 0.25, 0.5, 1.0, 2.0]
    smile = pd.DataFrame(index=strikes, columns=mats, dtype=float)
    for T in mats:
        for K in strikes:
            smile.loc[K, T] = implied_vol(heston_price(S0, float(K), T, r, params), S0, float(K), T, r)
    sigma_atm = float(smile.loc[100.0, 0.25])
    tab_strikes, tab_mats = [70, 85, 100, 115, 130], [0.25, 0.5, 1.0, 2.0]
    price_diff, vol_diff = {}, {}
    for T in tab_mats:
        for K in tab_strikes:
            h = heston_price(S0, float(K), T, r, params)
            b = bs_price(S0, float(K), T, r, sigma_atm)
            price_diff[(T, K)] = h - b
            vol_diff[(T, K)] = implied_vol(h, S0, float(K), T, r) - sigma_atm
    # case-study.js arrays regenerated (S0=100, r=2%, T=0.5; Feller violated by design)
    cs = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)
    cs_strikes = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]
    cs_calls = [heston_price(100.0, float(K), 0.5, 0.02, cs) for K in cs_strikes]
    cs_ivs = [implied_vol(c, 100.0, float(K), 0.5, 0.02) for c, K in zip(cs_calls, cs_strikes)]
    flat = 0.192
    bs_calls = [bs_price(100.0, float(K), 0.5, 0.02, flat) for K in cs_strikes]
    hist_counts, hist_edges = np.histogram(r_.to_numpy() * 100, bins=120)
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    width = hist_edges[1] - hist_edges[0]
    normal_counts = len(r_) * width / 100 * stats.norm.pdf(hist_centers / 100, r_.mean(), r_.std())
    skew_lo, skew_hi = float(smile.loc[80.0, 0.25]), float(smile.loc[120.0, 0.25])

    key_numbers = [
        key_number("realised-vol-min", fmt(100 * float(roll.min()), 1, suffix="%"), "Minimum 21-day realised vol of SPY", 100 * float(roll.min()), 1, suffix="%", page=page, stated="≈3% (README: 3.4%)"),
        key_number("realised-vol-max", fmt(100 * float(roll.max()), 0, suffix="%"), "Maximum 21-day realised vol of SPY", 100 * float(roll.max()), 0, suffix="%", page=page, stated="93%"),
        key_number("realised-vol-ratio", fmt(float(roll.max() / roll.min()), 0, suffix="×"), "Max / min realised vol", float(roll.max() / roll.min()), 0, suffix="×", page=page, stated="nearly thirty-fold"),
        key_number("full-sample-vol", fmt(100 * ann_vol, 1, suffix="%"), "Full-sample annualised vol (the 'constant' σ)", 100 * ann_vol, 1, suffix="%", page="notebooks/03"),
        key_number("excess-kurtosis", "≈" + fmt(exkurt, 0), "Excess kurtosis of SPY daily returns", exkurt, 0, page=page, stated="about 14"),
        key_number("skewness", fmt(skew, 2), "Skewness of SPY daily returns", skew, 2, page="notebooks/03"),
        key_number("worst-day", fmt(100 * float(r_.min()), 1, suffix="%"), "Worst daily return", 100 * float(r_.min()), 1, suffix="%", page="notebooks/03", stated="-10.9% (March 2020)"),
        key_number("worst-day-sigma", "≈" + fmt(abs(worst_sigma), 0, suffix="σ"), "Worst day under a Gaussian yardstick", abs(worst_sigma), 0, suffix="σ", page=page, stated="≈10σ",
                   note=f"Unrounded {abs(worst_sigma):.2f}σ; Gaussian single-day probability {p_gauss:.2e}."),
        key_number("worst-day-gaussian-p", f"{p_gauss:.2e}", "Gaussian probability of the worst day", page="notebooks/03", stated="around 10^-23"),
        key_number("abs-acf-lag1", fmt(acf_abs[0], 2), "Lag-1 autocorrelation of absolute returns", acf_abs[0], 2, page=page, stated="about 0.35"),
        key_number("acf-lag1", fmt(acf_raw[0], 2), "Lag-1 autocorrelation of returns", acf_raw[0], 2, page="notebooks/03", stated="≈ -0.12"),
        key_number("skew-low-strike", "≈" + fmt(100 * skew_lo, 0, suffix="%"), "Heston implied vol at K/S₀ = 0.8, three months", 100 * skew_lo, 0, suffix="%", page=page, stated="25%"),
        key_number("skew-high-strike", "≈" + fmt(100 * skew_hi, 0, suffix="%"), "Heston implied vol at K/S₀ = 1.2, three months", 100 * skew_hi, 0, suffix="%", page=page, stated="16%"),
        key_number("feller-lhs", fmt(2 * params.kappa * params.theta, 3), "Feller condition 2κθ", 2 * params.kappa * params.theta, 3, page="notebooks/03", stated="0.180"),
        key_number("feller-rhs", fmt(params.xi**2, 3), "Feller condition ξ²", params.xi**2, 3, page="notebooks/03", stated="0.122"),
        key_number("sigma-atm-3m", fmt(100 * sigma_atm, 2, suffix="%"), "Heston ATM implied vol at three months (BS calibration point)", 100 * sigma_atm, 2, suffix="%", page="notebooks/03", stated="19.91%"),
        key_number("case-study-atm-iv", fmt(100 * cs_ivs[6], 1, suffix="%"), "ATM implied vol of the case-study Heston surface (flat BS σ)", 100 * cs_ivs[6], 1, suffix="%", page="case-study.html", stated="19.2%"),
        key_number("observations", f"{len(r_):,}", "SPY daily observations", len(r_), 0, page=page, stated="≈2,900"),
        key_number("sample-start", iso(r_.index[0]), "First SPY return", page=page, stated="January 2015"),
        key_number("sample-end", iso(r_.index[-1]), "Last SPY return", page=page, stated="July 2026"),
    ]
    rv_x, rv_y = thin_series(roll.index, 100 * roll.to_numpy())
    mat_colors = ("c1", "c2", "c3", "c4", "c5")
    figures = [
        figure("realised-vol", "line", "SPY realised volatility: the constant that isn't", "% annualised · rolling 21-day · 2015-02 to 2026-07 · notebook 03",
               "Line chart of annualised twenty-one day realised volatility of SPY, ranging from about three percent in 2017 to about ninety-three percent in March 2020, against the full-sample constant.",
               x={"label": "date", "values": rv_x}, series=[series("21-day realised vol", "c1", rv_y)],
               rules=[{"axis": "y", "at": sig(100 * ann_vol), "label": "full-sample vol", "color": "c5"}], yMin=0,
               annotations=[{"x": iso(roll.idxmin()), "y": sig(100 * float(roll.min())), "text": "minimum"}, {"x": iso(roll.idxmax()), "y": sig(100 * float(roll.max())), "text": "maximum"}]),
        figure("return-histogram", "histogram", "Fat tails: SPY daily returns against a matched Gaussian", "count · daily return (%) · 120 bins · notebook 03",
               "Histogram of SPY daily returns with the expected counts of a Gaussian of the same mean and volatility; the empirical tails sit far above the Gaussian.",
               x={"label": "daily return (%)", "values": sigs(hist_centers)}, series=[series("observed", "c1", [int(c) for c in hist_counts])],
               overlay={"key": "Gaussian, same mean and vol", "color": "c2", "values": sigs(normal_counts)}),
        figure("acf", "columns", "Volatility clustering: |r| remembers, r forgets", "autocorrelation · lags 1–60 days · notebook 03",
               "Columns of the autocorrelation of absolute returns and of raw returns by lag, with the 95% confidence band; absolute returns stay significantly autocorrelated past lag fifty.",
               x={"label": "lag (days)", "values": [int(l) for l in lags]},
               series=[series("ACF of |returns|", "c1", sigs(acf_abs)), series("ACF of returns", "c2", sigs(acf_raw))],
               rules=[{"axis": "y", "at": sig(ci), "label": "+1.96/√n", "color": "c7"}, {"axis": "y", "at": sig(-ci), "label": "−1.96/√n", "color": "c7"}], zero=True),
        figure("heston-smile", "line", "The smile Black–Scholes cannot draw: Heston-implied vols by strike", "implied vol (%) · S₀ = 100, r = 3% · Feller-satisfying parameters · notebook 03",
               "Lines of Black–Scholes implied volatility by strike for five maturities generated by a single Heston specification, against the flat twenty percent line a constant-volatility model implies.",
               x={"label": "strike K", "values": sigs(strikes)},
               series=[series(f"T = {T:g}y", c, sigs(100 * smile[T].to_numpy())) for T, c in zip(mats, mat_colors)],
               rules=[{"axis": "y", "at": sig(100 * math.sqrt(params.v0)), "label": "Black–Scholes flat √v₀ = 20%", "color": "c7"}]),
        figure("bs-vs-heston-calls", "line", "Six-month call prices across strikes: flat Black–Scholes against Heston", "call price · S₀ = 100, r = 2%, T = 0.5y · BS σ = 19.2% · case study",
               "Two lines of six-month European call prices by strike: Heston with the case-study parameters and Black–Scholes at the flat at-the-money volatility; the wings disagree.",
               x={"label": "strike K", "values": cs_strikes},
               series=[series("Heston", "c1", sigs(cs_calls)), series("Black–Scholes, flat σ = 19.2%", "c2", sigs(bs_calls))],
               table={"columns": ["Strike", "Heston call", "Heston implied vol", "BS call (flat σ)", "IV residual", "BS − Heston"],
                      "rows": [[K, sig(h), sig(iv), sig(b), sig(flat - iv), sig(b - h)] for K, h, iv, b in zip(cs_strikes, cs_calls, cs_ivs, bs_calls)]}),
    ]
    tables = [
        table("moments", "Return moments of SPY daily returns", ["Statistic", "Value"],
              [["Observations", len(r_)], ["Mean (daily)", sig(float(r_.mean()))], ["Std (daily)", sig(float(r_.std()))], ["Annualised vol", sig(ann_vol)],
               ["Skewness", sig(skew)], ["Excess kurtosis", sig(exkurt)], ["Worst day", sig(float(r_.min()))], ["Worst day date", iso(r_.idxmin())],
               ["Worst day in σ", sig(worst_sigma)], ["Gaussian probability of worst day", sig(p_gauss)], ["Lag-1 ACF of |r|", sig(acf_abs[0])], ["Lag-1 ACF of r", sig(acf_raw[0])],
               ["Min 21-day realised vol", sig(float(roll.min()))], ["Max 21-day realised vol", sig(float(roll.max()))]]),
        table("heston-parameters", "Heston parameters used", ["Specification", "S₀", "r", "v₀", "κ", "θ", "ξ", "ρ", "2κθ", "ξ²", "Feller satisfied"],
              [["Smile by maturity (notebook 03)", S0, r, params.v0, params.kappa, params.theta, params.xi, params.rho, sig(2 * params.kappa * params.theta), sig(params.xi**2), params.feller_satisfied],
               ["Six-month price comparison (case study)", 100.0, 0.02, cs.v0, cs.kappa, cs.theta, cs.xi, cs.rho, sig(2 * cs.kappa * cs.theta), sig(cs.xi**2), cs.feller_satisfied]],
              note="The case-study specification violates Feller by design; its prices come from the semi-analytic characteristic-function integral, not simulation."),
        table("smile", "Heston-implied volatility by strike and maturity", ["Strike"] + [f"T = {T:g}y" for T in mats],
              [[sig(K)] + [sig(float(smile.loc[K, T])) for T in mats] for K in strikes]),
        table("divergence-price", "Heston − Black-Scholes price difference (per S₀ = 100), BS calibrated to σ_ATM(3m)", ["T (years)"] + [str(K) for K in tab_strikes],
              [[T] + [sig(price_diff[(T, K)]) for K in tab_strikes] for T in tab_mats]),
        table("divergence-vol", "Same gap in implied-vol points", ["T (years)"] + [str(K) for K in tab_strikes],
              [[T] + [sig(100 * vol_diff[(T, K)]) for K in tab_strikes] for T in tab_mats]),
    ]
    payload = assemble(common, "volatility", page, prose, meta, "notebooks/03_derivatives_pricing.ipynb", key_numbers, figures, tables,
                       math_note_ids=("ito", "gbm", "risk-neutral-pricing", "feynman-kac"), ledger_id="stochastic-volatility",
                       parameters={"underlying": "SPY", "rolling_window": 21, "seed": SEED, "heston_integration": "128-node Gauss–Legendre, little-trap formulation"})
    return payload


# ---------------------------------------------------------------------------
# 5. monte-carlo  (notebook 04)
# ---------------------------------------------------------------------------
def build_monte_carlo(common: Common) -> dict:
    page = "finding-monte-carlo-error.html"
    sim_sections = page_sections("simulations.html", only_ids={"prose-gbm", "prose-ou", "prose-merton", "prose-heston", "prose-regime"})
    prose, meta = finding_prose(page, extra_sections=sim_sections)
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    bs_ref = bs_price(S0, K, T, r, sigma)
    # section 1: exact vs Euler with common random numbers
    N = 400_000
    bias_rows = []
    for n_steps in (1, 2, 4, 12, 52):
        rng = np.random.default_rng(SEED)
        Z = rng.standard_normal((N, n_steps))
        dt = T / n_steps
        st_exact = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))[:, -1]
        s = np.full(N, S0)
        for k in range(n_steps):
            s = s * (1.0 + r * dt + sigma * np.sqrt(dt) * Z[:, k])
        p_ex, se_ex = mc_price(np.maximum(st_exact - K, 0), r, T)
        p_eu, _ = mc_price(np.maximum(s - K, 0), r, T)
        bias_rows.append({"n_steps": n_steps, "exact": p_ex, "exact_bias": p_ex - bs_ref, "euler": p_eu, "euler_bias": p_eu - bs_ref, "se": se_ex})
    # section 2: antithetic variates, 30 replications at 20,000 paths
    anti = {}
    for flag in (False, True):
        ps = []
        for rep in range(30):
            rng = np.random.default_rng(1000 + rep)
            st = simulate_gbm(S0, r, sigma, T, n_steps=1, n_paths=20_000, rng=rng, antithetic=flag)[:, -1]
            ps.append(mc_price(np.maximum(st - K, 0), r, T)[0])
        anti[flag] = {"mean": float(np.mean(ps)), "se": float(np.std(ps, ddof=1)), "bias": float(np.mean(ps) - bs_ref)}
    ratio = anti[False]["se"] / anti[True]["se"]
    # section 6: the N^-1/2 law, 16 replications per path count (plus the same protocol with antithetic pairs)
    Ns = np.array([250, 1000, 4000, 16000, 64000, 256000])
    n_reps = 16
    rmse, rmse_anti = [], []
    for n in Ns:
        errs, errs_a = [], []
        for rep in range(n_reps):
            rng = np.random.default_rng(7000 + rep)
            st = simulate_gbm(S0, r, sigma, T, n_steps=1, n_paths=int(n), rng=rng)[:, -1]
            errs.append(mc_price(np.maximum(st - K, 0), r, T)[0] - bs_ref)
            rng = np.random.default_rng(7000 + rep)
            st = simulate_gbm(S0, r, sigma, T, n_steps=1, n_paths=int(n), rng=rng, antithetic=True)[:, -1]
            errs_a.append(mc_price(np.maximum(st - K, 0), r, T)[0] - bs_ref)
        rmse.append(float(np.sqrt(np.mean(np.square(errs)))))
        rmse_anti.append(float(np.sqrt(np.mean(np.square(errs_a)))))
    rmse = np.array(rmse)
    (slope, intercept), cov = np.polyfit(np.log(Ns), np.log(rmse), 1, cov=True)
    slope_se_ols = float(np.sqrt(cov[0, 0]))
    # replication noise: an RMSE over n independent errors has relative SE ≈ 1/sqrt(2(n-1)),
    # so each log-RMSE point carries that noise and the fitted slope inherits it.
    log_n = np.log(Ns)
    slope_se = float((1.0 / math.sqrt(2 * (n_reps - 1))) / math.sqrt(float(((log_n - log_n.mean()) ** 2).sum())))
    reference = rmse[0] * np.sqrt(Ns[0] / Ns)
    law_n = [1_000, 10_000, 100_000]
    law = [math.sqrt(law_n[0] / n) for n in law_n]
    # section 3: OU half-life recovery
    kappa, theta_ou, sigma_ou, x0 = 2.0, 0.0, 0.5, 2.0
    rng = np.random.default_rng(SEED)
    X = simulate_ou(x0, kappa, theta_ou, sigma_ou, T=3.0, n_steps=756, n_paths=3000, rng=rng)
    dt_ou = 3.0 / 756
    rho_hat = np.polyfit(X[:, :-1].ravel(), X[:, 1:].ravel(), 1)[0]
    kappa_hat = float(-np.log(rho_hat) / dt_ou)
    # section 4: CIR boundary behaviour
    cir_rows = []
    for name, p in (("Feller satisfied", dict(kappa=2.0, theta=0.04, sigma=0.30)), ("Feller violated", dict(kappa=0.5, theta=0.04, sigma=0.45))):
        rng = np.random.default_rng(SEED)
        V = simulate_cir(0.04, p["kappa"], p["theta"], p["sigma"], T=5.0, n_steps=1260, n_paths=2000, rng=rng)
        cir_rows.append([name, p["kappa"], p["theta"], p["sigma"], sig(2 * p["kappa"] * p["theta"]), sig(p["sigma"] ** 2), sig(float((V == 0.0).any(axis=1).mean()))])
    # section 5: regime switching
    mus, sigmas = [0.12, -0.25], [0.12, 0.38]
    P = np.array([[0.995, 0.005], [0.030, 0.970]])
    rng = np.random.default_rng(SEED)
    S_rs, regimes = simulate_regime_switching(100.0, mus, sigmas, P, T=5.0, n_steps=1260, n_paths=4000, rng=rng)
    lr = np.log(S_rs[:, -1] / 100.0)
    crisis_share = float((regimes == 1).mean())
    stationary = float(P[0, 1] / (P[0, 1] + P[1, 0]))
    # section 7: discretely monitored barrier
    rng = np.random.default_rng(SEED)
    paths = simulate_gbm(S0, r, sigma, T, n_steps=252, n_paths=40_000, rng=rng, antithetic=True)
    vanilla = np.maximum(paths[:, -1] - K, 0)
    barrier_rows = [["vanilla call (BS closed form)", sig(bs_ref), None, 1.0]]
    van_mc, van_se = mc_price(vanilla, r, T)
    barrier_rows.append(["vanilla call (MC, same paths)", sig(van_mc), sig(van_se), 1.0])
    for label, stride in (("up-and-out call, B=130, daily monitoring (252/yr)", 1), ("up-and-out call, B=130, monthly monitoring (12/yr)", 21)):
        alive = paths[:, ::stride].max(axis=1) < 130.0
        p, se = mc_price(vanilla * alive, r, T)
        barrier_rows.append([label, sig(p), sig(se), sig(float(alive.mean()))])

    within_noise = abs(slope + 0.5) <= 2 * slope_se
    key_numbers = [
        key_number("error-law", "N⁻¹ᐟ²", "Error rate of the Monte Carlo estimator", page=page),
        key_number("paths-for-tenfold", "≈100×", "paths needed for 10× less error", 100, 0, prefix="≈", suffix="×", page=page, stated="≈100× paths"),
        key_number("fitted-slope", fmt(float(slope), 2), "Fitted log–log RMSE slope across 16 replications", float(slope), 2, page=page, stated="within noise of −0.5",
                   unverified=not within_noise, note=f"Replication-noise SE of the slope {slope_se:.3f} (OLS residual SE {slope_se_ols:.3f}); |slope + 0.5| = {abs(slope + 0.5):.3f} = {abs(slope + 0.5) / slope_se:.2f} SE."),
        key_number("antithetic-se-reduction", "≈30%", "Antithetic SE reduction at the same 20,000-path budget", 30, 0, suffix="%", page=page, stated="≈30%",
                   note=f"SE ratio independent / antithetic = {ratio:.2f} across 30 replications (equivalent to {ratio**2:.1f}× more paths); the notebook reports this ratio as 'about 30%'."),
        key_number("antithetic-se-ratio", fmt(ratio, 2, suffix="×"), "Empirical SE, independent ÷ antithetic", ratio, 2, suffix="×", page="notebooks/04", stated="1.30x"),
        key_number("primary-seed", "42", "Primary seed (PCG64)", 42, 0, page=page, stated="42"),
        key_number("antithetic-budget", "20,000", "Paths per replication in the antithetic test", 20_000, 0, page=page, stated="20,000"),
        key_number("antithetic-replications", "30", "Independent replications in the antithetic test", 30, 0, page=page, stated="30"),
        key_number("law-1e3", fmt(law[0], 2), "Relative error at 10³ paths (N⁻¹ᐟ² law, normalised)", law[0], 2, page=page, stated="1.00"),
        key_number("law-1e4", fmt(law[1], 2), "Relative error at 10⁴ paths", law[1], 2, page=page, stated="0.32"),
        key_number("law-1e5", fmt(law[2], 2), "Relative error at 10⁵ paths", law[2], 2, page=page, stated="0.10"),
        key_number("ou-kappa-recovered", fmt(kappa_hat, 3), "OU κ recovered from the AR(1) fit (true 2.0)", kappa_hat, 3, page="notebooks/04", stated="2.011"),
        key_number("regime-crisis-share", fmt(100 * crisis_share, 1, suffix="%"), "Time spent in the crisis regime", 100 * crisis_share, 1, suffix="%", page="notebooks/04", stated="14.0%"),
    ]
    next(k for k in key_numbers if k["id"] == "fitted-slope")["noise_se"] = sig(slope_se, 4)
    figures = [
        figure("se-by-paths", "line", "Monte Carlo error decays like 1/√N", "RMSE of the price estimate · 16 replications per N · seeds 7000 + rep · notebook 04",
               "Lines of root-mean-square pricing error against path count on a logarithmic axis for independent and antithetic sampling, against the inverse-square-root reference line.",
               x={"label": "number of paths N (log scale)", "scale": "log", "values": [int(n) for n in Ns]},
               series=[series("RMSE, independent draws", "c1", sigs(rmse)), series("RMSE, antithetic pairs", "c2", sigs(rmse_anti)),
                       series("N⁻¹ᐟ² reference", "c7", sigs(reference), dash=True)], yMin=0),
        figure("bias-by-step", "line", "Discretisation bias: exact scheme versus Euler–Maruyama", "price − closed form · 400,000 common random numbers · seed 42 · notebook 04",
               "Lines of the pricing bias of the exact GBM scheme and of Euler–Maruyama against the number of time steps; the exact scheme sits inside its standard-error band while Euler bias shrinks linearly with the step.",
               x={"label": "time steps per year", "values": [b["n_steps"] for b in bias_rows]},
               series=[series("exact scheme bias", "c1", sigs([b["exact_bias"] for b in bias_rows]), band=(sigs([-b["se"] for b in bias_rows]), sigs([b["se"] for b in bias_rows]))),
                       series("Euler bias", "c2", sigs([b["euler_bias"] for b in bias_rows]))], zero=True),
        figure("antithetic", "columns", "Antithetic variates at an equal 20,000-path budget", "empirical SE across 30 replications · seeds 1000 + rep · notebook 04",
               "Two columns comparing the empirical standard error of the price estimate with independent draws and with antithetic pairs.",
               x={"label": "sampling", "values": ["independent", "antithetic"]},
               series=[series("empirical SE", "c1", [sig(anti[False]["se"]), sig(anti[True]["se"])])], yMin=0),
        figure("error-law", "line", "Relative sampling error · N⁻¹ᐟ² law, normalised at 10³ paths", "relative error · theoretical law · finding page",
               "Line of the theoretical inverse-square-root law normalised to one at one thousand paths, falling to 0.32 at ten thousand and 0.10 at one hundred thousand paths.",
               x={"label": "paths (log scale)", "scale": "log", "values": law_n}, series=[series("relative error", "c1", sigs(law), dash=True)], yMin=0),
    ]
    tables = [
        table("bias", "Exact versus Euler GBM against the closed-form call (S₀ = K = 100, r = 5%, σ = 20%, T = 1)",
              ["Steps", "Exact scheme", "Exact bias", "Euler", "Euler bias", "MC std err"],
              [[b["n_steps"], sig(b["exact"]), sig(b["exact_bias"]), sig(b["euler"]), sig(b["euler_bias"]), sig(b["se"])] for b in bias_rows]),
        table("antithetic", "Antithetic variates across 30 independent replications", ["Antithetic", "Mean estimate", "Empirical SE", "Bias vs BS"],
              [[flag, sig(anti[flag]["mean"]), sig(anti[flag]["se"]), sig(anti[flag]["bias"])] for flag in (False, True)],
              note=f"SE reduction factor {ratio:.2f}× at equal path count."),
        table("convergence", "RMSE by path count", ["N", "RMSE (independent)", "RMSE (antithetic)", "N⁻¹ᐟ² reference"],
              [[int(n), sig(a), sig(b), sig(c)] for n, a, b, c in zip(Ns, rmse, rmse_anti, reference)],
              note=f"Fitted log–log slope {slope:.3f}; replication-noise SE {slope_se:.3f}, OLS SE {slope_se_ols:.3f}."),
        table("process-checks", "Process checks", ["Check", "Value", "Reference"],
              [["OU κ recovered from AR(1) fit", sig(kappa_hat), 2.0], ["OU half-life recovered (years)", sig(math.log(2) / kappa_hat), sig(math.log(2) / kappa)],
               ["Regime model: share of time in crisis", sig(crisis_share), sig(stationary)], ["Regime model: terminal log-return skew", sig(float(stats.skew(lr))), 0.0],
               ["Regime model: terminal log-return excess kurtosis", sig(float(stats.kurtosis(lr))), 0.0]]),
        table("cir", "CIR boundary behaviour (2,000 paths, 5 years, full-truncation Euler)", ["Case", "κ", "θ", "σ", "2κθ", "σ²", "Fraction of paths touching zero"], cir_rows),
        table("barrier", "A payoff with no closed form: the discretely monitored up-and-out call (40,000 antithetic paths, 252 steps)", ["Contract", "Price", "Std err", "P(survive)"], barrier_rows),
    ]
    payload = assemble(common, "monte-carlo", page, prose, meta, "notebooks/04_monte_carlo_simulation.ipynb", key_numbers, figures, tables,
                       math_note_ids=("monte-carlo-error",), ledger_id="monte-carlo",
                       seeds={"primary": SEED, "antithetic_replications": "1000 + replicate", "convergence_replications": "7000 + replicate", "generator": "PCG64 (numpy default_rng)"},
                       parameters={"contract": {"S0": S0, "K": K, "r": r, "sigma": sigma, "T": T, "bs_price": sig(bs_ref)}, "bias_paths": N, "convergence_replications": n_reps})
    return payload


# ---------------------------------------------------------------------------
# 6. pricing-validation  (notebook 01)
# ---------------------------------------------------------------------------
def build_pricing_validation(common: Common) -> dict:
    entry = common.ledger["pricing-validation"]
    f = entry["fields_html"]
    order = ["Research question", "Testable claim", "Assumptions", "Dataset / version / seed", "Method", "Falsification criterion", "Result and uncertainty"]
    ledger_html = "".join(f"<p><strong>{html_mod.escape(k)}</strong> {f[k][3:-4] if f[k].startswith('<p>') and f[k].endswith('</p>') and f[k].count('<p>') == 1 else f[k]}</p>" for k in order if k in f)
    pricing_sections = page_sections("asset-pricing.html")
    prose = {
        "decision_summary_html": f"<h3>{html_mod.escape(entry['title'])}</h3>" + f["Result and uncertainty"],
        "sections": [{"id": "ledger", "heading": entry["number"], "html": ledger_html, "page": "/lab/research-log#pricing-validation"}] + pricing_sections,
        "limitations_html": f["Failure modes"],
        "sources": bibliography(read_page("asset-pricing.html")),
    }
    meta = {"title": entry["title"], "lede": None, "dates": page_dates(read_page("research-log.html")), "revisions": [], "decision_heading": entry["title"]}
    S0, r, q, sigma, T = 100.0, 0.05, 0.0, 0.20, 1.0
    rng = np.random.default_rng(SEED)
    ST = simulate_gbm(S0, r - q, sigma, T, n_steps=1, n_paths=200_000, rng=rng, antithetic=True)[:, -1]
    tol = {"Binomial(1000)": 1e-2, "Crank-Nicolson": 1e-2, "Monte Carlo": 3.0}
    rows, err_rows, parity_rows = [], [], []
    labels = []
    for K in (90.0, 100.0, 110.0):
        prices = {}
        for opt in ("call", "put"):
            analytic = bs_price(S0, K, T, r, sigma, option_type=opt, q=q)
            tree = binomial_price(S0, K, T, r, sigma, n_steps=1000, option_type=opt, q=q)
            pde = cn_bs_price(S0, K, T, r, sigma, option_type=opt, n_s=400, n_t=400)
            payoff = np.maximum(ST - K, 0.0) if opt == "call" else np.maximum(K - ST, 0.0)
            mc, se = mc_price(payoff, r, T)
            prices[opt] = {"Black-Scholes": analytic, "Binomial(1000)": tree, "Crank-Nicolson": pde, "Monte Carlo": mc}
            label = f"K = {K:.0f} {opt}"
            labels.append(label)
            err_rows.append([abs(tree - analytic), abs(pde - analytic), abs(mc - analytic), se])
            for method, price in (("Black-Scholes", analytic), ("Binomial(1000)", tree), ("Crank-Nicolson", pde), ("Monte Carlo", mc)):
                err = abs(price - analytic)
                if method == "Black-Scholes":
                    budget, passed, unit = None, True, ""
                elif method == "Monte Carlo":
                    budget, passed, unit = tol[method], err / se <= tol[method], "× SE"
                else:
                    budget, passed, unit = tol[method], err <= tol[method], "abs"
                rows.append([label, method, sig(price), sig(err), sig(se) if method == "Monte Carlo" else None,
                             sig(err / se) if method == "Monte Carlo" else None, f"{budget} {unit}".strip() if budget else "reference", "pass" if passed else "fail"])
        target = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        for method in ("Black-Scholes", "Binomial(1000)", "Crank-Nicolson", "Monte Carlo"):
            viol = prices["call"][method] - prices["put"][method] - target
            if method == "Monte Carlo":
                se_par = np.exp(-r * T) * float(np.std(ST, ddof=1)) / np.sqrt(len(ST))
                budget, passed = f"{3.0} × SE ({se_par:.4f})", abs(viol) <= 3.0 * se_par
            elif method == "Black-Scholes":
                budget, passed = "1e-08 abs", abs(viol) <= 1e-8
            else:
                budget, passed = f"{tol[method]} abs (method budget)", abs(viol) <= tol[method]
            parity_rows.append([sig(K), method, sig(prices["call"][method] - prices["put"][method]), sig(target), sig(viol), budget, "pass" if passed else "fail"])
    err = np.array(err_rows)
    ns = np.unique(np.geomspace(10, 2000, 40).astype(int))
    bs_atm = bs_price(S0, 100.0, T, r, sigma)
    crr_err = np.array([abs(binomial_price(S0, 100.0, T, r, sigma, n_steps=int(n)) - bs_atm) for n in ns])
    sig_true = np.linspace(0.08, 0.60, 27)
    iv_strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    recovered = np.array([[implied_vol(bs_price(S0, K, 0.5, r, s), S0, K, 0.5, r) for s in sig_true] for K in iv_strikes])
    iv_err = float(np.nanmax(np.abs(recovered - sig_true[None, :])))
    det_parity = max(abs(row[4]) for row in parity_rows if row[1] != "Monte Carlo")
    all_pass = all(row[-1] == "pass" for row in rows) and all(row[-1] == "pass" for row in parity_rows)

    key_numbers = [
        key_number("tree-max-error", f"{err[:, 0].max():.4f}", "Largest |CRR(1000) − Black–Scholes| across the six contracts", float(err[:, 0].max()), 4, page="research-log.html", stated="roughly 10⁻³"),
        key_number("pde-max-error", f"{err[:, 1].max():.5f}", "Largest |Crank–Nicolson − Black–Scholes|", float(err[:, 1].max()), 5, page="research-log.html", stated="roughly 10⁻³"),
        key_number("mc-max-deviation", fmt(float((err[:, 2] / err[:, 3]).max()), 2, suffix=" SE"), "Largest Monte Carlo deviation in standard-error units", float((err[:, 2] / err[:, 3]).max()), 2, suffix=" SE", page="research-log.html", stated="statistically consistent (< 3 SE)"),
        key_number("iv-roundtrip-error", f"{iv_err:.1e}", "Implied-volatility round-trip maximum error over 135 inversions", iv_err, 9, page="research-log.html", stated="about 10⁻⁸",
                   unverified=True, note="Computed maximum error is 1.2 × 10⁻⁷ (solver xtol 10⁻⁸); the page's order-of-magnitude statement is not reproduced to printed precision."),
        key_number("parity-max-violation", f"{det_parity:.1e}", "Largest put–call parity violation among deterministic methods", det_parity, 12, page="research-log.html", stated="parity holds"),
        key_number("mc-paths", "200,000", "Antithetic Monte Carlo paths", 200_000, 0, page="research-log.html", stated="200,000"),
        key_number("seed", "42", "PCG64 seed", 42, 0, page="research-log.html", stated="42"),
        key_number("all-budgets-met", "pass" if all_pass else "fail", "Every method inside its declared error budget", page="findings.html", stated="agreed within declared tolerances"),
        key_number("tree-tolerance", "1e-02", "Declared CRR tolerance (tests/test_models_advanced.py)", 1e-2, 2, page="tests"),
        key_number("pde-tolerance", "1e-02", "Declared Crank–Nicolson tolerance", 1e-2, 2, page="tests"),
        key_number("mc-tolerance", "3 SE", "Declared Monte Carlo tolerance", 3, 0, suffix=" SE", page="tests"),
    ]
    figures = [
        figure("errors", "dots", "Absolute pricing error against the closed form · six benchmark contracts", "|price − Black–Scholes| · log scale · notebook 01",
               "Dots of the absolute pricing error of the CRR tree, the Crank–Nicolson solver and Monte Carlo for calls and puts at three strikes; tree and PDE errors sit near one thousandth, Monte Carlo errors inside their standard errors.",
               x={"label": "contract", "values": labels},
               series=[series("|CRR(1000) − BS|", "c1", sigs(err[:, 0])), series("|Crank–Nicolson − BS|", "c2", sigs(err[:, 1])),
                       series("|Monte Carlo − BS|", "c3", sigs(err[:, 2])),
                       series("Monte Carlo std err", "c7", sigs(err[:, 3]), dash=True)], yScale="log"),
        figure("crr-convergence", "line", "CRR binomial convergence to Black–Scholes (ATM call)", "|CRR(n) − BS| · n from 10 to 2000 · notebook 01",
               "Line of the absolute CRR pricing error against the number of tree steps on logarithmic axes, oscillating inside a one-over-n envelope.",
               x={"label": "tree steps n (log scale)", "scale": "log", "values": [int(n) for n in ns]},
               series=[series("|CRR(n) − BS|", "c1", sigs(crr_err)), series("O(1/n) reference", "c7", sigs(crr_err[0] * ns[0] / ns), dash=True)], yScale="log"),
    ]
    tables = [
        table("cross-method", "Four roads to one price · S₀ = 100, r = 5%, σ = 20%, T = 1", ["Contract", "Method", "Price", "|error| vs closed form", "MC std err", "|error| / SE", "Tolerance", "Verdict"], rows),
        table("parity", "Put–call parity by method", ["K", "Method", "C − P", "S − K e^{-rT}", "Violation", "Tolerance", "Verdict"], parity_rows),
        table("iv-roundtrip", "Implied-volatility round trip", ["Strikes", "True vols", "Inversions", "Max |recovered − true|"], [[iv_strikes.tolist(), f"{sig_true[0]:.2f}–{sig_true[-1]:.2f} (27 points)", int(recovered.size), sig(iv_err)]]),
    ]
    payload = assemble(common, "pricing-validation", "asset-pricing.html", prose, meta, "notebooks/01_asset_pricing_models.ipynb", key_numbers, figures, tables,
                       ledger_id="pricing-validation",
                       parameters={"contract": {"S0": S0, "r": r, "q": q, "sigma": sigma, "T": T, "strikes": [90, 100, 110]}, "tree_steps": 1000, "pde_grid": [400, 400], "mc_paths": 200_000, "seed": SEED,
                                   "tolerances": {"tree_abs": 1e-2, "pde_abs": 1e-2, "monte_carlo_se_multiple": 3.0, "parity_abs_analytic": 1e-8, "parity_abs_numerical": 1e-2}})
    card = common.cards.get("research-log.html#pricing-validation", {})
    payload.update({"kicker": card.get("kicker"), "result": card.get("result"), "evidence_class": card.get("evidence_class"), "metric": card.get("metric"), "page": "/lab/research-log#pricing-validation"})
    return payload


# ---------------------------------------------------------------------------
# 7. stock-bond-regimes  (notebook 06 + dossier + atlas)
# ---------------------------------------------------------------------------
def build_stock_bond_regimes(common: Common, risk: dict) -> dict:
    page = "finding-diversification-regimes.html"
    dossier_sections = page_sections("stock-bond-regime-dossier.html")
    prose, meta = finding_prose(page, extra_source_pages=("stock-bond-regime-dossier.html",), extra_sections=dossier_sections)
    wide = to_returns(load_prices(["SPY", "TLT"]))
    roll = wide["SPY"].rolling(126).corr(wide["TLT"]).dropna()
    mean_1519 = float(roll.loc[:"2019"].mean())
    peak_2223 = float(roll.loc["2022":"2023"].max())
    peak_date = roll.loc["2022":"2023"].idxmax()
    latest = float(roll.iloc[-1])
    vols = correlation_stress_vols(risk)
    undiv = undiversified_vol(risk)
    lambdas = np.linspace(0, 1, 21)
    episodes, heatmap = atlas_episodes()
    last_2019 = roll.loc[:"2019"].index[-1]
    first_2022 = roll.loc["2022":].index[0]
    last_2023 = roll.loc[:"2023"].index[-1]
    target = risk["target"]

    key_numbers = [
        key_number("corr-2015-2019", fmt(mean_1519, 2, sign=True), "SPY–TLT 126-day correlation · 2015–2019 average", mean_1519, 2, page=page, stated="−0.35"),
        key_number("corr-2022-2023-peak", fmt(peak_2223, 2, sign=True), "SPY–TLT 126-day correlation · 2022–2023 peak", peak_2223, 2, page=page, stated="+0.37"),
        key_number("corr-sample-end", fmt(latest, 2, sign=True), "SPY–TLT 126-day correlation · sample end", latest, 2, page=page, stated="+0.31"),
        key_number("corr-swing", "≈" + fmt(peak_2223 - mean_1519, 1), "Swing from the 2015–2019 average to the 2022–2023 peak", peak_2223 - mean_1519, 1, page="notebooks/06", stated="≈0.7"),
        key_number("vol-observed", fmt(100 * vols[0], 1, suffix="%"), "Portfolio volatility at the observed correlation structure", 100 * vols[0], 1, suffix="%", page=page, stated="11.6%"),
        key_number("vol-stressed", fmt(100 * vols[-1], 1, suffix="%"), "Portfolio volatility with every correlation pushed to one", 100 * vols[-1], 1, suffix="%", page=page, stated="15.8%"),
        key_number("vol-half-stress", fmt(100 * vols[10], 2, suffix="%"), "Portfolio volatility at λ = 0.5", 100 * vols[10], 2, suffix="%", page="notebooks/06", stated="13.86%"),
        key_number("diversification-points", "≈" + fmt(100 * (vols[-1] - vols[0]), 0), "Percentage points of volatility diversification removes", 100 * (vols[-1] - vols[0]), 0, page=page, stated="about four percentage points"),
        key_number("undiversified-vol", fmt(100 * undiv, 1, suffix="%"), "Weighted-average volatility of the components", 100 * undiv, 1, suffix="%", page=page, stated="15.8%"),
        key_number("rolling-window", "126", "Rolling window (trading days)", 126, 0, page=page, stated="126"),
        key_number("corr-series-start", iso(roll.index[0]), "First rolling correlation", page="notebooks/06"),
        key_number("corr-series-end", iso(roll.index[-1]), "Last rolling correlation", page="notebooks/06", stated="6 July 2026"),
        key_number("peak-date", iso(peak_date), "Date of the 2022–2023 peak", page="notebooks/06"),
    ]
    figures = [
        figure("spy-tlt-correlation", "line", "SPY–TLT correlation: the insurance that lapsed", "rolling 126-day correlation of daily returns · full daily series · notebook 06",
               "Line chart of the rolling one-hundred-and-twenty-six day correlation between SPY and TLT daily returns from 2015 to 2026, negative through 2019 and positive from 2022 with vertical rules at the episode boundaries.",
               x={"label": "date", "values": [iso(d) for d in roll.index]}, series=[series("SPY–TLT 126-day correlation", "c1", sigs(roll.to_numpy()))],
               rules=[{"axis": "y", "at": 0, "label": "zero correlation", "color": "c7"},
                      {"axis": "x", "at": iso(last_2019), "label": "end of 2015–2019 window", "color": "c5"},
                      {"axis": "x", "at": iso(first_2022), "label": "2022–2023 inflation shock begins", "color": "c3"},
                      {"axis": "x", "at": iso(last_2023), "label": "end of 2022–2023 window", "color": "c3"}],
               annotations=[{"x": iso(roll.loc[:"2019"].index[len(roll.loc[:"2019"]) // 2]), "y": sig(mean_1519), "text": "2015–2019 average"},
                            {"x": iso(peak_date), "y": sig(peak_2223), "text": "2022–2023 peak"},
                            {"x": iso(roll.index[-1]), "y": sig(latest), "text": "sample end"}], yMin=-1, yMax=1),
        figure("correlation-stress", "columns", "Diversification decay under correlation stress", "% annualised · λ from 0 (observed) to 1 (all correlations → 1) · notebook 06",
               "Columns of annualised volatility of the 60/40-plus-alternatives portfolio as its correlation matrix is blended toward perfect correlation, rising from 11.6 to 15.8 percent.",
               x={"label": "correlation stress λ", "values": sigs(lambdas)}, series=[series("portfolio volatility", "c1", [sig(100 * v) for v in vols])],
               rules=[{"axis": "y", "at": sig(100 * vols[0]), "label": "observed structure", "color": "c2"}, {"axis": "y", "at": sig(100 * undiv), "label": "weighted-average vol (no diversification)", "color": "c5"}], yMin=0),
        heatmap,
    ]
    tables = [
        table("correlation-levels", "Published rolling-correlation levels", ["Level", "Value", "Window"],
              [["2015–2019 average", sig(mean_1519), f"{iso(roll.index[0])} – {iso(last_2019)}"], ["2022–2023 peak", sig(peak_2223), f"{iso(first_2022)} – {iso(last_2023)} (peak {iso(peak_date)})"], ["Sample end", sig(latest), iso(roll.index[-1])]]),
        table("correlation-stress", "Portfolio volatility along the correlation-stress path", ["λ", "Annualised volatility"], [[sig(l), sig(v)] for l, v in zip(lambdas, vols)],
              note="Volatilities held fixed; correlations blended toward one and the covariance projected back to the PSD cone."),
        table("portfolio", "The 60/40-plus-alternatives portfolio", ["Sleeve", "ETF", "Weight"],
              [["US equity", "SPY", 0.30], ["US equity", "QQQ", 0.10], ["US equity", "IWM", 0.05], ["International equity", "EFA", 0.10], ["Bonds", "AGG", 0.20], ["Bonds", "TLT", 0.10], ["Alternatives", "GLD", 0.075], ["Alternatives", "VNQ", 0.075]]),
        episodes,
        table("regime-map", "Scenario map (dossier): mechanisms to test, not forecasts", ["Dominant shock", "Policy constraint", "Possible stock channel", "Possible nominal-bond channel"],
              [["Disinflationary growth shock", "Room to ease", "Lower expected earnings", "Lower yields can support prices"],
               ["Inflationary supply or policy shock", "Inflation limits easing", "Higher discount rates and weaker margins", "Higher expected rates reduce prices"],
               ["Fiscal or term-premium shock", "More duration must be absorbed", "Tighter financial conditions", "Higher risk compensation reduces prices"]]),
    ]
    assert all(float(w) == float(target[t]) for _, t, w in tables[2]["rows"])
    payload = assemble(common, "stock-bond-regimes", page, prose, meta, "notebooks/06_market_risk_dashboard.ipynb", key_numbers, figures, tables,
                       ledger_id="market-risk",
                       parameters={"pair": ["SPY", "TLT"], "rolling_window": 126, "stress_lambdas": 21, "covariance_window": [iso(risk["returns"].index[0]), iso(risk["returns"].index[-1])]},
                       related_pages=["/lab/stock-bond-regime-dossier", "/lab/regime-atlas"])
    return payload


# ---------------------------------------------------------------------------
# ledger.json and manifest.json
# ---------------------------------------------------------------------------
LEDGER_MAP = [
    ("pricing-validation", "pricing-validation", ["pricing-validation"]),
    ("portfolio-construction", "portfolio-estimation", ["portfolio-estimation"]),
    ("stochastic-volatility", "volatility", ["volatility"]),
    ("monte-carlo", "monte-carlo", ["monte-carlo"]),
    ("factor-exposures", "factor-etfs", ["factor-etfs"]),
    ("market-risk", "var-backtest", ["var-backtest", "stock-bond-regimes"]),
]
NOTEBOOKS = {
    "pricing-validation": "notebooks/01_asset_pricing_models.ipynb",
    "portfolio-estimation": "notebooks/02_portfolio_optimization.ipynb",
    "volatility": "notebooks/03_derivatives_pricing.ipynb",
    "monte-carlo": "notebooks/04_monte_carlo_simulation.ipynb",
    "factor-etfs": "notebooks/05_factor_investing.ipynb",
    "var-backtest": "notebooks/06_market_risk_dashboard.ipynb",
}


def build_ledger(common: Common, revisions: dict[str, list]) -> dict:
    entries = []
    for ledger_id, slug, artifacts in LEDGER_MAP:
        e = common.ledger[ledger_id]
        f = e["fields"]
        dataset_key = next(k for k in f if k.startswith("Dataset"))
        entries.append({
            "investigation": slug,
            "artifacts": [f"{a}.json" for a in artifacts],
            "record": f"/lab/research-log#{ledger_id}",
            "number": e["number"],
            "title": e["title"],
            "status": e["status"],
            "question": f["Research question"],
            "claim": f["Testable claim"],
            "assumptions": split_list(f["Assumptions"]),
            "dataset": f[dataset_key],
            "method": f["Method"],
            "falsifier": f["Falsification criterion"],
            "result": f["Result and uncertainty"],
            "failures": split_list(f["Failure modes"]),
            "revisions": revisions.get(slug, []),
            "commit": e["commit"],
            "notebook_url": f"{REPO}/blob/{common.source_commit}/{NOTEBOOKS[slug]}",
            "notebook_url_evidence_commit": f"{REPO}/blob/{e['commit']}/{NOTEBOOKS[slug]}",
            "updated": e["updated"],
        })
    return {"schema_version": SCHEMA, "generated_at": common.generated_at, "source_commit": common.source_commit, "entries": entries}


def write_json(name: str, payload: dict) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    (OUT / name).write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def main() -> None:
    common = Common()
    risk = bfrd.compute_risk_study()
    fs = bfrd.compute_factor_study()
    builders = [
        ("portfolio-estimation", lambda: build_portfolio_estimation(common)),
        ("var-backtest", lambda: build_var_backtest(common, risk)),
        ("factor-etfs", lambda: build_factor_etfs(common, fs)),
        ("volatility", lambda: build_volatility(common)),
        ("monte-carlo", lambda: build_monte_carlo(common)),
        ("pricing-validation", lambda: build_pricing_validation(common)),
        ("stock-bond-regimes", lambda: build_stock_bond_regimes(common, risk)),
    ]
    revisions: dict[str, list] = {}
    sizes = {}
    for slug, build in builders:
        payload = build()
        check_series_lengths(payload)
        revisions[slug] = finding_prose_revisions(payload)
        sizes[f"{slug}.json"] = write_json(f"{slug}.json", payload)
    ledger = build_ledger(common, revisions)
    sizes["ledger.json"] = write_json("ledger.json", ledger)
    manifest = {
        "schema_version": SCHEMA,
        "generated_at": common.generated_at,
        "source_commit": common.source_commit,
        "evidence_commit": EVIDENCE_COMMIT,
        "artifacts": [f"{slug}.json" for slug, _ in builders],
        "ledger": "ledger.json",
        "tests": common.tests,
        "data": common.data,
    }
    sizes["manifest.json"] = write_json("manifest.json", manifest)
    for name, size in sizes.items():
        print(f"wrote {OUT.relative_to(ROOT) / name}  {size / 1024:.1f} KB")


def finding_prose_revisions(payload: dict) -> list:
    return payload.get("revisions") or []


if __name__ == "__main__":
    main()
