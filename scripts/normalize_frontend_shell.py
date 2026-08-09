"""Keep the static lab's global identity, wayfinding, and social asset consistent."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"


def section_for(page: Path) -> str | None:
    name = page.name
    if name == "index.html":
        return "Overview"
    if name == "findings.html" or name.startswith("finding-") or name == "case-study.html":
        return "Findings"
    if name in {"tools.html", "asset-pricing.html", "portfolio.html", "risk.html", "simulations.html", "factors.html"}:
        return "Tools"
    if name in {"method.html", "math-notes.html", "about.html"}:
        return "Method"
    if name == "research-log.html":
        return "Log"
    return None


def header(active: str | None) -> str:
    links = [
        ("Overview", "index.html"),
        ("Findings", "findings.html"),
        ("Tools", "tools.html"),
        ("Method", "method.html"),
        ("Log", "research-log.html"),
    ]
    navigation = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if label == active else ""}>{label}</a>'
        for label, href in links
    )
    return (
        '<header class="site-header">\n'
        '  <div class="wrap">\n'
        '    <div class="identity-lockup" aria-label="Site identity">\n'
        '      <a class="identity-parent" href="https://kylewisniewski.com">Kyle Wisniewski</a>\n'
        '      <span class="identity-separator" aria-hidden="true">/</span>\n'
        '      <a class="wordmark" href="index.html">Quantitative Markets <span class="tail">Lab</span></a>\n'
        '    </div>\n'
        '    <nav class="site-nav" aria-label="Primary navigation">'
        f'{navigation}<a class="nav-professional" href="https://kylewisniewski.com/projects/quantitative-markets-research-lab">How I work</a>'
        '</nav>\n'
        '  </div>\n'
        '</header>'
    )


FOOTER = (
    '<footer class="site-footer"><div class="wrap">'
    '<span><a href="https://kylewisniewski.com">Kyle Wisniewski</a> · Quantitative Markets Research Lab · MMXXVI</span>'
    '<span><a href="findings.html">Findings</a> · <a href="tools.html">Tools</a> · '
    '<a href="method.html">Method</a> · <a href="about.html">About</a> · '
    '<a href="feed.xml">Feed</a> · <a href="https://github.com/kwisniewski-ops/quant-finance-research-lab">GitHub</a> · '
    '<a href="https://kylewisniewski.com/contact">Contact</a> · Not investment advice</span>'
    '</div></footer>'
)


def normalize(page: Path) -> None:
    source = page.read_text(encoding="utf-8")
    source, header_count = re.subn(
        r'<header class="site-header">.*?</header>',
        header(section_for(page)),
        source,
        count=1,
        flags=re.DOTALL,
    )
    source, footer_count = re.subn(
        r'<footer class="site-footer">.*?</footer>',
        FOOTER,
        source,
        count=1,
        flags=re.DOTALL,
    )
    if header_count != 1 or footer_count != 1:
        raise RuntimeError(f"Expected one global header and footer in {page}")

    source = re.sub(
        r'https://kylewisniewski\.com/lab/(?:images/og-[^"<]+\.jpg|og-lab\.jpg)',
        "https://kylewisniewski.com/lab/og-lab-v2.png",
        source,
    )
    source = source.replace('<meta property="og:image:type" content="image/jpeg">', '<meta property="og:image:type" content="image/png">')
    if '<meta property="og:image"' in source and '<meta property="og:image:type"' not in source:
        source = source.replace(
            '<meta property="og:image" content="https://kylewisniewski.com/lab/og-lab-v2.png">',
            '<meta property="og:image" content="https://kylewisniewski.com/lab/og-lab-v2.png">\n<meta property="og:image:type" content="image/png">',
            1,
        )
    if 'type="application/atom+xml"' not in source:
        source = source.replace(
            "</head>",
            '<link rel="alternate" type="application/atom+xml" title="Quantitative Markets Research Lab findings" href="feed.xml">\n</head>',
            1,
        )
    if page.name not in {"index.html", "404.html"} and 'id="breadcrumb-schema"' not in source:
        canonical_match = re.search(r'<link rel="canonical" href="([^"]+)">', source)
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", source, re.DOTALL)
        if not canonical_match or not title_match:
            raise RuntimeError(f"Cannot build breadcrumb metadata for {page}")
        page_title = html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1))).strip()
        active = section_for(page)
        group_urls = {
            "Findings": "https://kylewisniewski.com/lab/findings.html",
            "Tools": "https://kylewisniewski.com/lab/tools.html",
            "Method": "https://kylewisniewski.com/lab/method.html",
            "Log": "https://kylewisniewski.com/lab/research-log.html",
        }
        crumbs = [
            ("Kyle Wisniewski", "https://kylewisniewski.com"),
            ("Research Lab", "https://kylewisniewski.com/lab"),
        ]
        group_url = group_urls.get(active or "")
        canonical = canonical_match.group(1)
        if active and group_url and canonical != group_url:
            crumbs.append((active, group_url))
        crumbs.append((page_title, canonical))
        schema = {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": index,
                    "name": name,
                    "item": url,
                }
                for index, (name, url) in enumerate(crumbs, start=1)
            ],
        }
        source = source.replace(
            "</head>",
            '<script id="breadcrumb-schema" type="application/ld+json">\n'
            + json.dumps(schema, ensure_ascii=False, indent=2)
            + "\n</script>\n</head>",
            1,
        )
    if 'src="js/engagement.js"' not in source:
        source = source.replace("</body>", '<script src="js/engagement.js"></script>\n</body>', 1)
    page.write_text(source, encoding="utf-8")


def main() -> None:
    pages = sorted(FRONTEND.glob("*.html"))
    for page in pages:
        normalize(page)
    print(f"Normalized global shell and social asset across {len(pages)} pages")


if __name__ == "__main__":
    main()
