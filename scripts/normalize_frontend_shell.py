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
        return "Start"
    if name == "findings.html" or name.startswith("finding-") or name == "case-study.html":
        return "Questions"
    if name in {"tools.html", "asset-pricing.html", "portfolio.html", "risk.html", "simulations.html", "factors.html"}:
        return "Models"
    if name in {"method.html", "math-notes.html", "about.html"}:
        return "Method"
    if name == "research-log.html":
        return "Research record"
    return None


def header(active: str | None) -> str:
    links = [
        ("Start", "index.html"),
        ("Questions", "findings.html"),
        ("Models", "tools.html"),
        ("Method", "method.html"),
    ]
    navigation = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if label == active else ""}>{label}</a>'
        for label, href in links
    )
    return (
        '<header class="site-header">\n'
        '  <div class="wrap">\n'
        '    <div class="identity-lockup" aria-label="Site identity">\n'
        '      <a class="wordmark" href="index.html">Quantitative Markets <span class="tail">Research Lab</span></a>\n'
        '    </div>\n'
        '    <nav class="site-nav" aria-label="Primary navigation">'
        f'{navigation}'
        '</nav>\n'
        '  </div>\n'
        '</header>'
    )


FOOTER = (
    '<footer class="site-footer"><div class="wrap">'
    '<span>Quantitative Markets Research Lab · MMXXVI</span>'
    '<span><a href="findings.html">Questions</a> · <a href="tools.html">Models</a> · '
    '<a href="method.html">Method</a> · <a href="research-log.html">Research record</a> · '
    '<a href="about.html">About the lab</a> · <a href="feed.xml">Feed</a> · '
    '<a href="https://github.com/kwisniewski-ops/quant-finance-research-lab">Source code</a> · '
    '<a href="https://kylewisniewski.com">Main site</a> · Not investment advice</span>'
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
        r'"author"\s*:\s*\{\s*"@type"\s*:\s*"Person"\s*,\s*'
        r'"name"\s*:\s*"Kyle Wisniewski"\s*,\s*'
        r'"url"\s*:\s*"https://kylewisniewski\.com/?"\s*\}',
        '"author": {"@id": "https://kylewisniewski.com/lab#organization"}',
        source,
        flags=re.DOTALL,
    )
    source = re.sub(
        r'"author"\s*:\s*\{\s*"@type"\s*:\s*"Organization"\s*,\s*'
        r'"name"\s*:\s*"Quantitative Markets Research Lab"\s*,\s*'
        r'"url"\s*:\s*"https://kylewisniewski\.com/lab/?"\s*\}',
        '"author": {"@id": "https://kylewisniewski.com/lab#organization"}',
        source,
        flags=re.DOTALL,
    )

    source = re.sub(
        r'https://kylewisniewski\.com/lab/(?:images/og-[^"<]+\.jpg|og-lab\.jpg)',
        "https://kylewisniewski.com/lab/og-lab-v2.png",
        source,
    )
    source = source.replace("warm paper", "ivory research paper")
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
    if page.name not in {"index.html", "404.html"}:
        canonical_match = re.search(r'<link rel="canonical" href="([^"]+)">', source)
        title_match = re.search(r"<h1[^>]*>(.*?)</h1>", source, re.DOTALL)
        if not canonical_match or not title_match:
            raise RuntimeError(f"Cannot build breadcrumb metadata for {page}")
        page_title = html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1))).strip()
        active = section_for(page)
        group_urls = {
            "Questions": "https://kylewisniewski.com/lab/findings.html",
            "Models": "https://kylewisniewski.com/lab/tools.html",
            "Method": "https://kylewisniewski.com/lab/method.html",
            "Research record": "https://kylewisniewski.com/lab/research-log.html",
        }
        crumbs = [("Research Lab", "https://kylewisniewski.com/lab")]
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
        breadcrumb_block = (
            '<script id="breadcrumb-schema" type="application/ld+json">\n'
            + json.dumps(schema, ensure_ascii=False, indent=2)
            + "\n</script>"
        )
        if 'id="breadcrumb-schema"' in source:
            source = re.sub(
                r'<script id="breadcrumb-schema" type="application/ld\+json">.*?</script>',
                breadcrumb_block,
                source,
                count=1,
                flags=re.DOTALL,
            )
        else:
            source = source.replace("</head>", breadcrumb_block + "\n</head>", 1)
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
