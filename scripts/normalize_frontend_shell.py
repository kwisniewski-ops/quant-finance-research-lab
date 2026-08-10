"""Keep the static lab's global identity, wayfinding, and social asset consistent."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"
PUBLIC_HTML_NAME = re.compile(r"^[a-z0-9-]+\.html$")
BRAND_NAME = "Quantitative Markets & Institutions Lab"
BRAND_NAME_HTML = html.escape(BRAND_NAME)
SOCIAL_IMAGE_URL = "https://www.kylewisniewski.com/lab/og-lab-v4.png"
SOCIAL_IMAGE_ALT = (
    "Quantitative Markets & Institutions Lab: independent research on models, regimes, "
    "and institutions, shown beside a market chart and open research notebook."
)


def section_for(page: Path) -> str | None:
    name = page.name
    if name in {"index.html", "about.html"}:
        return "Overview"
    if name == "findings.html" or name.startswith("finding-") or name in {"case-study.html", "stock-bond-regime-dossier.html", "regime-atlas.html"}:
        return "Research"
    if name in {"tools.html", "asset-pricing.html", "portfolio.html", "risk.html", "simulations.html", "factors.html"}:
        return "Models"
    if name in {"method.html", "math-notes.html"}:
        return "Methods"
    if name == "research-log.html":
        return "Record"
    return None


def header(active: str | None) -> str:
    links = [
        ("Overview", "index.html"),
        ("Research", "findings.html"),
        ("Models", "tools.html"),
        ("Methods", "method.html"),
        ("Record", "research-log.html"),
    ]
    navigation_parts = []
    for label, href in links:
        current = ' aria-current="page"' if label == active else ""
        navigation_parts.append(f'<a href="{href}"{current}>{label}</a>')
    navigation = "".join(navigation_parts)
    return (
        '<header class="site-header">\n'
        '  <div class="wrap">\n'
        '    <div class="identity-lockup" aria-label="Site identity">\n'
        '      <a class="wordmark" href="index.html">Quantitative Markets &amp; Institutions <span class="tail">Lab</span></a>\n'
        '    </div>\n'
        '    <nav class="site-nav" aria-label="Primary navigation">'
        f'{navigation}'
        '</nav>\n'
        '  </div>\n'
        '</header>'
    )


FOOTER = (
    '<footer class="site-footer"><div class="wrap">'
    f'<span>{BRAND_NAME_HTML} · 2026</span>'
    '<span><a href="findings.html">Research</a> · <a href="regime-atlas.html">Regime atlas</a> · '
    '<a href="stock-bond-regime-dossier.html">Stock–bond dossier</a> · <a href="tools.html">Models</a> · '
    '<a href="method.html">Methods</a> · <a href="research-log.html">Research record</a> · '
    '<a href="about.html">Mission</a> · <a href="feed.xml">Feed</a> · '
    '<a href="https://github.com/kwisniewski-ops/quant-finance-research-lab">Source code</a> · '
    '<a href="https://www.kylewisniewski.com">Main site</a> · Educational research; not investment advice</span>'
    '</div></footer>'
)


def normalize(page: Path) -> None:
    source = page.read_text(encoding="utf-8")
    source = source.replace("Quantitative Markets Research Lab", BRAND_NAME_HTML)
    source = re.sub(
        r'(<script[^>]+type="application/ld\+json"[^>]*>.*?</script>)',
        lambda match: match.group(1).replace(BRAND_NAME_HTML, BRAND_NAME),
        source,
        flags=re.DOTALL,
    )
    source = source.replace(
        f'title="{BRAND_NAME_HTML} findings"',
        f'title="{BRAND_NAME_HTML} research"',
    )
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
        r'"url"\s*:\s*"https://(?:www\.)?kylewisniewski\.com/?"\s*\}',
        '"author": {"@id": "https://www.kylewisniewski.com/lab#organization"}',
        source,
        flags=re.DOTALL,
    )
    source = re.sub(
        r'"author"\s*:\s*\{\s*"@type"\s*:\s*"Organization"\s*,\s*'
        rf'"name"\s*:\s*"{re.escape(BRAND_NAME)}"\s*,\s*'
        r'"url"\s*:\s*"https://(?:www\.)?kylewisniewski\.com/lab/?"\s*\}',
        '"author": {"@id": "https://www.kylewisniewski.com/lab#organization"}',
        source,
        flags=re.DOTALL,
    )

    source = re.sub(
        r'https://(?:www\.)?kylewisniewski\.com/lab/(?:images/og-[^"<]+\.(?:jpg|png)|og-lab(?:-v[234])?\.(?:jpg|png))',
        SOCIAL_IMAGE_URL,
        source,
    )
    source = source.replace("warm paper", "ivory research paper")
    source = source.replace('<meta property="og:image:type" content="image/jpeg">', '<meta property="og:image:type" content="image/png">')
    if '<meta property="og:image"' in source and '<meta property="og:image:type"' not in source:
        source = source.replace(
            f'<meta property="og:image" content="{SOCIAL_IMAGE_URL}">',
            f'<meta property="og:image" content="{SOCIAL_IMAGE_URL}">\n<meta property="og:image:type" content="image/png">',
            1,
        )
    if '<meta property="og:image"' in source:
        og_alt = f'<meta property="og:image:alt" content="{html.escape(SOCIAL_IMAGE_ALT)}">'
        if '<meta property="og:image:alt"' in source:
            source = re.sub(r'<meta property="og:image:alt" content="[^"]*">', og_alt, source, count=1)
        else:
            source = source.replace('<meta property="og:image:height" content="630">', '<meta property="og:image:height" content="630">\n' + og_alt, 1)
    if '<meta name="twitter:image"' in source:
        twitter_alt = f'<meta name="twitter:image:alt" content="{html.escape(SOCIAL_IMAGE_ALT)}">'
        if '<meta name="twitter:image:alt"' in source:
            source = re.sub(r'<meta name="twitter:image:alt" content="[^"]*">', twitter_alt, source, count=1)
        else:
            twitter_image = re.search(r'<meta name="twitter:image" content="[^"]*">', source)
            if twitter_image:
                source = source[:twitter_image.end()] + "\n" + twitter_alt + source[twitter_image.end():]
    if 'type="application/atom+xml"' not in source:
        source = source.replace(
            "</head>",
            f'<link rel="alternate" type="application/atom+xml" title="{BRAND_NAME_HTML} research" href="feed.xml">\n</head>',
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
            "Research": "https://www.kylewisniewski.com/lab/findings.html",
            "Models": "https://www.kylewisniewski.com/lab/tools.html",
            "Methods": "https://www.kylewisniewski.com/lab/method.html",
            "Record": "https://www.kylewisniewski.com/lab/research-log.html",
        }
        crumbs = [(BRAND_NAME, "https://www.kylewisniewski.com/lab")]
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
    source = re.sub(
        r'\s*<script src="(?:/lab/)?js/engagement\.js"></script>',
        "",
        source,
    )
    source = source.replace("</body>", '\n<script src="js/engagement.js"></script>\n</body>', 1)
    if page.name == "404.html":
        source = re.sub(r'href="(?!https?://|mailto:|/|#)([^"]+)"', r'href="/lab/\1"', source)
        source = re.sub(r'src="(?!https?://|/)([^"]+)"', r'src="/lab/\1"', source)
    page.write_text(source, encoding="utf-8")


def main() -> None:
    pages = sorted(page for page in FRONTEND.glob("*.html") if PUBLIC_HTML_NAME.fullmatch(page.name))
    for page in pages:
        normalize(page)
    print(f"Normalized global shell and social asset across {len(pages)} pages")


if __name__ == "__main__":
    main()
