#!/usr/bin/env python3
"""Generate app/frontend/sitemap.xml from the pages that actually exist.

Run in CI before the Pages artifact is uploaded, so adding a page to the site
adds it to the sitemap with no second step to remember.

`lastmod` comes from each file's last commit date rather than the build clock:
stamping every URL with "now" on every deploy trains crawlers to ignore the
signal entirely. Falls back to file mtime when git history is unavailable
(shallow clone, tarball export).
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

SITE = "https://lab.kylewisniewski.com"
FRONTEND = Path(__file__).resolve().parent.parent / "app" / "frontend"

# Error pages are served by status code, never crawled as destinations.
EXCLUDE = {"404.html"}


def last_modified(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            capture_output=True, text=True, check=True, cwd=path.parent,
        ).stdout.strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def canonical(path: Path) -> str:
    # Matches the <link rel="canonical"> in each page: index is the bare root,
    # everything else keeps the .html the internal navigation links to.
    return f"{SITE}/" if path.name == "index.html" else f"{SITE}/{path.name}"


def main() -> None:
    pages = sorted(p for p in FRONTEND.glob("*.html") if p.name not in EXCLUDE)
    if not pages:
        raise SystemExit(f"no pages found in {FRONTEND}")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for page in pages:
        lines += [
            "  <url>",
            f"    <loc>{escape(canonical(page))}</loc>",
            f"    <lastmod>{last_modified(page)}</lastmod>",
            "  </url>",
        ]
    lines.append("</urlset>")

    out = FRONTEND / "sitemap.xml"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} with {len(pages)} urls")


if __name__ == "__main__":
    main()
