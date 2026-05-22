"""Render the final Markdown report to HTML and optionally PDF.

The PDF step uses a local Chrome/Edge installation when available.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import markdown


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
MD_PATH = REPORT_DIR / "FINAL_REPORT.md"
HTML_PATH = REPORT_DIR / "FINAL_REPORT.html"
PDF_PATH = REPORT_DIR / "FINAL_REPORT.pdf"

CSS = """
:root { color-scheme: light; }
body { font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #111827; max-width: 980px; margin: 40px auto; padding: 0 28px; }
h1 { font-size: 32px; margin-bottom: 10px; }
h2 { font-size: 23px; margin-top: 36px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
h3 { font-size: 18px; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border: 1px solid #d1d5db; padding: 8px 10px; vertical-align: top; }
th { background: #f3f4f6; text-align: left; }
img { max-width: 100%; display: block; margin: 18px auto 26px; border: 1px solid #e5e7eb; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
pre { background: #111827; color: #f9fafb; padding: 14px 16px; overflow-x: auto; border-radius: 6px; }
a { color: #1d4ed8; }
@media print { body { margin: 18mm auto; } h2 { break-after: avoid; } img, table, pre { break-inside: avoid; } }
"""

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def render_html() -> Path:
    body = markdown.markdown(
        MD_PATH.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    html = f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Polymarket Reddit Sentiment - Projektbericht</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")
    return HTML_PATH


def render_pdf() -> Path | None:
    chrome = next((path for path in CHROME_CANDIDATES if path.exists()), None)
    if chrome is None:
        return None
    subprocess.run(
        [
            str(chrome),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH.resolve().as_uri(),
        ],
        check=True,
    )
    return PDF_PATH


def main() -> None:
    html = render_html()
    print(f"HTML: {html.relative_to(ROOT).as_posix()}")
    pdf = render_pdf()
    if pdf is not None:
        print(f"PDF:  {pdf.relative_to(ROOT).as_posix()}")
    else:
        print("PDF:  skipped (Chrome/Edge not found)")


if __name__ == "__main__":
    main()
