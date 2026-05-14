"""Generate a single-file PDF coverage report.

Run after pytest:

    cd backend
    pytest                                # produces .coverage data
    python scripts/coverage_pdf.py        # writes ../coverage/coverage.pdf

The script captures `coverage report -m` as text and renders it into a
formatted PDF. We deliberately keep this to one self-contained PDF
artifact rather than the dozens of per-file HTML pages coverage.py
emits — the PDF is what reviewers open, the HTML is what local devs
generate ad-hoc.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


def _run_coverage_report() -> str:
    """Capture the output of `coverage report -m`."""
    result = subprocess.run(
        ["coverage", "report", "-m"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 2):
        # 2 = under threshold, still emits useful output. Anything else is a real failure.
        raise SystemExit(
            f"coverage report failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def _build_pdf(report_text: str, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Wikipedia RAG Chat — Backend Coverage",
        author="Article Companion",
    )

    styles = getSampleStyleSheet()
    code_style = ParagraphStyle(
        "Mono",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
    )
    body_style = styles["BodyText"]

    story = []
    story.append(Paragraph("<b>Backend Coverage Report</b>", styles["Title"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            body_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            "Per-module line coverage from <code>coverage report -m</code>. "
            "Excluded modules (entrypoint, Pydantic schemas, DI wiring) are "
            "declared in <code>backend/pyproject.toml</code> and map 1:1 to "
            "the brief's three permitted exclusion categories.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(Preformatted(report_text.rstrip(), code_style))

    doc.build(story)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    pdf_path = repo_root / "coverage" / "coverage.pdf"

    text = _run_coverage_report()
    _build_pdf(text, pdf_path)
    print(f"Wrote {pdf_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
