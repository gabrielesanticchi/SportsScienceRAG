#!/usr/bin/env python3
"""Compare Unstructured vs Docling on a local PDF."""

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.services.parsers import _run_docling, _run_unstructured, parse_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PDF parsers")
    parser.add_argument("pdf_path", type=Path)
    args = parser.parse_args()

    pdf_bytes = args.pdf_path.read_bytes()
    settings = get_settings()

    unstructured_doc, unstructured_metrics = _run_unstructured(pdf_bytes, settings)
    docling_doc, docling_metrics = _run_docling(pdf_bytes)

    compare_doc = parse_pdf(pdf_bytes, settings=settings.model_copy(update={"pdf_parser": "compare"}))

    print(json.dumps(
        {
            "selected": compare_doc.parser_used,
            "comparison": compare_doc.comparison,
            "unstructured_ok": unstructured_doc is not None,
            "docling_ok": docling_doc is not None,
            "unstructured_metrics": unstructured_metrics.__dict__ if unstructured_metrics else None,
            "docling_metrics": docling_metrics.__dict__ if docling_metrics else None,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
