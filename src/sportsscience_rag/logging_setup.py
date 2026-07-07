"""Structured JSONL logging for per-document ingestion events."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO


class JsonlLogger:
    """Writes one JSON object per line to a file or stream.

    The logger is used by the ingestion pipeline to emit structured,
    machine-parseable event records (one JSON object per line) describing
    the outcome of per-document processing stages (e.g. parsing, chunking,
    embedding, upserting).

    Attributes:
        _stream: The open text stream events are written to.
    """

    def __init__(self, path: Path | None = None, stream: TextIO | None = None) -> None:
        """Initializes the logger with a destination stream or file path.

        Args:
            path: Filesystem path to append JSONL events to. Ignored if
                `stream` is provided. If neither `path` nor `stream` is
                given, events are written to `sys.stdout`.
            stream: A writable text stream to write JSONL events to. Takes
                precedence over `path` when both are provided.
        """
        if stream is not None:
            self._stream = stream
        elif path is not None:
            self._stream = open(path, "a", encoding="utf-8")  # noqa: SIM115
        else:
            self._stream = sys.stdout

    def event(self, **fields: Any) -> dict[str, Any]:
        """Writes a single structured event as one JSON line.

        Args:
            **fields: Arbitrary keyword arguments describing the event
                (e.g. source, content_hash, stage, duration_ms, n_chunks,
                status, error_class). Values are serialized with
                `json.dumps(..., default=str)` so non-JSON-native types
                (e.g. Path, datetime) are coerced to strings.

        Returns:
            The dict of fields that was written, unmodified.
        """
        self._stream.write(json.dumps(fields, default=str) + "\n")
        self._stream.flush()
        return fields
