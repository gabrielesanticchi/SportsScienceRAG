import io
import json

from sportsscience_rag.logging_setup import JsonlLogger


def test_event_writes_one_json_line():
    buf = io.StringIO()
    logger = JsonlLogger(stream=buf)
    written = logger.event(
        source="s3://b/k.pdf", content_hash="h", stage="upsert",
        duration_ms=12, n_chunks=5, status="done", error_class=None,
    )
    line = buf.getvalue().strip()
    parsed = json.loads(line)
    assert parsed["status"] == "done"
    assert parsed["n_chunks"] == 5
    assert written == parsed
