"""Argument-parsing tests for the CLI surface.

These tests exercise only ``build_arg_parser()``; they must never call
``main()`` since that would require real S3/Qdrant/model network access.
"""

from sportsscience_rag.cli import build_arg_parser


def test_defaults():
    args = build_arg_parser().parse_args([])
    assert args.prefixes is None
    assert args.dry_run is False
    assert args.limit is None
    assert args.derived_prefix == "derived/"


def test_repeatable_prefix_and_flags():
    args = build_arg_parser().parse_args(
        ["--prefix", "a/", "--prefix", "b/", "--collection", "c",
         "--limit", "2", "--dry-run", "-v", "--quarantine-report", "q.json"]
    )
    assert args.prefixes == ["a/", "b/"]
    assert args.collection == "c"
    assert args.limit == 2
    assert args.dry_run is True
    assert args.verbose is True
    assert args.quarantine_report == "q.json"


def test_no_multimodal_flag():
    parser = build_arg_parser()
    # --multimodal must not exist anymore
    import pytest
    with pytest.raises(SystemExit):
        parser.parse_args(["--multimodal"])
