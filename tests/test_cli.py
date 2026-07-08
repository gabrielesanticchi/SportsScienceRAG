"""Argument-parsing tests for the subcommand CLI surface.

These tests exercise only ``build_arg_parser()``; they never call ``main()``,
which would require real S3/Qdrant/model network access.
"""

import pytest

from sportsscience_rag.cli import build_arg_parser


def test_ingest_defaults():
    args = build_arg_parser().parse_args(["ingest"])
    assert args.command == "ingest"
    assert args.prefixes is None
    assert args.dry_run is False
    assert args.limit is None
    assert args.derived_prefix == "derived/"


def test_ingest_repeatable_prefix_and_flags():
    args = build_arg_parser().parse_args(
        ["ingest", "--prefix", "a/", "--prefix", "b/", "--collection", "c",
         "--limit", "2", "--dry-run", "-v", "--quarantine-report", "q.json"]
    )
    assert args.command == "ingest"
    assert args.prefixes == ["a/", "b/"]
    assert args.collection == "c"
    assert args.limit == 2
    assert args.dry_run is True
    assert args.verbose is True
    assert args.quarantine_report == "q.json"


def test_search_one_shot_args():
    args = build_arg_parser().parse_args(["search", "training load", "--limit", "5", "--json"])
    assert args.command == "search"
    assert args.query == "training load"
    assert args.limit == 5
    assert args.json is True


def test_search_repl_mode_has_no_query():
    args = build_arg_parser().parse_args(["search"])
    assert args.command == "search"
    assert args.query is None
    assert args.limit == 10


def test_eval_defaults():
    args = build_arg_parser().parse_args(["eval"])
    assert args.command == "eval"
    assert args.gold == "eval/gold.jsonl"
    assert args.limit == 10
    assert args.report is None


def test_no_subcommand_is_error():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([])
