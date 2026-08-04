"""Command-line interface for MPSI."""

from __future__ import annotations

import argparse

from mpsi.config import load_config
from mpsi.graph import TouchdownGraph
from mpsi.pipeline import MPSIPipeline
from mpsi.scoring import compute_scores, write_scores


def build_command(args):
    config = load_config(args.config)
    pipeline = MPSIPipeline(config, args.level, args.output_dir)
    generated, accepted, failed = pipeline.run(
        limit=args.limit,
        skip_quality_filter=args.skip_quality_filter,
    )
    print(
        f"Generated {generated} panoramas; accepted {accepted}; failed {failed}."
    )


def score_command(args):
    config = load_config(args.config)
    graph = TouchdownGraph(config.nodes_file, config.links_file)
    scores = compute_scores(graph.nx, config.route_data)
    write_scores(scores, args.output or config.score_file)
    print(f"Wrote {len(scores)} node scores.")


def main():
    parser = argparse.ArgumentParser(prog="mpsi")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Generate a Rule-VLN level.")
    build.add_argument("--config", required=True)
    build.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--skip-quality-filter", action="store_true")
    build.set_defaults(func=build_command)
    score = subparsers.add_parser("score", help="Compute node importance scores.")
    score.add_argument("--config", required=True)
    score.add_argument("--output", default=None)
    score.set_defaults(func=score_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
