from __future__ import annotations

import argparse
import sys

from .graph import APP
from .regulatory import index_regulatory_corpus, load_rag_config_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='intake')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Render an offline or live intake draft')
    run_parser.add_argument('--provider', help='Hosted provider name', default=None)
    run_parser.add_argument('--model', help='Exact model ID', required=True)
    run_parser.add_argument('--live', action='store_true', help='Use Tavily, Jina Reader, and Hugging Face Hub')

    index_parser = subparsers.add_parser('index-regulatory', help='Index the regulatory corpus into Pinecone')
    index_parser.set_defaults(command='index-regulatory')
    return parser


def run_command(args: argparse.Namespace) -> int:
    result = APP.invoke({'live_mode': bool(args.live), 'provider': args.provider, 'model': args.model})

    report_markdown = result.get('report_markdown')
    error = result.get('error')
    if report_markdown:
        print(report_markdown)
    if error:
        print(error, file=sys.stderr)
        return 2
    if not report_markdown:
        print('Report generation failed.', file=sys.stderr)
        return 1

    return 0


def index_regulatory_command() -> int:
    report = index_regulatory_corpus(load_rag_config_from_env())

    print('# Regulatory index status')
    print(f"- Corpus version: {report.get('corpus_version')}")
    print(f"- Configured index: {report.get('index_name') or 'n/a'}")
    print(f"- Namespace: {report.get('namespace') or 'n/a'}")
    print(f"- Compatible: {'yes' if report.get('compatible') else 'no'}")
    print(f"- Indexed chunks: {report.get('indexed')}")
    print(f"- Skipped blocked chunks: {report.get('skipped_blocked')}")

    needed = report.get('needed') or []
    if needed:
        print('- Needed configuration:')
        for item in needed:
            print(f'  - {item}')

    blockers = report.get('blockers') or []
    if blockers:
        print('- Blockers:')
        for item in blockers:
            print(f'  - {item}')

    if needed or blockers or not report.get('compatible') or not report.get('indexed'):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'run':
        return run_command(args)
    if args.command == 'index-regulatory':
        return index_regulatory_command()

    parser.error('Unknown command.')
    return 2
