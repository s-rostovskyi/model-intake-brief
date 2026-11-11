from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .graph import APP
from .regulatory import index_regulatory_corpus, load_rag_config_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='intake')
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Render an offline or live intake draft')
    run_parser.add_argument('--provider', help='Hosted provider name', default=None)
    run_parser.add_argument('--model', help='Exact model ID', required=True)
    run_parser.add_argument('--live', action='store_true', help='Use Tavily, Jina Reader, and Hugging Face Hub')

    stability_parser = subparsers.add_parser(
        'stability', help='Run the same analysis several times and compare the statuses')
    stability_parser.add_argument('--provider', help='Hosted provider name', default=None)
    stability_parser.add_argument('--model', help='Exact model ID', required=True)
    stability_parser.add_argument('--live', action='store_true', help='Use live sources (costs API credits on every run)')
    stability_parser.add_argument('-n', '--runs', type=int, default=5, help='Number of runs (2 to 20, default 5)')
    stability_parser.add_argument('--pause', type=float, default=2.0, help='Seconds to wait between runs')
    stability_parser.add_argument('--out', help='Also write the comparison to this Markdown file')

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


def stability_command(args: argparse.Namespace) -> int:
    from .stability import compare, render

    if not 2 <= args.runs <= 20:
        print('--runs must be between 2 and 20.', file=sys.stderr)
        return 2
    reports: list[str | None] = []
    seconds: list[float | None] = []
    failures: list[str | None] = []
    for index in range(args.runs):
        if index and args.pause > 0:
            time.sleep(args.pause)
        print(f'Run {index + 1} of {args.runs}...', file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            result = APP.invoke({'live_mode': bool(args.live), 'provider': args.provider, 'model': args.model})
        except Exception as exc:  # one bad run must not throw away the others
            result = {'error': f'{type(exc).__name__}: {exc}'}
        seconds.append(time.perf_counter() - started)
        report = result.get('report_markdown')
        error = result.get('error')
        usable = bool(report) and not error
        reports.append(report if usable else None)
        failures.append(None if usable else (error or 'no report was produced'))
    text = render(compare(reports, args.model, seconds, failures))
    if not args.live:
        text = ('> Offline fixtures: nothing is assessed in this mode, so this comparison says nothing about '
                'stability. Add --live to run against real sources.\n\n') + text
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
        print(f'Written to {args.out}', file=sys.stderr)
    return 0 if sum(r is not None for r in reports) >= 2 else 2


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
    if args.command == 'stability':
        return stability_command(args)
    if args.command == 'index-regulatory':
        return index_regulatory_command()

    parser.error('Unknown command.')
    return 2
