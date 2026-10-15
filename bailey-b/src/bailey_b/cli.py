"""Command-line entry point: ``bailey-b <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bailey_b.config import get_settings
from bailey_b.evals import MIN_EVAL_QUESTIONS, load_eval_set, run_eval


def _config(_: argparse.Namespace) -> int:
    print(get_settings().model_dump_json(indent=2))
    return 0


def _corpus_statutes(args: argparse.Namespace) -> int:
    from bailey_b.corpus import statutes

    settings = get_settings().corpus
    if not args.skip_download:
        manifest = statutes.download(settings, force=args.force)
        mb = sum(v["bytes"] for v in manifest.values()) / 1e6
        print(f"raw: {len(manifest)} chapters, {mb:.1f} MB in {statutes.raw_dir(settings)}")
    chunks = list(statutes.iter_chunks(settings, max_words=args.max_words))
    path = statutes.chunks_path(settings)
    statutes.write_chunks(chunks, path)
    by_chapter = Counter(c.metadata["chapter"] for c in chunks)
    sections = len({c.citation_id for c in chunks})
    print(f"chunks: {len(chunks)} from {sections} sections -> {path}")
    for chapter in settings.statute_chapters:
        print(f"  ch. {chapter}: {by_chapter[chapter]}")
    return 0


def _index_statutes(args: argparse.Namespace) -> int:
    import time

    from bailey_b.config import VectorBackend
    from bailey_b.corpus import statutes
    from bailey_b.index import index_chunks
    from bailey_b.index.embedder import BGEM3Embedder
    from bailey_b.index.pgvector_store import PgVectorStore

    settings = get_settings()
    if settings.vector_store.backend != VectorBackend.PGVECTOR:
        print("only the pgvector backend is implemented", file=sys.stderr)
        return 1
    chunks = statutes.read_chunks(statutes.chunks_path(settings.corpus))
    if args.limit:
        chunks = chunks[: args.limit]
    embedder = BGEM3Embedder(settings.embed)
    started = time.monotonic()

    def progress(done: int, total: int) -> None:
        rate = done / max(time.monotonic() - started, 1e-9)
        eta = (total - done) / rate if rate else 0
        print(f"  embedded {done}/{total}  ({rate:.2f}/s, ~{eta / 60:.0f} min left)", flush=True)

    with PgVectorStore(
        settings.vector_store.postgres_dsn.get_secret_value(),
        settings.vector_store.collection,
        dense_dim=embedder.dense_dim,
        sparse_dim=embedder.sparse_dim,
    ) as store:
        store.ensure_schema()
        report = index_chunks(
            chunks,
            corpus="fl-statutes",
            store=store,
            embedder=embedder,
            batch_size=settings.embed.batch_size,
            max_delete_fraction=1.0 if args.allow_deletes else 0.2,
            progress=progress,
        )
        counts = store.counts()
    print(f"embedded {report.embedded}, unchanged {report.unchanged}, deleted {report.deleted}")
    for corpus, (n, cites) in counts.items():
        print(f"  {corpus}: {n} chunks, {cites} citations")
    return 0


def _eval_validate(args: argparse.Namespace) -> int:
    try:
        questions = load_eval_set(args.path)
    except ValueError as e:
        print(f"invalid: {e}", file=sys.stderr)
        return 1
    areas = Counter(q.practice_area.value for q in questions)
    print(f"{len(questions)} questions OK")
    for area, n in sorted(areas.items()):
        print(f"  {area}: {n}")
    if len(questions) < MIN_EVAL_QUESTIONS:
        print(f"warning: fewer than {MIN_EVAL_QUESTIONS} questions", file=sys.stderr)
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    from bailey_b.search import retrieve

    questions = load_eval_set(args.path)
    k = args.k or get_settings().search.top_n
    report = run_eval(questions, lambda q: retrieve(q, top_n=k), k=k)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bailey-b")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print resolved settings").set_defaults(func=_config)

    corpus = sub.add_parser("corpus", help="download and chunk sources").add_subparsers(
        dest="corpus_command", required=True
    )
    st = corpus.add_parser("statutes", help="Florida Statutes: download chapters, write chunks")
    st.add_argument("--force", action="store_true", help="re-download even if present")
    st.add_argument("--skip-download", action="store_true", help="chunk existing raw files only")
    st.add_argument("--max-words", type=int, default=1200, help="split sections longer than this")
    st.set_defaults(func=_corpus_statutes)

    ix = sub.add_parser("index", help="embed chunks into the vector store").add_subparsers(
        dest="index_command", required=True
    )
    ixs = ix.add_parser("statutes", help="embed the Florida Statutes chunk file")
    ixs.add_argument("--limit", type=int, default=0, help="only the first N chunks (testing)")
    ixs.add_argument(
        "--allow-deletes", action="store_true", help="allow removing >20%% of stored chunks"
    )
    ixs.set_defaults(func=_index_statutes)

    ev = sub.add_parser("eval", help="retrieval eval set").add_subparsers(
        dest="eval_command", required=True
    )
    v = ev.add_parser("validate", help="check an eval JSONL file")
    v.add_argument("path", type=Path)
    v.set_defaults(func=_eval_validate)
    r = ev.add_parser("run", help="score retrieval against an eval JSONL file")
    r.add_argument("path", type=Path)
    r.add_argument("-k", type=int, default=None, help="cutoff (default: search.top_n)")
    r.set_defaults(func=_eval_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
