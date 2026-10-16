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


def _search(args: argparse.Namespace) -> int:
    from bailey_b.search import Mode, default_retriever

    retriever = default_retriever()
    results = retriever.retrieve(
        args.question, top_n=args.n, mode=Mode(args.mode), rerank=not args.no_rerank
    )
    for r in results:
        c = r.chunk
        piece = f"  [{c.metadata['subsections']}]" if c.metadata.get("subsections") else ""
        print(f"{r.rank}. {c.display_citation}  {c.title}{piece}")
        dense = f"{r.dense_score:.3f}" if r.dense_score is not None else "-"
        sparse = f"{r.sparse_score:.3f}" if r.sparse_score is not None else "-"
        print(f"   score {r.score:.3f}  (meaning {dense}, keywords {sparse})  {c.source_url}")
    return 0


def _eval_compare(args: argparse.Namespace) -> int:
    import time

    from bailey_b.search import Mode, default_retriever

    questions = load_eval_set(args.path)
    retriever = default_retriever()
    configs = [
        ("meaning only", Mode.DENSE, False),
        ("keywords only", Mode.SPARSE, False),
        ("hybrid (RRF)", Mode.HYBRID, False),
    ]
    if not args.no_rerank:
        configs.append(("hybrid + rerank", Mode.HYBRID, True))
    reports = {}
    for name, mode, rerank in configs:
        started = time.monotonic()
        reports[name] = run_eval(
            questions,
            lambda q, mode=mode, rerank=rerank: retriever.retrieve(
                q, top_n=args.k, mode=mode, rerank=rerank
            ),
            k=args.k,
        )
        per_q = (time.monotonic() - started) / max(len(questions), 1)
        print(f"  ran {name}: {per_q:.1f}s per question", file=sys.stderr, flush=True)

    print(f"\n{len(questions)} questions, k={args.k}\n")
    print(f"{'configuration':<18}{'recall@k':>10}{'MRR':>8}  found")
    for name, rep in reports.items():
        found = sum(1 for r in rep.results if not r.missed)
        print(f"{name:<18}{rep.mean_recall_at_k:>10.2f}{rep.mrr:>8.2f}  {found}/{rep.n}")
    best = list(reports.values())[-1]
    misses = [r for r in best.results if r.missed]
    if misses:
        print(f"\nmissed by {list(reports)[-1]}:")
        for r in misses:
            print(f"  {r.id}: wanted {', '.join(r.missed)}; got {', '.join(r.retrieved)}")
    if args.json:
        args.json.write_text(
            json.dumps({n: r.model_dump(mode="json") for n, r in reports.items()}, indent=2)
        )
    return 0


def _eval_blend(args: argparse.Namespace) -> int:
    """Grid over keyword weight x reranker weight, reranking each candidate only once."""
    from bailey_b.search import Mode, blend, default_retriever

    retriever = default_retriever()
    base = retriever.settings
    cache_path = get_settings().corpus.data_dir / "rerank-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    sparse_weights = [float(x) for x in args.sparse_weights.split(",")]
    rerank_weights = [float(x) for x in args.rerank_weights.split(",")]
    sets = {p.stem: load_eval_set(p) for p in args.paths}
    hits: dict[tuple[str, float, float], int] = {}
    dense_only: dict[str, int] = {}

    for set_name, questions in sets.items():
        for qi, q in enumerate(questions, 1):
            print(f"  {set_name} {qi}/{len(questions)}", file=sys.stderr, flush=True)
            want = set(q.expected_citations)
            retriever.settings = base
            top = [c.chunk.citation_id for c in retriever.candidates(q.question, Mode.DENSE)]
            dense_only[set_name] = dense_only.get(set_name, 0) + (want <= set(top[: args.k]))
            for sw in sparse_weights:
                retriever.settings = base.model_copy(update={"sparse_weight": sw})
                cands = retriever.candidates(q.question)
                scores = cache.setdefault(q.question, {})
                todo = [c for c in cands if c.chunk.chunk_id not in scores]
                if todo:
                    for c, sc in zip(
                        todo,
                        retriever.reranker.score(q.question, [c.chunk.text for c in todo]),
                        strict=True,
                    ):
                        scores[c.chunk.chunk_id] = sc
                    cache_path.write_text(json.dumps(cache))
                for c in cands:
                    c.rerank = scores[c.chunk.chunk_id]
                for rw in rerank_weights:
                    order = [c.chunk.citation_id for c in blend(list(cands), rw)]
                    key = (set_name, sw, rw)
                    hits[key] = hits.get(key, 0) + (want <= set(order[: args.k]))
    retriever.settings = base

    total = sum(len(qs) for qs in sets.values())
    names = list(sets)
    print(
        f"\nRight section(s) in the top {args.k}. Rows: keyword weight; columns: reranker weight."
    )
    print(
        "dense only (no keywords, no reranker): "
        + ", ".join(f"{n} {dense_only[n]}/{len(sets[n])}" for n in names)
        + f", all {sum(dense_only.values())}/{total}"
    )
    header = "kw weight " + "".join(f"{'rr ' + str(rw):>12}" for rw in rerank_weights)
    print(header)
    for sw in sparse_weights:
        cells = []
        for rw in rerank_weights:
            n = sum(hits[(name, sw, rw)] for name in names)
            cells.append(f"{n:>9}/{total}")
        print(f"{sw:<10}" + "".join(f"{c:>12}" for c in cells))
    if len(names) > 1:
        for name in names:
            print(f"\n{name}:")
            for sw in sparse_weights:
                print(
                    f"{sw:<10}"
                    + "".join(
                        f"{hits[(name, sw, rw)]:>9}/{len(sets[name])}" + " " * 2
                        for rw in rerank_weights
                    )
                )
    return 0


def _verify(args: argparse.Namespace) -> int:
    from bailey_b.corpus import statutes
    from bailey_b.verify import CorpusIndex, Status, verify

    settings = get_settings().corpus
    text = args.file.read_text(encoding="utf-8") if args.file else args.text
    if not text:
        print("give the text to check, or --file", file=sys.stderr)
        return 2
    corpus = CorpusIndex.from_chunks(
        statutes.read_chunks(statutes.chunks_path(settings)), year=settings.statutes_year
    )
    sources = [f"fl-stat:{s}" for s in args.sources.split(",")] if args.sources else None
    report = verify(text, corpus, sources=sources)
    marks = {
        Status.VERIFIED: "OK  ",
        Status.SUBSECTION_MISSING: "FLAG",
        Status.NOT_FOUND: "FLAG",
        Status.OUTSIDE_CORPUS: "??  ",
        Status.UNVERIFIABLE: "??  ",
    }
    for c in report.checks:
        mark = "FLAG" if c.in_sources is False else marks[c.status]
        print(f"{mark}  {c.raw:<28} {c.message}")
    n = len(report.flagged)
    print(f"\n{len(report.checks)} citations; {n} flagged" + ("" if n else "; nothing flagged"))
    return 1 if n else 0


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

    se = sub.add_parser("search", help="search the corpus for one question")
    se.add_argument("question")
    se.add_argument("-n", type=int, default=5, help="results to show")
    se.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    se.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    se.set_defaults(func=_search)

    vf = sub.add_parser("verify", help="check every citation in a piece of text")
    vf.add_argument("text", nargs="?", default="")
    vf.add_argument("--file", type=Path, help="read the text from a file")
    vf.add_argument("--sources", help="comma-separated sections the answer was based on")
    vf.set_defaults(func=_verify)

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
    cmp = ev.add_parser("compare", help="score dense, sparse, hybrid and reranked side by side")
    cmp.add_argument("path", type=Path)
    cmp.add_argument("-k", type=int, default=5, help="cutoff (default 5)")
    cmp.add_argument("--no-rerank", action="store_true", help="skip the slow reranked run")
    cmp.add_argument("--json", type=Path, help="write full per-question results here")
    cmp.set_defaults(func=_eval_compare)
    bl = ev.add_parser("blend", help="grid over keyword weight and reranker weight")
    bl.add_argument("paths", type=Path, nargs="+")
    bl.add_argument("-k", type=int, default=5)
    bl.add_argument("--sparse-weights", default="1.0,0.5,0.25,0.0")
    bl.add_argument("--rerank-weights", default="1.0,0.75,0.5,0.25,0.0")
    bl.set_defaults(func=_eval_blend)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
