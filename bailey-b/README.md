# Bailey B — Retrieval Module

Internal retrieval layer for Bay Area Legal Services. One corpus (Florida Statutes,
CourtListener, Florida appellate opinions), one retrieval function, one citation
verification pass, many consumers. The full spec is in [docs/BUILD.md](docs/BUILD.md).

## Status

| Build step | Package | Status |
|---|---|---|
| 0. Scaffold: layout, deps, config | — | done |
| Eval harness (runs alongside step 1) | `bailey_b.evals` | harness done; **questions needed** |
| 1. Corpus download + chunking (statutes first) | `bailey_b.corpus` | **statutes done** (1,273 sections → 1,479 chunks); case law next |
| 2. Embedding + vector store | `bailey_b.index` | not started |
| 3. Hybrid search + reranking | `bailey_b.search` | interface only (`retrieve`) |
| 4. Citation verification | `bailey_b.verify` | not started |
| 5. Attorney interface (intake triage) | — | not started |
| 6. Draft analysis | — | not started |
| 7. Roadmap (deadline tracking first) | — | not started |

Each step is its own session and its own commit, reviewed before the next one starts.

## Layout

```
bailey-b/
├── docs/BUILD.md            master build document
├── src/bailey_b/
│   ├── config.py            settings from env / .env (prefix BAILEY_)
│   ├── models.py            Chunk, RetrievedChunk, canonical citation IDs
│   ├── corpus/              step 1
│   ├── index/               step 2
│   ├── search/              step 3: retrieve(), the single retrieval function
│   ├── verify/              step 4
│   ├── evals/               eval set schema, loader, recall@k / MRR
│   └── cli.py               `bailey-b` command
├── evals/                   eval question files (JSONL)
├── data/                    downloaded corpus + chunks (gitignored)
├── tests/
└── docker-compose.yml       local Postgres + pgvector (stand-in for Supabase)
```

## Setup

```bash
cd bailey-b
uv venv && uv pip install -e ".[dev]"       # core + tests
uv pip install -e ".[corpus]"               # step 1
uv pip install -e ".[embed,pgvector]"       # steps 2-3
cp .env.example .env                        # add CourtListener / LegiScan keys
docker compose --profile pgvector up -d     # local stand-in for Supabase
pytest
```

## Hosting

- **Database:** Supabase (Postgres + pgvector), on a BALS-owned paid plan. Paid
  projects don't auto-pause; the free tier pauses after inactivity, so it is
  not used for anything beyond throwaway testing.
- **Models + scheduled refreshes:** a small always-on cloud VM (Supabase does
  not run bge-m3 or the reranker).
- **Client material** (intake questions, uploaded drafts, private evals) lives
  only in BALS-owned systems, never in this repo or a personal account.

## Corpus: Florida Statutes

```bash
bailey-b corpus statutes            # download the 16 BALS chapters, then chunk
bailey-b corpus statutes --force    # re-download (annual refresh)
```

- **Source:** `flsenate.gov/Laws/Statutes/<year>/Chapter<N>/All`, one page per
  chapter, about 7 MB total. Each raw file is recorded in `manifest.json` with its
  URL, retrieval time and SHA-256.
- **Output:** `data/chunks/fl-statutes-<year>.jsonl`, one JSON `Chunk` per line.
- **One section = one chunk.** About 7% of sections run over 1,200 words; those
  are split at subsection boundaries into pieces that share the section's
  citation ID. `metadata.subsections` records the span, e.g. `(9)(a)7.`.
- **Chunk text** = citation and catchline, chapter and part, the section body, and
  any notes. Notes stay in the text because they can change what a section means.
  Legislative history goes to `metadata.history`.

## Citation IDs

Every chunk carries a canonical citation ID. Verification and evals both compare
against these, so they must be stable:

| Source | Format | Example |
|---|---|---|
| Florida Statutes section | `fl-stat:<section>` | `fl-stat:83.56` |
| CourtListener opinion | `cl-opinion:<id>` | `cl-opinion:2481923` |
| Florida Second DCA | `fl-2dca:<docket>` | `fl-2dca:2D2024-1234` |
| Florida Supreme Court | `fl-sc:<docket>` | `fl-sc:SC2023-0456` |
| Caselaw Access Project | `cap:<case_id>` | `cap:1234567` |

## Eval set

The eval set is built in parallel with step 1 and scored from day one. Format: one JSON
object per line in `evals/questions.jsonl`. See `evals/questions.example.jsonl`.

| Field | Meaning |
|---|---|
| `id` | stable unique ID, e.g. `eviction-007` |
| `question` | the question as it actually arrives at intake |
| `practice_area` | family, eviction, consumer, foreclosure, probate, immigration, benefits, other |
| `expected_citations` | canonical IDs a correct retrieval must surface |
| `answer` | known-good answer |
| `reviewed_by` | initials of the BALS attorney who reviewed it |

Target: 20+ questions, weighted toward family law and eviction (highest volume, weakest
case-law coverage). **Strip client names and identifying facts before committing.**
Anything not yet anonymized goes in `evals/private/`, which is gitignored.

```bash
bailey-b eval validate evals/questions.jsonl     # schema check + per-area counts
bailey-b eval run evals/questions.jsonl -k 5     # recall@k and MRR (after step 3)
```
