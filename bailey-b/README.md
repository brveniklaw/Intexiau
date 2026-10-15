# Bailey B — Retrieval Module

Internal retrieval layer for Bay Area Legal Services. One corpus (Florida Statutes,
CourtListener, Florida appellate opinions), one retrieval function, one citation
verification pass, many consumers. The full spec is in [docs/BUILD.md](docs/BUILD.md).

## Status

| Build step | Package | Status |
|---|---|---|
| 0. Scaffold: layout, deps, config | — | done |
| Eval harness (runs alongside step 1) | `bailey_b.evals` | harness done; **30 draft questions awaiting attorney review** |
| 1. Corpus download + chunking (statutes first) | `bailey_b.corpus` | **statutes done** (1,273 sections → 1,479 chunks); case law next |
| 2. Embedding + vector store | `bailey_b.index` | **done**: all 1,479 statute chunks embedded (24 MB in Postgres) |
| 3. Hybrid search + reranking | `bailey_b.search` | **done**; smoke recall@5 0.82 (18/22); probate terms-of-art gap open |
| 4. Citation verification | `bailey_b.verify` | **done**: 0 wrong calls on 3,031 real statute cross-references |
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
pytest                                      # DB tests need BAILEY_TEST_POSTGRES_DSN
```

The database tests create and drop their own table. Point them at a dev database:
`BAILEY_TEST_POSTGRES_DSN=postgresql://bailey:bailey@localhost:5432/bailey_b pytest`.
pgvector must be 0.7 or newer (sparse vectors); Ubuntu's packaged 0.6 is too old.

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

## Index: embeddings + pgvector

```bash
bailey-b index statutes             # embed new/changed chunks into Postgres
bailey-b index statutes --limit 50  # quick trial
```

- **Model:** bge-m3, one pass giving a 1,024-dim dense vector and a sparse
  token-weight vector (hybrid search in step 3).
- **Table:** `bailey_b_chunks`: chunk fields + `dense vector(1024)` (HNSW,
  cosine) + `sparse sparsevec(250002)` (exact inner product).
- **First full run:** about 2.5 hours on 4 CPU cores, no GPU (~0.15 chunks/s). A GPU
  cuts this to minutes; after that, refreshes only touch changed sections.
- **Notes for step 3** from spot checks: dedupe pieces of the same section in
  results (one § 61.13 question returned three pieces of it), and check that
  remedy sections surface (a "what can she sue for" question found § 559.72 but
  not § 559.77).
- **Incremental:** only chunks whose text or model changed are re-embedded;
  removed sections are deleted. Indexing refuses to delete more than 20% of a
  corpus (a sign of a broken download) unless `--allow-deletes` is passed.

## Search: hybrid + rerank

```bash
bailey-b search "Tenant got a 3-day notice on a Friday..."   # top 5 with scores
bailey-b eval compare evals/smoke.jsonl                      # 4 configurations side by side
```

Meaning search (40) + keyword search (40) → reciprocal rank fusion → one result
per section → top 20 → bge-reranker-v2-m3 → best 5. Scores shown to attorneys are
the reranker's 0–1 relevance.

**Smoke results (Oct 15, 2026).** 22 plain-language questions in `evals/smoke.jsonl`,
one expected section each, written by Claude and marked UNREVIEWED. They compare
configurations; they do not replace the attorney-reviewed eval set.

| Configuration | Found in top 5 | Recall@5 | MRR |
|---|---|---|---|
| Meaning only | 16/22 | 0.73 | 0.56 |
| Keywords only | 12/22 | 0.55 | 0.36 |
| Hybrid (RRF) | 16/22 | 0.73 | 0.54 |
| Hybrid + rerank | **18/22** | **0.82** | 0.55 |

- Reranking fixed the defective-notice question (§ 83.60, now #1) and the
  collector-remedies question (§ 559.77, missing from every other configuration).
- **All 4 misses are probate/guardianship, and all are vocabulary gaps**: everyday
  words vs. terms of art ("widow … without a will" vs. "surviving spouse …
  intestate"; "wife claim a share" vs. "elective share"; "manage his affairs" vs.
  "incapacity"; "simpler than probate" vs. "summary administration"). Three of the
  four were among the 20 candidates but the general-purpose reranker scored them
  near zero; § 735.201 never reached the candidates.
- **Next fix, to be measured on the attorney eval set (not tuned to these 22):** an
  attorney-maintained glossary that expands plain words to Florida terms of art
  before searching, and blending the fusion rank into the final order so the
  reranker cannot bury a strong candidate. The attorney feedback loop (roadmap
  10.3) is the long-term fix for the reranker itself.
- **Speed:** search without reranking is under 1 s; the reranker takes ~55 s per
  question on this 4-core CPU. Production needs a GPU for it, or `rerank=false`.

## Verify: citation checking

```bash
bailey-b verify "Under § 83.60(2) ... see also § 83.66."      # exit code 1 if anything is flagged
bailey-b verify --file draft.txt --sources 83.60,83.56          # also flag cites not in the sources
```

Every citation in model-written text gets one of five results:

| Result | Meaning | Blocks the text? |
|---|---|---|
| verified | Section (and subsection, if cited) exists in the corpus | no |
| subsection_missing | Section exists, cited subsection does not | **yes** |
| not_found | Section is in an indexed chapter but does not exist (confabulation) | **yes** |
| outside_corpus | Florida statute in a chapter not indexed yet; not checked | no, shown as unchecked |
| unverifiable | Case law, court rules, federal statutes; not in the corpus yet | no, shown as unchecked |

When the retrieved sources are given, a real citation the model was *not* given is
also flagged: the model should cite only what retrieval handed it.

**Validated on real data:** run over every cross-reference inside the 2026 statutes
themselves (3,031 citations), it verified 1,684, reported 1,108 as outside the
indexed chapters, and flagged 239, every one a "Former s. …" note pointing at a
renumbered or repealed section. No live cross-reference was wrongly flagged.

It catches invented and mistyped citations. It does **not** catch a real section
cited for something it does not say; that stays with the attorney.

## Eval set (draft)

`evals/questions.jsonl`: 30 questions written by Claude from the 2026 statute text,
weighted to family law (10) and eviction (8), plus probate/guardianship (6),
consumer (4) and foreclosure (2). Each has expected sections and a short answer
citing the statute, and `reviewed_by: "PENDING"`. A BALS attorney should confirm or
fix each one and put their initials in `reviewed_by`; add, remove or rewrite freely.
**First scores (Oct 15, 2026), right section in the top 5:**

| Configuration | Smoke set (22) | Draft eval set (30) |
|---|---|---|
| Meaning only | 16 | **26** |
| Keywords only | 12 | 14 |
| Hybrid (RRF) | 16 | 18 |
| Hybrid + rerank | **18** | 24 |

Diagnosis: the 20-candidate shortlist almost always contains the right section
(21/22 and 29/30), so fusion is not losing answers. The general-purpose reranker
is: it demotes correct sections that meaning search ranked well. Candidate fixes, to
be chosen on attorney-reviewed questions: blend the reranker score with the fusion
rank instead of replacing it; a legal-domain reranker; the attorney feedback loop.
Both question sets were written by Claude from the statute text, which may favor
meaning search; attorney-written questions are the real test.

Known gap: domestic violence injunctions (ch. 741) and eviction service (ch. 48) are
not in the indexed chapters, so questions on them cannot be answered from the corpus.

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
