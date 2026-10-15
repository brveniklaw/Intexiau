# Bailey B — Retrieval Module: Master Build Document

**Purpose:** Internal retrieval layer for Bay Area Legal Services (BALS). Serves attorney-facing and staff-facing tools first (intake triage, research, document drafting support). Client-facing pro se tools come later. One corpus, one verification pass, many consumers.

## 1. Corpus Sources (all free, all API-accessible or downloadable)

- **Florida Statutes:** Online Sunshine (flsenate.gov).
  - *Implemented as:* per-chapter pages at `flsenate.gov/Laws/Statutes/<year>/Chapter<N>/All`. The Online Sunshine zip host (leg.state.fl.us) did not resolve from the build environment; the Senate pages carry the same official text with per-section markup. Current-year searchable zip, chunked by section (each section is a natural unit with its own citation). Refresh annually when the new edition drops. Use the LegiScan free tier (10,000 queries/month) to catch interim amendments between editions.
- **Federal case law:** CourtListener API via Free Law Project LSO Group Membership (300 calls/day, unlimited docket/opinion alerts, MCP integration). Query on demand at question time; cache results in the vector store to preserve the rate limit.
- **Florida state case law:** CourtListener is federal-heavy, so supplement with Second DCA opinions (2dca.flcourts.gov) and Florida Supreme Court opinions (supremecourt.flcourts.gov). Also Google Scholar's Florida case law filter and the Caselaw Access Project (all published US decisions through June 2018) for depth.

**Coverage pattern:** Statutes are fully covered across all BALS practice areas (family law Ch. 61/39/742/743, eviction Ch. 83, consumer Ch. 501/559/721, foreclosure Ch. 702, probate Ch. 731–736/744). The weak link is Florida appellate case law, especially for family law and eviction, BALS's highest-volume areas. Immigration and benefits are federal, so CourtListener covers those well.

**LexisNexis is NOT a retrieval source.** Its terms prohibit placing Licensed Content in any third-party application, prohibit automated access, and ban agentic AI querying without prior written permission. Lexis stays a human-facing verification backstop for attorneys. Nonprofit pricing exists but belongs in the attorney workflow, not the platform.

## 2. Indexing Pipeline

- Chunk by statute section (natural unit, citation-preserving). For case law, chunk by opinion with headnotes/syllabus where available.
- Embed with bge-m3: dense + sparse vectors in one pass, giving hybrid search for free.
- Store in pgvector or Qdrant. Both are open source and cheap to run on a single server.
  - **Decided:** pgvector on Supabase (BALS-owned, paid plan). Qdrant remains a supported local alternative.

## 3. Query Path

1. Embed the incoming question.
2. Run hybrid search (dense + sparse) against the index.
3. Pull the top 20 candidates.
4. Rerank with a cross-encoder; keep the best 5.
5. Feed those 5 to the model with citations attached.

## 4. Citation Verification Layer (build LAST)

Every citation the model produces gets checked against the corpus before it reaches the user: a real statute section or a real CourtListener opinion ID, or it gets flagged. Legal citation is a closed-world problem (every citation either exists in the corpus or doesn't), which makes verification tractable. This layer catches confabulated citations but not subtle reasoning errors (a real case cited for a proposition it doesn't support). That's a human-in-the-loop problem.

## 5. Architecture Decisions (already made — do not re-litigate)

- **Single retrieval function:** one function that takes a question and returns ranked chunks with citations. The model underneath is swappable. If GLM-Legal or another model beats Claude on legal tasks later, nothing gets rebuilt.
- **Attorney vs. client visibility split:** attorneys see citations and confidence scores; clients get plain-language answers. Same engine, different presentation. (Future feature, not in v1.)
- **Internal first:** attorney/staff-facing before pro se. Attorneys catch errors, and their workflows are structured enough to automate.

## 6. Build Order

1. Corpus download + chunking (statutes first, the easiest win).
2. Embedding + vector store.
3. Hybrid search + reranking.
4. Citation verification.
5. Attorney-facing interface (intake triage is the first workflow to wire up).
6. Draft analysis workflow (depends on verification being solid; see section 9).
7. Roadmap features (section 10). Ship one at a time, deadline tracking first.

## 7. The Hard Parts (where the months go, not the build)

- **Eval set:** 20+ real BALS questions with known-good answers, built BEFORE wiring anything. Measure whether retrieval surfaces the right chunks. If it misses, no model choice saves you.
- **Tuning:** semantic drift (a question about "change of control" missing a clause that says "transfer of ownership") is the core failure mode. Hybrid search + reranking is the mitigation.
- **Staleness:** Florida law changes constantly. A corpus that can't refresh is a liability.
- **UPL compliance:** the self-help desk must give information without crossing into unauthorized practice of law. This is a legal design problem, not a technical one, and it's the founder's domain.

## 8. Cost Reality Check

The entire stack runs on one server. Statutes are free. CourtListener LSO membership is free for nonprofit legal aid. The only real costs are compute for embedding/reranking and engineering time. This is a near-zero-cost architecture, which is exactly the pitch if BALS ever scales Bailey B beyond internal use.

## 9. Draft Analysis Workflow (attorney-facing)

**Use case:** An attorney has an initial draft of a memo, motion, or brief and uploads it to get AI help analyzing the reasoning and amplifying the citations and authorities, to make a more persuasive and authoritative argument.

**Design principle:** Same retrieval function, different query path. No new infrastructure. The draft itself becomes the query.

- The draft gets chunked by paragraph.
- Each paragraph becomes a retrieval query against the corpus.
- Results are compared against what the draft already cites.
- Output is a gap report, not an answer: claims with no supporting authority, citations that don't actually support the proposition, and controlling authorities the corpus shows are missing.

**Two critical design choices:**

1. Run the gap analysis BEFORE the model suggests new citations. Otherwise it fills gaps with confabulated ones and the verification layer just catches them after the fact.
2. Present the report as suggestions for the attorney to accept or reject, never auto-inserted text. Attorneys must own the argument.

**Build dependency:** This workflow comes after the citation verification layer is working. It is the first workflow that depends on verification being solid, because the model will be generating new citations to fill gaps, and every one of those needs checking before it reaches the attorney.

## 10. Roadmap Features (ship one at a time)

### 10.1 Deadline and Statute-of-Limitations Tracking (FIRST)

The retrieval layer already knows Florida law, so it can flag when a filing deadline is approaching based on the case type and filing date. Smallest build, highest trust payoff. Binary stakes (you file on time or you lose) make attorneys trust it fastest. It also gives attorneys a daily reason to open the tool.

### 10.2 Document Generation from Templates

Once retrieval surfaces the right statute and form, the model drafts the pleading or motion with citations attached. This is where the draft analysis workflow pays off: the attorney uploads a draft, gets the gap report, and the system can regenerate the corrected version. Build only after draft analysis is proven.

### 10.3 Feedback Loop

Attorneys mark which retrieved chunks were actually useful, and that signal retrains the reranker over time. Most legal AI products never close this loop, which is why they plateau. This is the long-term differentiator. Start collecting the signal early even if you don't act on it until later.

**Sequencing rule:** Do not build all three at once. Ship deadline tracking first. Each feature should earn its place by proving the previous one works.

## Working Rules

1. Scaffold the project structure first (directory layout, dependencies, config) before writing any retrieval logic.
2. Implement each build-order step (section 6) as a separate session, so each piece can be reviewed before the next compounds on it.
3. Build the eval set (section 7) in parallel with step 1. Don't wait for the pipeline to be finished.
4. When a session produces working code, commit it before starting the next session so there's a clean rollback point.
