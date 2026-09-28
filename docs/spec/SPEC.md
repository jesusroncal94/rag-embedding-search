# SPEC — Internal Documentation Q&A (RAG) Proof of Concept

**Status:** v0.5 — implemented, with a Streamlit ui; 60 unit scenarios and 1 ui test green; `@live` run in part against `gemini-3.6-flash` (§11.5)
**Date:** 2026-09-17
**Deliverable:** (a) a containerized working PoC, (b) a write-up of the architecture and the path to production.

---

## 1. Business case

> **Note on framing (v0.3).** The hypothesis describes "technical documentation" for engineers. The
> supplied corpus (`corpus/`) is customer-facing **banking policy** — payments, card security,
> disputes, onboarding. The business case below is written against what the corpus actually is.
> If engineering documentation was intended, §1.2 changes substantially; flagged as R-10.

### 1.1 Hypothesis under test

> An internal API that answers natural-language questions over our technical documentation
> would significantly improve engineering productivity.

The PoC does **not** prove the productivity claim. It proves the narrower, prerequisite claim:

> **H-TECH:** The specified pipeline (paragraph chunking → Gemini Embedding → vector
> retrieval → Gemini 2.5 Flash) produces grounded, useful answers over our documentation
> corpus at acceptable latency and cost, using a stack we can operate.

Only if H-TECH holds is it rational to fund the productivity experiment.

### 1.2 Value model (to be validated, not asserted)

| Lever | Mechanism | Leading indicator we could instrument later |
|---|---|---|
| Search time avoided | Answer + citation replaces manual doc traversal | median time-to-answer per query |
| Interrupt deflection | Fewer "where is X documented?" pings to senior engineers | volume of doc-lookup questions in support channels |
| Onboarding ramp | New agents self-serve policy knowledge | time-to-independent-handling |
| Doc quality feedback | Unanswerable questions expose documentation gaps | rate of low-confidence / no-context answers |
| Consistency | Every agent gives the *same* policy answer, with a citation | variance in answers to identical questions |

For a regulated policy corpus, **consistency and citation are worth more than convenience**. The
risk being managed is not "an engineer wastes ten minutes" but "an agent states the wrong
liability cap to a customer". This reframes the whole system: it is an *assistive retrieval tool
with mandatory sourcing*, not an answer oracle — which is precisely why FR-API-05 (citations),
FR-API-06a (deterministic refusal) and FR-API-13 (no parametric supplement) are non-negotiable
rather than nice-to-have.

The fourth lever is the sleeper: a RAG system is also a **documentation gap detector**. It has
value even when it fails to answer, provided failures are logged. This is a deliberate part of
the strategy narrative.

### 1.3 Cost envelope of the PoC

Order-of-magnitude only; refined once corpus size is fixed (see Q6).

- One-time ingestion: `n_chunks × 1 embedding call`. Re-run on corpus change.
- Per query: `1 embedding call + 1 LLM call over k chunks of context`.
- Dominant production cost driver is **query volume × context size**, not storage.

### 1.4 Explicit non-goals for the PoC

- Not multi-tenant, not authenticated, not rate-limited.
- No evaluation harness beyond a smoke-level answer-quality check.
- No UI. API only.
- No incremental / streaming re-ingestion. Full rebuild only.
- No horizontal scaling, HA, or persistence guarantees beyond a Docker volume.

Each of these is a named item on the **production roadmap** (§6) — the PoC's job is to make the
gap between "works on my laptop" and "runs for 500 engineers" explicit and costed, not to close it.

---

## 2. Scope of the PoC

### 2.1 In scope

1. `ingest.py` — chunk, embed, and load sample documents into a vector database.
2. A REST service exposing the RAG query flow.
3. `Dockerfile` + `docker-compose.yml` — full stack via one `docker-compose up`.
4. Structured logging of critical events only.
5. External configuration file for all hyperparameters.

### 2.2 Fixed up front (not open to redesign in the PoC)

| Component | Specified value |
|---|---|
| Chunking strategy | Paragraph-based |
| Embedding model | Gemini Embedding |
| LLM | Gemini 2.5 Flash · *(v0.5)* retired for new projects; the PoC runs `gemini-3.6-flash` (R-11) |

These are **givens**. Where I believe a given is suboptimal (see §7 Risks), the PoC implements
the given as specified and the concern is raised in §7 — not silently "improved".

---

## 3. Functional requirements

Notation: `MUST` / `SHOULD` / `MAY` per RFC 2119. Each requirement is traceable to a Gherkin
scenario in `docs/spec/features/`.

### 3.1 Ingestion (`FR-ING-*`)

| ID | Requirement |
|---|---|
| FR-ING-01 | The system MUST ingest all supported documents found under a configured source directory. |
| FR-ING-01a | *(v0.2)* Files of unsupported type MUST be skipped with a log line, not abort the run. |
| FR-ING-02 | The system MUST split each document into chunks on paragraph boundaries. |
| FR-ING-03 | Paragraph chunks smaller than `min_chunk_chars` MUST be merged forward into the next chunk. |
| FR-ING-03a | *(v0.2)* A short final paragraph with no successor MUST be stored as its own chunk, never discarded. |
| FR-ING-04 | Paragraph chunks larger than `max_chunk_chars` MUST be split at the nearest sentence boundary below the limit. |
| FR-ING-04a | *(v0.2)* A chunk over the limit with no sentence boundary MUST be hard-split at the limit, with a warning naming the document. |
| FR-ING-05 | Each chunk MUST be embedded using the configured Gemini embedding model. |
| FR-ING-06 | Each stored chunk MUST carry metadata: source document path, chunk ordinal, and character offsets. |
| FR-ING-06a | *(v0.2)* The collection MUST record the embedding model name and dimensionality in its own metadata. |
| FR-ING-07 | Embedding calls MUST be batched to `embed_batch_size` chunks per request. |
| FR-ING-08 | A transient embedding API failure MUST be retried with exponential backoff up to `max_retries`; permanent failure MUST abort ingestion with a non-zero exit code and a structured error log. |
| FR-ING-09 | Re-running ingestion MUST produce a corpus equivalent to a clean build (idempotent — collection is dropped and rebuilt). |
| FR-ING-10 | The script MUST emit a structured summary on completion: documents processed, chunks produced, embedding calls made, wall-clock duration. |

### 3.2 Query API (`FR-API-*`)

| ID | Requirement |
|---|---|
| FR-API-01 | The service MUST expose `POST /api/v1/query` accepting `{"question": string, "top_k": int?}`. |
| FR-API-02 | The service MUST embed the question with the same embedding model used at ingestion. |
| FR-API-03 | The service MUST retrieve the `top_k` nearest chunks by vector similarity (default from config). |
| FR-API-03a | *(v0.2)* A caller-supplied `top_k` outside `[1, max_top_k]` MUST yield HTTP 400 naming the permitted range. |
| FR-API-04 | The service MUST generate the answer with Gemini 2.5 Flash, grounded in the retrieved chunks via a configured prompt template. |
| FR-API-05 | The response MUST include the answer text and the citations (source document + chunk ordinal) of every chunk placed in the context window. |
| FR-API-05a | *(v0.5)* Each citation MUST also carry the text of its chunk (`snippet`) and whether that chunk contains instruction-like text (`injection_suspected`). Only chunks already placed in the context are returned; the flag is observability, never a filter. |
| FR-API-06 | When no chunk scores above `min_similarity`, the service MUST return an explicit "insufficient context" answer rather than an ungrounded generation. |
| FR-API-06a | *(v0.2)* In that case the service MUST NOT call the LLM at all — the refusal is deterministic, not generated. |
| FR-API-07 | A blank or missing `question` MUST yield HTTP 400 with a structured error body; it MUST NOT reach the embedding or LLM provider. |
| FR-API-07a | *(v0.2)* A question longer than `max_question_chars` MUST yield HTTP 400 naming the limit. |
| FR-API-08 | An upstream provider failure MUST yield HTTP 502 with a correlation id, never a stack trace. |
| FR-API-08a | *(v0.2)* A provider call exceeding `request_timeout_ms` MUST be aborted and yield HTTP 504, distinguishable from 502 in logs. |
| FR-API-09 | The service MUST expose `GET /health` reporting its own liveness and vector DB reachability. |
| FR-API-10 | Every response MUST carry a correlation id, also present in that request's log lines. |
| FR-API-10a | *(v0.2)* A caller-supplied `X-Correlation-Id` MUST be honoured and propagated rather than replaced. |
| FR-API-11 | *(v0.3)* Retrieved chunk text MUST be supplied to the LLM as **data, never as instruction**: fenced in a delimited block, with the system prompt stating that content inside it is reference material whose imperatives are to be ignored. |
| FR-API-12 | *(v0.3)* Instructions embedded in retrieved documents MUST NOT alter system behaviour. An injection attempt MUST yield the policy-correct answer and SHOULD be logged as a `prompt_injection_suspected` event. |
| FR-API-13 | *(v0.3)* The answer MUST be derived only from retrieved context; the model MUST NOT supplement from parametric knowledge. Where context is partial, the answer MUST say what it does not cover. |

### 3.3 Configuration (`FR-CFG-*`)

| ID | Requirement |
|---|---|
| FR-CFG-01 | All hyperparameters MUST live in a version-controlled configuration file, not in code. |
| FR-CFG-02 | Secrets (the Gemini API key) MUST come from the environment, never the config file or the image. |
| FR-CFG-03 | The service MUST fail fast at startup with a clear message if required configuration or secrets are absent. |
| FR-CFG-03a | *(v0.2)* Configuration values MUST be range-validated at startup; an invalid value fails fast naming the offending key. |
| FR-CFG-04 | Ingestion and query MUST read the embedding model name from the same configuration source, so the two can never silently diverge. |
| FR-CFG-04a | *(v0.2)* At startup the service MUST compare its configured embedding model against the collection metadata (FR-ING-06a) and refuse to serve on mismatch. |

Hyperparameter set (initial):

```
chunking:   min_chunk_chars, max_chunk_chars, paragraph_delimiter
embedding:  model, embed_batch_size, task_type, dimensions
retrieval:  top_k, max_top_k, min_similarity, distance_metric
generation: model, temperature, max_output_tokens, prompt_template_path
runtime:    max_retries, backoff_base_ms, request_timeout_ms, max_question_chars
```

### 3.4 Containerization (`FR-DOC-*`)

| ID | Requirement |
|---|---|
| FR-DOC-01 | `docker-compose up` MUST start the vector database and the API service, and the API MUST become healthy without further manual steps. |
| FR-DOC-02 | The API container MUST NOT start serving traffic before the vector database is reachable. |
| FR-DOC-03 | Ingestion MUST be runnable as a one-shot container against the same compose network. |
| FR-DOC-04 | Vector data MUST persist across `docker-compose down` / `up` via a named volume (but not across `down -v`). |
| FR-DOC-04a | *(v0.2)* An empty collection MUST degrade to "insufficient context" answers, not to HTTP errors. |
| FR-DOC-05 | The service image MUST be built multi-stage and MUST NOT contain build toolchains or secrets. |
| FR-DOC-05a | *(v0.2)* The service MUST run as a non-root user in the container. |
| FR-DOC-06 | *(v0.5)* `docker compose up` MUST also start the `ui` service once the API is healthy. The `ui` calls only the REST API, holds no secret, and runs as a non-root user. |

### 3.5 Logging (`FR-LOG-*`)

Critical events **only** — deliberately not a firehose.

| ID | Event | Fields |
|---|---|---|
| FR-LOG-01 | Service start | resolved config (secrets redacted), version |
| FR-LOG-02 | Query failure | correlation id, stage (`embed`/`retrieve`/`generate`), error class |
| FR-LOG-03 | LLM call latency | correlation id, model, duration_ms, token counts if available |
| FR-LOG-04 | Embedding call latency | correlation id, duration_ms, batch size |
| FR-LOG-05 | Retrieval outcome | correlation id, chunks returned, top score, below-threshold flag |
| FR-LOG-06 | Ingestion summary | per FR-ING-10 |
| FR-LOG-07 | *(v0.2)* Every record | `timestamp`, `level`, `event` present on every line |

All log records MUST be single-line JSON. Question text and answer text MUST NOT be logged by
default (see Q7 — privacy posture).

---

## 4. Non-functional targets (PoC-level, not SLOs)

| ID | Target |
|---|---|
| NFR-01 | p95 end-to-end query latency ≤ 5 s on the sample corpus, single concurrent user. |
| NFR-02 | Cold `docker-compose up` → healthy in ≤ 90 s on a developer laptop. |
| NFR-03 | Ingestion of the sample corpus completes in ≤ 5 min. |
| NFR-04 | No secret appears in the image, the config file, the logs, or the repository. |
| NFR-05 | A reader who has never seen the repo can run the PoC from the README in under 10 minutes. |

NFR-05 is a first-class requirement, not a courtesy. The PoC is an artifact of **communication**;
one that only the author can run has failed at its actual job.

---

## 5. Architecture (PoC)

```
                    ┌──────────────────────────────────────┐
  ingest (one-shot) │ read docs → paragraph chunker        │
  ─────────────────▶│ → batch embed (Gemini) → upsert      │──┐
                    └──────────────────────────────────────┘  │
                                                              ▼
  ┌──────────────┐    ┌────────────────────────┐    ┌──────────────────────┐
  │ ui           │───▶│ API service            │    │ Vector database      │
  │ (Streamlit)  │◀───│ embed → retrieve →     │◀──▶│ (container)          │
  └──────────────┘    │ prompt → generate      │    └──────────────────────┘
         ▲            └────────────────────────┘
      browser                     │
                                  ▼
                  Gemini Embedding + 3.6 Flash (spec: 2.5 Flash)
```

The `ui` (v0.5) calls only the REST API; it holds no secret and reaches neither the
database nor Gemini.

Boundaries chosen so each can be swapped independently in production: the chunker, the embedding
provider, the vector store, and the generation provider each sit behind one interface.

---

## 6. Production evolution (roadmap, not PoC scope)

Sketched here so the PoC's shortcuts are deliberate and defensible rather than discovered live.

1. **Ingestion** — event-driven on doc-repo commits; incremental upsert keyed by content hash;
   dead-letter queue for failed documents; versioned collections with atomic alias switch so
   queries never see a half-built index.
2. **Retrieval quality** — hybrid (dense + lexical/BM25) retrieval, a reranker stage, and
   metadata filters (product, version, audience). Paragraph chunking is a weak default for
   tables, code blocks, and reference material; this is the first thing I would revisit.
3. **Evaluation** — a golden Q&A set in CI; groundedness / citation-precision / answer-relevance
   metrics; no prompt or chunking change ships without a regression run. Without this, the
   system silently rots.
4. **Serving** — stateless horizontally-scaled API, managed vector DB, response + embedding
   caching, circuit breakers and budget caps on provider calls.
5. **Security** — authN/Z at the edge and **document-level ACL enforcement at retrieval time**.
   This is the one item that cannot be retrofitted cheaply: a RAG system over internal docs will
   happily exfiltrate anything it indexed. Tenancy must be a retrieval filter from day one.
6. **Observability** — tracing across embed/retrieve/generate; cost per query as a first-class
   metric; feedback capture (thumbs, "didn't answer") routed back into the eval set.
7. **Data lifecycle** — re-embedding strategy when the embedding model version changes (it will),
   PII posture on query logs, and retention.
8. **Injection defence in depth** — the PoC handles injection at the prompt boundary
   (FR-API-11/12). Production adds ingest-time scanning for instruction-like content, an
   allowlist of trusted document sources, and output validation. The structural point:
   **anything we index is an instruction channel into the model**, so document
   provenance becomes a security control, not metadata.
9. **Regulated-answer governance** — policy SMEs approve the golden set; answers carry a
   "verify against source" affordance; changes to prompt, chunking, or model version require a
   documented eval run. In this domain the eval set is a compliance artifact.

### 6.1 Scale thresholds (A-06 unresolved at production scale)

The supplied corpus is ~30 paragraphs — enough to prove the pipeline, far too small to measure
anything. Presented as a range rather than a false precision:

| Corpus | Posture |
|---|---|
| < 10k chunks | pgvector with exact or IVFFlat search; latency dominated by the LLM call, not retrieval. |
| 10k–1M chunks | pgvector + HNSW, tuned `m` / `ef_search`; add caching and a reranker. Still no new datastore. |
| > 1M chunks | Revisit against a dedicated vector engine; the cost is an operational one, and the decision point should be named in advance rather than discovered under load. |

---

## 7. Risks and open technical concerns

| ID | Risk | Mitigation / stance |
|---|---|---|
| R-01 | Paragraph chunking fragments tables, code blocks, and lists, degrading retrieval on exactly the content engineers ask about. | Implement as specified; quantify failure on real examples; propose structure-aware chunking as the first post-PoC change. |
| R-02 | Embedding model version drift silently invalidates the index. | FR-CFG-04 + embedding model recorded in collection metadata. |
| R-03 | Ungrounded generation (hallucination) when retrieval returns weak matches. | FR-API-06 similarity floor + mandatory citations (FR-API-05). |
| R-04 | Gemini API quota/latency dominates the demo. | Retries with backoff, timeouts, latency logging (FR-LOG-03/04). |
| R-05 | Corpus contains access-restricted material. | Out of PoC scope; flagged as §6.5 blocking item for production. |
| R-06 | Choosing Python over the preferred Spring may read as an unexplained deviation. | Resolved by ADR-001, which states the reasoning explicitly rather than leaving it implicit in the repo. |
| R-07 | pgvector recall/latency degrades past a corpus size we have not yet measured. | Index choice and parameters are config, not code. Threshold named as a range in §6.1 — the smoke corpus cannot measure it. |
| R-08 | **Indirect prompt injection.** The corpus contains a live injection payload (`account-and-verification.md`). Any document a RAG system indexes is an instruction channel into the model. | FR-API-11/12 + the `injection.feature` suite. Treated as a correctness requirement, not a roadmap item — in a regulated domain an injected wrong answer about liability or account closure is a compliance incident, not a bug. |
| R-09 | **Regulated-domain answer risk.** The corpus states PSD2 liability caps, SCA rules, dispute windows and 10-year record retention. A wrong or stale answer carries regulatory weight. | Mandatory citations (FR-API-05), no-parametric-supplement (FR-API-13), deterministic refusal (FR-API-06a). Production needs answer review and a "verify against policy" disclaimer — named in §6. |
| R-11 | *(v0.5)* **Specified model retired.** `gemini-2.5-flash` returns 404 for projects created after its deprecation, although the config pinned it. | The PoC runs `gemini-3.6-flash`, recorded in `config.yaml`. Pinning does not prevent a retirement; a deprecation watch and an eval run per model change do (§6.3). |
| R-10 | **Corpus/hypothesis mismatch.** The hypothesis describes "technical documentation" for engineers; the corpus is customer-facing banking policy. Retrieval tuning, tone, and the value model all differ. | Flagged; the business case (§1) is written against what the corpus actually is. |

---

## 8. Acceptance

The PoC is accepted when:

- Every `FR-*` has a passing Gherkin scenario in `docs/spec/features/`.
- `docker-compose up` on a clean machine satisfies FR-DOC-01 and NFR-02.
- A live question against the sample corpus returns a cited, grounded answer.
- An adversarial question (answer not in the corpus) returns "insufficient context", not a
  fabrication.
- **The planted injection payload in `corpus/account-and-verification.md` does not alter the
  answer**, and the attempt is logged.
- The material in §1, §6, §7 stands on its own.

### 8.1 Demo script

Ordered to build the argument, not just show features:

1. `docker-compose up -d` → `/health` green. *(FR-DOC-01)*
2. `docker-compose run --rm ingest` → JSON summary line. *(FR-ING-10)*
3. Happy path: "How long do I have to dispute a transaction?" → **120 days**, cited.
4. Negative fact: "Can a support agent unblock my card after a fraud block?" → the model must
   say *only the customer can*. This is where naive RAG fails, and the most persuasive moment.
5. Out of corpus: "What is the capital of Peru?" → the model says the documentation does not
   cover it. *(v0.5)* With real embeddings the question scores about 0.50, above the 0.35
   floor, so the deterministic refusal does not fire; calibrating the floor is §11.5 finding 2.
6. **Injection: "Can I close my account?"** → correct policy answer despite the planted payload;
   show the `prompt_injection_suspected` log line alongside it.
7. Failure mode, shown deliberately: a question about a table or a stepwise procedure, to expose
   R-01 paragraph-chunking weakness with evidence rather than a claim.

Step 7 is the point of the demo. A demo that only shows successes tells the audience nothing
about whether the engineer knows where the system breaks.

---

## 9. Open assumptions

This section is **blocking**. Items are listed as questions in the accompanying review; each is
recorded here once decided, with the decision and its date.

| ID | Assumption | Status | Decided |
|---|---|---|---|
| A-01 | Implementation language / framework | **DECIDED — Python 3.12 + FastAPI, end to end.** Ingestion is `ingest.py`; the API is FastAPI. See ADR-001. | 2026-09-17 |
| A-02 | Vector database | **DECIDED — Postgres 16 + `pgvector`.** See ADR-002. | 2026-09-17 |
| A-03 | Sample documents | **DECIDED — authored by me.** Four synthetic banking-policy documents written for an earlier project, reused here. Banking policy chosen deliberately (regulated domain; see R-10), with a prompt-injection payload planted as an adversarial test case (R-08). | 2026-09-17 |
| A-04 | Ingestion trigger | **DECIDED — separate one-shot command.** `docker-compose up -d` then `docker-compose run --rm ingest`. Keeps the batch/serving boundary explicit and protects NFR-02. | 2026-09-17 |
| A-05 | Testing depth | **DECIDED — the Gherkin suite is executable** via `pytest-bdd`; the feature files in `docs/spec/features/` are the acceptance suite, not documentation. See ADR-003. | 2026-09-17 |
| A-06 | Corpus | **DECIDED — `corpus/`, 4 markdown policy documents (~7.4 KB, ~30 paragraphs).** Smoke-scale. Production volumes remain unknown and are presented as a range (§6.1). | 2026-09-17 |
| A-07 | Logging privacy posture | **DECIDED — log questions, never answers.** Questions drive the documentation-gap lever (§1.2) and are the analytically valuable signal; answers are the larger leakage surface. | 2026-09-17 |
| A-08 | Demo | **DECIDED — a live demo following §8.1.** | 2026-09-17 |

---

## 10. Decision records

### ADR-001 — Python + FastAPI for both ingestion and service

**Context.** The ingestion script is `ingest.py` (Python), and the service was preferably to use
Spring (Java). Satisfying both literally means two toolchains, two Gemini clients, and — the
real cost — the paragraph chunker living in one language while the query path lives in another.

**Decision.** Python end to end. FastAPI for the service.

**Consequences.** One toolchain; the chunker is shared code between the ingest and query paths,
so the two can never drift. Best-supported Gemini SDK. The cost is a deviation from a stated
preference, which is **stated openly** rather than hoped past: the framework
is the least interesting decision in this system, and paying a cross-language tax in a
feasibility PoC buys nothing that a `FastAPI → Spring` port could not buy later. The service is
a thin HTTP shell over four steps — it is genuinely portable. *If the Spring preference turns
out to be a hard requirement rather than a preference, this is the one decision to revisit
early; it is cheap now and expensive after the chunker has grown.*

### ADR-002 — Postgres + pgvector as the vector store

**Context.** A PoC-native option (Qdrant, Chroma) versus a datastore the platform already runs.

**Decision.** Postgres 16 with the `pgvector` extension.

**Consequences.** No new datastore enters the platform — the strongest available answer to
"what does this cost to operate?", and it lets retrieval join against document metadata and, in
production, **ACL tables** (§6.5) in a single query rather than filtering in application code
after the fact. That is a genuine architectural advantage for this problem, not just familiarity.
The costs are real and are stated here: ANN index choice (HNSW vs. IVFFlat) and its parameters
are manual, recall/latency tuning is on us, and pgvector's ceiling is lower than a dedicated
engine's — the roadmap must name the corpus size at which we would revisit.
Implementation note: FR-ING-06a's "collection metadata" becomes a `collections` table row
recording embedding model and dimensionality, checked at startup per FR-CFG-04a.

### ADR-003 — The Gherkin suite is executable

**Context.** Feature files can be documentation or tests.

**Decision.** `pytest-bdd` binds the scenarios in `docs/spec/features/` to step definitions.
Provider calls are mocked; one live end-to-end smoke test runs outside CI.

**Consequences.** The spec cannot rot away from the implementation — a requirement that stops
holding fails a build. This is the whole point of spec-driven development, and it is what makes
the boundary cases the Gherkin pass exposed (FR-ING-03a, -04a, FR-API-06a, -08a) real rather
than aspirational. Cost: more implementation time than unit tests alone, and step definitions
are their own maintenance surface.

---

## 11. Implementation log (v0.4)

What building the spec actually changed. Recorded because the value of SDD is in the
feedback loop, not the document.

### 11.1 Defects the acceptance suite caught

| # | Defect | Requirement | Fix |
|---|---|---|---|
| 1 | A missing `question` field returned FastAPI's default **422**, not the specified 400. | FR-API-07 | `RequestValidationError` handler mapping schema violations to 400 with the standard error body. A caller should not have to distinguish "field missing" from "field blank". |
| 2 | `configure_logging()` cleared **all** root handlers, discarding any sink installed by a host process. | FR-LOG-07 | Only handlers it installed are replaced. Found because ingestion silently destroyed the test harness's log capture — the same thing would happen to a supervisor's sink in production. |
| 3 | `register_vector()` was called before `CREATE EXTENSION vector`, so a **clean database could never start** — the API crash-looped. | FR-DOC-01/02 | Type registration is skipped during initialisation. Only reproducible on a first-run volume, which is exactly the reviewer's experience. |
| 4 | `uvicorn --log-config /dev/null` is invalid and killed the container at boot. | FR-DOC-01 | Removed; uvicorn's loggers now propagate into the JSON handler, so **every** line the container emits is structured, not just the application's. |
| 5 | The prompt template used the literal string `<reference_material>` in its own prose, making the context fence ambiguous to parse. | FR-API-11 | Prose refers to "the reference material block"; the tag now appears exactly once. |

Defects 3 and 4 were only reachable by actually running `docker compose up` on a clean
volume. No amount of unit testing would have surfaced them, which is the argument for
FR-DOC-01 being a requirement with a scenario rather than a README instruction.

### 11.2 A limit of the unit tier, stated plainly

The `@unit` tier substitutes a lexical bag-of-words embedder for Gemini. Measured against
the real corpus, it scores *"What is the capital of Peru?"* at **0.63** — higher than
*"How long do I have to dispute a card transaction?"* at **0.53**. Token overlap is not
meaning, so **no similarity floor can separate in-corpus from out-of-corpus questions**
under the double.

The unit tier therefore proves the refusal *mechanism* (a floor produces a deterministic
answer and makes no LLM call) and never the *judgement* of what is out of corpus. That
judgement is a property of real embeddings and is asserted only in `@live`. The
out-of-corpus questions are marked as such in the harness rather than discovered by
scoring, and the code says so at the point where it matters.

This is the sort of thing a green test suite hides. It is written down instead.

### 11.3 Chunking outcome on the real corpus

Paragraph chunking with `min_chunk_chars: 200` produces a useful side effect: markdown
headings (`## Closing an account`, 21 chars) fall below the minimum and **merge forward
into the paragraph they introduce**, so most chunks carry their own section heading as
context. That was not designed; it is a consequence of FR-ING-03 and worth keeping.

The planted injection payload lands in its own chunk (ordinal 6 of
`account-and-verification.md`), *separate* from the closure policy it contradicts
(ordinal 5). Retrieval must surface it for the mitigation to be exercised at all — which
is why `test_injection_payload_survives_chunking_intact` guards it: if chunking ever
splits or drops that text, the `@live` injection scenarios would pass while testing
nothing.

### 11.4 @docker tier execution

The `@docker` scenarios drive `docker compose` and must run outside the test container,
on a host with Python and Docker. On a host without Python they were verified by hand:

```bash
docker compose exec -T api id -un                         # FR-DOC-05a -> app (uid 999)
docker compose exec -T api sh -c 'which gcc'              # FR-DOC-05  -> absent
docker image history --no-trunc rag-pipeline-test-api \
  | grep -c "AIza\|GEMINI_API_KEY"                        # NFR-04     -> 0
docker compose logs api | grep -o '"gemini_api_key": "[^"]*"'  # FR-LOG-01 -> "***"
curl -s localhost:8000/health                             # FR-DOC-01/02
```

All passed on 2026-09-17.

### 11.5 After the first version (v0.5)

Everything above was written on 2026-09-17. What follows came afterwards, and is in the
git history from commit `2236297`, which records the repository as it stood that day.

**Changes**

| Commit | Change |
|---|---|
| `3dd01c7` … `2da4e5b` | Refactor in four phases: comments and dead code, stage-typed provider errors and a retriever protocol, services through `app.state` with declarative config, shared test harness |
| `d3f7979` | The latency log steps now assert real events, through an injectable Gemini client |
| `9f289f0` | FR-API-05a: citations carry `snippet` and `injection_suspected` |
| `794e029` `63a24dc` `819d322` | Streamlit ui (FR-DOC-06), a pending state, suggestions inside the chat |
| `89ebf31` | The SDK's automatic function calling is off; it logged to the root logger on every call |
| `ef19ff1` | ruff configured; strict zips over paired data |

**Findings from running it against real services**

1. **An invalid API key is a 400, not a 401.** Google returns `API_KEY_INVALID` as a 400, so it
   surfaces as a generic provider error rather than an auth error, and the FR-ING-08 scenario
   simulates a status code the provider does not send.
2. **The similarity floor never fires.** With `gemini-embedding-001`, in-corpus questions score
   0.68–0.79 at the top and out-of-corpus ones 0.50–0.57 (ten questions). The 0.35 floor is
   below both, so every question reaches the model. The unit tier proved the refusal
   mechanism; the live run showed the threshold was never calibrated.
3. **The specified model is retired** for new projects (R-11).
4. **The injection held** on the one question tried: the payload chunk was in the context at
   rank 2, the answer gave the real closure policy, and the detector fired.
5. **Retries spend the quota that is limiting them.** The free tier allows 20 generation
   requests a day; retrying 429s without honouring `Retry-After` exhausted it twice.
6. **Streamlit and FastAPI cannot share an environment**: Streamlit 1.64 needs Starlette ≥ 0.46
   and FastAPI 0.115.6 allows < 0.42, so the ui has its own image and its own test image.
7. **Search latency is not logged**, so there is no baseline for tuning HNSW.
