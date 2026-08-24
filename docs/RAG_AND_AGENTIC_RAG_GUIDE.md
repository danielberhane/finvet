# RAG and Agentic RAG — A Complete Guide, Grounded in FinVet

Every claim in this document was verified against the code, the database, or a live
experiment on 2026-08-24. Line references point at
`finvet-v2.0.9/.claude/worktrees/rag-fix`.

---

# PART I — FOUNDATIONS

## 1. The problem RAG solves

A language model knows only what was in its training data, frozen at a cutoff date. Ask it
"what were Amazon's AWS segment net sales in fiscal 2025?" and one of three things happens:

1. It doesn't know and says so — useless but honest.
2. It doesn't know and **invents a plausible number** — actively dangerous.
3. It half-remembers something from training — worse still, because it sounds confident.

For a verification product, all three are unacceptable. You cannot check a claim against a
model's memory.

**Retrieval-Augmented Generation (RAG)** is the fix: before the model answers, go *fetch* the
relevant source text and put it in front of the model. The model's job changes from *recall*
to *reading comprehension* — a far more reliable operation.

Three properties follow, and they are the reason RAG matters in regulated domains:

- **Currency.** The corpus can be updated without retraining anything.
- **Attribution.** You know which passage the answer came from.
- **Falsifiability.** A reviewer can open the source and check.

FinVet needs all three. A verdict that cannot be traced to a filing passage is an opinion.

## 2. Why not just paste the whole filing into the prompt?

A 10-K is large. Measured on your own corpus with `cl100k_base`:

```
AAPL_10-K_2025-09-27.html   1.5 MB   49,093 tokens
AMZN_10-K_2025-12-31.html   2.0 MB   69,444 tokens
MSFT_10-K_2025-06-30.html   8.2 MB   77,123 tokens
```

Fifty to eighty thousand tokens of extracted text per filing. Three problems:

- **Cost and latency** scale with input length on every single query.
- **Attention dilutes.** Models reliably lose information buried in the middle of very long
  inputs.
- **You lose attribution.** If the whole document is context, "which passage supported this?"
  has no answer.

So you retrieve *passages*, not documents. Which raises the question RAG exists to answer:
**how do you find the right passage?**

## 3. What an embedding actually is

Keyword search fails on the obvious case: the filing says "Amazon Web Services," you searched
"AWS." Zero matches, despite being the same thing.

An **embedding** is a list of numbers representing *meaning*. A neural network reads a passage
and outputs a fixed-length vector — for FinVet, **768 numbers**. Passages with similar meaning
land near each other in that 768-dimensional space, regardless of shared vocabulary.

A two-dimensional intuition: if dimension 1 were "about money" and dimension 2 "about risk,"
then a revenue table sits at (0.9, 0.1) and a cybersecurity disclosure at (0.1, 0.9). Real
embeddings use 768 dimensions and nobody can say what any single one means — but the geometry
is the same. **Closeness equals similarity of meaning.**

## 4. Measuring closeness: cosine similarity

FinVet uses **cosine similarity** — the angle between two vectors, ignoring their length.

```
cos(A,B) = (A · B) / (|A| × |B|)      1.0 = identical direction, 0 = unrelated
```

**Why angle and not distance?** Because vector *magnitude* tracks passage length. A two-line
footnote and a three-page discussion of the same topic should score as similar; Euclidean
distance would separate them because one vector is longer. Cosine ignores that.

Measured live on your corpus:

```
query "AWS segment net sales" vs the AWS revenue sentence      cos = 0.857
query "AWS segment net sales" vs a data-security risk factor   cos = 0.417
```

The retrieval signal is real and large.

---

# PART II — THE INGESTION PIPELINE

Ingestion runs offline, once per filing. Four stages: **parse → chunk → embed → store.**

## 5. Stage 1 — Parsing HTML into labelled sections

Input: `finvet-v2.0.9/data/filings/AMZN/AMZN_10-K_2025-12-31.html` — the document as filed.
(The filings live in the main tree; the worktree shares the same Postgres corpus.)

`rag/parser.py` exploits the fact that every 10-K follows an SEC-mandated structure of
numbered Items. The heading regex (`parser.py:82`):

```python
_ITEM_PATTERN = re.compile(
    r"^(?:ITEM|Item)\s+(\d+[A-Ca-c]?)\.?\s+(.*)",
    re.IGNORECASE,
)
```

Note the trailing `(.*)` — it requires a **title** after the item number. That is why
Workiva-generated filings fail to parse: they put the number and title in separate table
cells (`<td>Item 1A.</td><td>Risk Factors</td>`), so a title-less "Item 1A." never matches.

Text between consecutive headings is captured and labelled:

`business` · `risk_factors` · `legal_proceedings` · `mda` · `market_risk` ·
`financial_statements_and_notes` · `cybersecurity` · `controls_and_procedures` ·
`executive_compensation`

**Why label sections at all?** So retrieval can be narrowed *before* similarity is computed.
In the live AWS run the agent set `section="financial_statements_and_notes"`, excluding ~30
chunks of risk-factor boilerplate. That filter did more work than the embeddings did — see §21.

> **Best practice.** Exploit document structure when it exists. Regulatory filings, contracts
> and RFCs all have mandated skeletons; a structural parser beats a generic splitter every
> time. Reach for fixed-size splitting only when the corpus has no structure.

> **Real failure this caused.** Section boundaries originally never closed — the loop broke
> only when a yielded element *was* the next heading tag, but it yielded leaf elements while
> headings were containers, so identity never matched. Every section ran from its heading to
> the end of the document, so "Legal Proceedings" (normally about a page) came back tens of
> thousands of characters long and every section terminated on the signature page. Section
> filters became meaningless, and the agent could be handed the signature page as evidence
> about risk factors.
>
> *(Pre-fix sizes are quoted from the diagnosis that produced the fix; the corpus has since
> been re-ingested, so they cannot be re-measured. The fix is verifiable: AAPL's 10-K
> `legal_proceedings` is now **3 chunks**.)*
>
> It survived for months because **`src/finvet/rag/` had no test file.**

## 6. Stage 2 — Chunking

```python
def chunk_sections(                    # parser.py:206
    sections: list[Section],
    max_tokens: int = 500,
    overlap_tokens: int = 100,
) -> list[Chunk]:
```

`ingest_filing` calls it with the defaults (`service.py:148`), so 500/100 are the values
actually in use.

### Why chunk

An embedding compresses whatever you give it into one fixed-size vector. Feed it a 30,000-word
section and every distinct idea inside averages into mush — the vector ends up near nothing in
particular. Retrieval should also return something a human can *read*, not a chapter.

### Why 500 tokens

Roughly 350–400 words — about a page. Large enough to contain a complete thought (a segment
table, one risk factor), small enough that the vector still points somewhere specific.

**The trade-off is fundamental:**

| Chunk size | Precision | Recall | Failure mode |
|---|---|---|---|
| Small (~100 tok) | high | low | context cut off mid-idea |
| **Medium (~500)** | **balanced** | **balanced** | FinVet's choice |
| Large (~2000) | low | high | one vector, many topics, diluted |

### Why 100 tokens of overlap — the subtlest parameter

Without overlap a boundary can slice a fact in half:

```
(illustrative, not a quote from a filing)
chunk 12 ends:   "...AWS segment net sales for 2025 were"
chunk 13 begins: "$128,725 million..."
```

Neither chunk answers the question. Overlap repeats the trailing ~100 tokens at the start of
the next chunk, so any fact spanning a seam survives intact somewhere. You pay roughly 20%
storage duplication to stop losing facts at boundaries. That is a good trade.

### How the splitting actually works (`parser.py:236-275`)

1. Split section text into **paragraphs**.
2. Accumulate paragraphs until adding the next would exceed 500 tokens.
3. Emit the chunk; seed the next one with `_get_overlap(...)` — the trailing parts fitting in
   100 tokens.
4. **If a single paragraph alone exceeds 500 tokens**, flush the buffer, then split that
   paragraph by sentences on `(?<=[.!?])\s+` and pack sentences instead.

Step 4 is the mark of a careful implementation. A naive splitter either emits a 5,000-token
chunk or hard-cuts mid-sentence. This degrades gracefully: paragraphs first, sentences only
when forced.

> **Best practice.** Split on semantic boundaries in descending order — section, paragraph,
> sentence, and only then characters. Never split on a fixed character count as the first
> resort.

### One inconsistency worth knowing

`_count_tokens` uses **`cl100k_base`** — OpenAI's tokenizer (`parser.py:75`). But embedding is
now done by **nomic-embed-text**, which uses a different vocabulary. So "500 tokens" is
measured with the wrong ruler.

It is harmless in practice: the counts are close enough, the target is a soft one, and
nomic's 2048-token context leaves ample headroom. But it is a leftover from the OpenAI era and
worth a comment so the next reader doesn't assume it is exact.

## 7. Stage 3 — Embedding

```python
EMBEDDING_MODEL  = "nomic-embed-text"    # constants.py:52
EMBEDDING_DIMS   = 768                   # constants.py:53
EMBED_BATCH_SIZE = 50                    # constants.py:54
```

### The model, from Ollama's own metadata

```
architecture : nomic-bert       parameters : 137M
layers       : 12               context    : 2048 tokens
embedding    : 768 dimensions   quantization: F16
```

**137 million parameters — tiny**, versus hundreds of billions for the model writing verdicts.
That is the point: an embedding model never generates text, it only *measures meaning*.

It is a **BERT-style encoder**: all 12 layers read the entire passage at once, in both
directions, unlike a chat model reading left-to-right predicting the next token. Per-token
outputs are pooled into a single 768-number vector.

### How the code calls it (`service.py:33-52`)

```python
response = httpx.post(
    f"{settings.ollama_url.rstrip('/')}/api/embed",
    json={"model": EMBEDDING_MODEL, "input": texts},
    timeout=settings.embedding_timeout_s,          # 60s
)
response.raise_for_status()
embeddings = response.json().get("embeddings") or []
if len(embeddings) != len(texts):
    raise RuntimeError(...)                        # count guard
if embeddings and len(embeddings[0]) != EMBEDDING_DIMS:
    raise RuntimeError(...)                        # dimension guard
```

**Both guards earn their place.** The count guard prevents silent misalignment — 50 chunks in,
49 vectors back, and every chunk after the gap is paired with the wrong embedding, corrupting
the store invisibly. The dimension guard catches exactly the split-brain state that occurred
earlier today: the corpus was ingested at 768 dims by Ollama while the query path still called
OpenAI at 1536.

> **Best practice.** Assert the shape of anything crossing a process boundary. Embedding bugs
> are silent — you get *worse results*, never an exception.

### Local versus hosted

| | OpenAI `text-embedding-3-small` | `nomic-embed-text` (current) |
|---|---|---|
| Dimensions | 1536 | 768 |
| Runs | remote API | **local Ollama** |
| Cost | ~$0.02/M tokens | **$0.00** |
| Fails when | credits exhausted | Ollama not running |

The switch was forced: the entire retrieval layer sat dead for months because of an exhausted
billing balance. A local model has no quota to exhaust.

**The cost of switching models is a full re-embed.** A 1536-dim vector cannot go into a
`vector(768)` column. Choose a dimension deliberately.

### The task-prefix question — tested, unresolved

`nomic-embed-text` is trained with instruction prefixes: `search_document:` for indexed text,
`search_query:` for queries. **FinVet uses neither** — ingest (`service.py:160`) and query
(`service.py:219`) call the same unprefixed function.

I measured rather than assumed. On one doc/query pair, prefixes raised similarity to the
relevant passage (0.857 → 0.888) but raised the distractor as well (0.417 → 0.490), so
*separation* was marginally worse. Ranking 40 real Amazon chunks both ways produced different
top-5 sets, with neither clearly better.

**Verdict: unresolved, and worth measuring properly.** It is a good first experiment once a
retrieval gold set exists.

## 8. Stage 4 — Storage and indexing

**The vector store is PostgreSQL with the pgvector extension.** No Pinecone, Weaviate or
Chroma. Verified schema:

```
 chunk_text    | text
 embedding     | vector(768)
 tsv           | tsvector  GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
 ticker, cik, filing_type, filing_date, period_end, section, section_title,
 chunk_index, token_count
```

**Why one database rather than a dedicated vector store?** Filtering, joins, transactions and
backups come free. `WHERE ticker='AMZN' AND section='financial_statements_and_notes'` is
ordinary SQL. A standalone vector DB would need that metadata mirrored and kept in sync — a
second source of truth, and a second thing to get wrong.

**`tsv` is a generated column.** Postgres derives it from `chunk_text` automatically, so the
keyword index can never drift out of sync with the text. Declarative beats a trigger.

### The indexes

```
idx_filing_chunks_embedding      hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)
idx_filing_chunks_tsv            gin (tsv)
idx_filing_chunks_ticker_period  btree (ticker, filing_type, period_end)
ix_filing_chunks_section         btree (section)
ix_filing_chunks_ticker          btree (ticker)
```

**HNSW — Hierarchical Navigable Small World.** Comparing a query against all 988 vectors is
trivial; against 10 million it is not. HNSW builds a layered graph where each vector links to
near neighbours, so search *walks* toward the answer rather than scanning everything. It is
**approximate** — occasionally missing a true nearest neighbour — in exchange for large speed
gains. `m=16` is links per node; `ef_construction=64` is effort at build time. Higher means
better recall, slower builds, more memory.

**GIN** is the classic inverted index — word → chunks containing it — powering keyword search.

**The btree indexes matter more than they look.** They make the metadata filters cheap, and
those filters are doing heavy lifting in FinVet's retrieval quality.

### Current corpus (measured)

988 chunks · AAPL 111, AMZN 214, MSFT 263, NVDA 194, TSLA 206 · all 2025-vintage filings.

---

# PART III — RETRIEVAL

## 9. Step 1 — Embed the query

```python
query_embedding = _embed_single(query)        # service.py:219
```

The query passes through the **same model** as the documents. This is non-negotiable: two
embedding models produce incompatible coordinate systems, and similarity between them is
noise.

> **This was an architectural defect, now fixed.** The call originally ran unguarded before
> both arms, so an embedder outage raised and took down the keyword arm too — which needs no
> embedding at all. A "hybrid" retriever that cannot survive losing one of its two inputs is
> not really hybrid. It is now wrapped:
>
> ```python
> try:
>     query_embedding = _embed_single(query)
> except Exception as e:
>     logger.warning(f"Embedding unavailable ({type(e).__name__}) — falling back to keyword-only")
>     query_embedding = None
> ```
>
> Losing the embedder now costs you the semantic arm, not the whole retriever.

## 10. Step 2 — Vector search (semantic)

```sql
SELECT id, chunk_text, section, section_title, filing_type, period_end, ticker,
       1 - (embedding <=> CAST(:query_vec AS vector)) AS vec_score
FROM filing_chunks
WHERE ticker = :ticker AND section = :section
ORDER BY embedding <=> CAST(:query_vec AS vector)
LIMIT 20
```

`<=>` is pgvector's cosine-distance operator; `1 - distance` converts to similarity. The
`ORDER BY ... LIMIT` is what lets Postgres use the HNSW index.

**`CAST(...)` rather than `::vector` is deliberate** — `::` collides with SQLAlchemy's
`:param` binding syntax. This is gotcha #4 in `CLAUDE.md`.

**Catches:** meaning. Finds "Amazon Web Services revenue" for the query "AWS segment net sales."

> **Defect: no distance threshold.** `LIMIT 20` always returns 20 rows, however unrelated.
> Query a ticker with nothing ingested and you still get confident-looking chunks. In a
> verification product that is a path to fabricated evidence. A minimum-similarity floor
> belongs here.

## 11. Step 3 — Keyword search (lexical)

```sql
SELECT ..., ts_rank(tsv, plainto_tsquery('english', :query_text)) AS kw_score
FROM filing_chunks
WHERE ... AND tsv @@ plainto_tsquery('english', :query_text)
ORDER BY ts_rank(...) DESC
LIMIT 20
```

`plainto_tsquery` lowercases, strips stopwords and stems to roots ("sales" → "sale"). `@@`
tests match; `ts_rank` scores by term frequency and rarity — the same family of ideas as BM25.

**Catches:** exact tokens. Tickers, defined terms, accession numbers, unusual figures — the
things embeddings blur precisely because they generalise.

## 12. Step 4 — Reciprocal Rank Fusion

**The problem.** Cosine similarity runs 0–1. `ts_rank` runs on a different, unbounded scale.
Adding them is meaningless — summing apples and metres. Normalising them requires distribution
assumptions that do not hold across queries.

**The solution: throw away the scores and use only the positions.**

```python
vec_rank  = vec_ranks.get(chunk_id, RRF_ABSENT_RANK)     # 1000 if absent
kw_rank   = kw_ranks.get(chunk_id, RRF_ABSENT_RANK)
rrf_score = 1.0 / (RRF_K + vec_rank) + 1.0 / (RRF_K + kw_rank)     # RRF_K = 60
```

**Why the constant 60?** It flattens the curve. Without it, rank 1 (`1/1`) would be ten times
rank 10 (`1/10`), letting a single arm dominate. With 60, rank 1 gives `1/61 = 0.0164` and
rank 10 gives `1/70 = 0.0143` — a gentle slope. **The consequence is the entire point of
hybrid search: a chunk ranked moderately well by *both* arms beats one ranked first by only
one.** Agreement between independent methods is the signal.

**Absent = rank 1000** contributes `1/1060 ≈ 0.0009` — near zero, so missing from one list is
a handicap, not a disqualification.

Sort descending, return `top_k` (default 5).

> **Honesty note.** `RRF_K = 60` is the original paper's default, adopted because everyone
> adopts it. Whether 60 is right *for financial filings* is unmeasured. This is exactly what a
> Ragas context-precision evaluation would settle.

## 13. What the pipeline looks like end to end

```
FILING HTML
   │  parse_filing_html()          → sections labelled by Item heading
   ▼
SECTIONS
   │  chunk_sections(500, 100)     → overlapping passages, paragraph-first
   ▼
CHUNKS
   │  _embed_texts() batch 50      → nomic-embed-text, 768 dims, local
   ▼
VECTORS ──▶ filing_chunks (pgvector 768 + generated tsvector)
                     │
   QUERY ────────────┤
     │  _embed_single()
     ├──▶ vector arm : cosine, HNSW, LIMIT 20
     ├──▶ keyword arm: ts_rank, GIN,  LIMIT 20
     └──▶ RRF fusion (k=60) ──▶ top_k = 5 chunks
```

---

# PART IV — THE AGENTIC LAYER

## 14. Classic RAG versus Agentic RAG

**Classic RAG is a conveyor belt.** Question in → always search → staple results to the prompt
→ generate. One search, always, before generation. The model decides nothing about retrieval.

**Agentic RAG is a researcher with a library card.** The model decides *whether* it needs the
library, *what* to look up, and may return several times with better queries. Retrieval is a
**tool it may reach for**, not a station every request passes through.

**FinVet is agentic RAG.** Three code facts establish it:

1. `search_filing_text` is decorated `@tool` — it is an option, not a stage.
2. **No retrieval node exists in the graph.** All twelve nodes were checked; none performs
   retrieval. Nothing forces a search.
3. Every retrieval parameter — `query`, `ticker`, `section`, `filing_type`, `top_k` — is
   chosen by the model at runtime.

Measured consequence: across 496 benchmark case-runs, DeepSeek called `search_filing_text`
**14 times**. Genuine discretion, not a fixed step.

**FinVet also contains classic RAG, for contrast.** `/memory-check` runs *before* the pipeline,
always, embedding the claim and searching past verifications with no model involvement. Same
repo, both patterns, different jobs.

## 15. How tool calling actually works

Newcomers often imagine the model "executing code." It does not. The mechanics:

**1. Tool definitions become part of the prompt.** LangChain converts each `@tool` function's
signature and docstring into a JSON schema sent with the request. The model sees names,
parameters, types and descriptions.

**2. The model emits a structured request, not a result.** Instead of prose it returns:

```json
{"name": "search_filing_text",
 "args": {"query": "AWS segment net sales", "ticker": "AMZN",
          "section": "financial_statements_and_notes", "filing_type": "10-K"}}
```

**3. The framework executes it.** LangGraph matches the name to the Python function, calls it
with those arguments, and captures the return value.

**4. The result is appended as a `ToolMessage`** and the whole conversation is sent back to the
model, which now reasons over what it received.

**This is why the docstring is load-bearing.** It is not documentation for humans — it is the
*only* description the model has when deciding whether the tool applies. FinVet's filing-search
docstring says: use for segment breakdowns, risk factors, MD&A; **do NOT use for standard GAAP
line items**. That sentence is what routed the AWS claim correctly.

> **Best practice.** Write tool docstrings as instructions to a competent colleague who cannot
> see your code. State what it is for, what it is *not* for, and what the arguments mean.
> Negative guidance ("do not use for X") is as valuable as positive.

## 16. The ReAct loop

**ReAct = Reason + Act**, iterated:

```
Thought → Action (tool call) → Observation (result) → Thought → ... → Answer
```

FinVet builds it with `create_react_agent` (`base.py:110`), bounded by:

```python
AGENT_MAX_ITERATIONS = 5                                    # constants.py:40
config={"recursion_limit": self.max_iterations * 2 + 1}     # = 11
```

**Why `× 2 + 1`?** Each iteration costs two graph steps — the model turn and the tool turn —
plus one final answer turn.

> **This bound is not theoretical.** In the agentic benchmark, **29 of DeepSeek's 34 failures
> were `Recursion limit of 11 reached`** — the agent exhausted its budget and returned
> NOT_ENOUGH_INFO with zero tool calls recorded. Worse, the failure taxonomy *mislabelled*
> them: `classify_failure` greps the reasoning text for "recursion", but
> `compose_failure_reasoning` rewrites the raw error into prose containing neither "recursion"
> nor "iteration limit". So the largest failure bucket was reported as a **retrieval** problem
> when it is a **control-flow** problem. Opposite fixes.

## 17. Anti-anchoring: a separate verdict call

After the loop, FinVet does **not** ask the agent for its verdict. It makes a *separate* LLM
call for structured extraction, deliberately excluding the agent's own free-text conclusion
(`base.py:271-289`).

**Why:** a model shown its own prior conclusion tends to ratify it. Removing that text forces
the verdict to be derived from the tool observations.

> **Best practice.** Separate "gather evidence" from "decide." They are different tasks with
> different failure modes, and merging them lets a confident narrative override the evidence.

## 18. The deterministic override — the layer that makes this trustworthy

LLMs are unreliable at numeric comparison; a frontier model will assert `0.98% > 1%`. When both
the claimed and retrieved values exist, FinVet **recomputes the comparison in Python** with
source-appropriate tolerances and overrules the model when they disagree. Both verdicts are
kept.

**This applies identically on the RAG path.** In the live AWS run, RAG supplied the number
($128,725M) and Python decided the verdict (10.66% > 1.5% → REFUTES). **RAG changes where the
number comes from, never how it is judged.**

> **Best practice — the general principle.** Use a deterministic check wherever the domain
> allows one; reserve the model for what only a model can do. Here the model reads prose and
> extracts a figure; arithmetic belongs to arithmetic.

## 19. The three agents and every tool

Agents are defined by their tools; the parser's `claim_type` picks exactly one.

**SEC agent** (`claim_type="sec"`) — 8 tools:

| Tool | Purpose |
|---|---|
| `get_company_info` | ticker → CIK, name, fiscal year end |
| `get_recent_filings` | list 10-K / 10-Q filings for a CIK |
| `get_income_statement` | XBRL revenue, income, EPS |
| `get_balance_sheet` | XBRL assets, liabilities, equity |
| `get_cash_flow` | XBRL operating / investing / financing |
| **`search_filing_text`** | **RAG — filing prose** |
| `corroborate_with_news` | **A2A — asks the News agent to confirm** |
| `search_past_verifications` | episodic memory |

**Market agent** (`claim_type="market"`): `get_stock_quote`, `get_daily_prices`,
`get_company_overview`, `get_earnings`, `search_past_verifications`.

**News agent** (`claim_type="news"`): `search_financial_news`, `verify_news_source`,
`get_macro_indicator`, `search_past_verifications`.

**RAG lives only in the SEC agent.** Market and News have no retrieval over documents.

**A2A (agent-to-agent)** is worth naming: `corroborate_with_news` lets the SEC agent call the
News agent *as a tool*, giving cross-source verification instead of one agent guessing outside
its domain.

## 20. Why the tool order is forced — the dependency chain

Newcomers ask why `get_company_info` runs first. The answer is in the signatures:

```python
get_company_info(ticker_or_name: str)                        # give it "AMZN"
get_recent_filings(cik: str, form_type: str, limit: int)     # REQUIRES cik
get_income_statement(cik: str, accession_number: str, ...)   # REQUIRES both
```

SEC identifies companies by **CIK** (Central Index Key), not ticker — tickers change, CIKs do
not. And an **accession number** identifies one specific filing, since a company files many.

```
"AMZN" ─get_company_info─▶ CIK 0001018724
                              └─get_recent_filings─▶ accession 0001018724-26-000004
                                                        └─get_income_statement─▶ numbers
```

The agent **could not** have called `get_income_statement` first — it had nothing to pass in.
**Nobody wrote this sequence down.** The model inferred it from the tool schemas. That is the
agentic contribution.

## 21. A complete worked example, from the live run

**Claim:** *"Amazon's AWS segment net sales were $115 billion in fiscal 2025"*
Parser output: `claim_type="sec"`, `ticker="AMZN"`, `value=115e9`, **`metric=null`** — because
"AWS segment net sales" is not among the 22 whitelisted GAAP metrics. Hold onto that null.

| Step | Tool | Why | Result |
|---|---|---|---|
| 1 | `get_company_info` | everything needs a CIK | `cik=0001018724`, FY ends 12-31 |
| 2 | `get_recent_filings` | need the specific 10-K | `accession=0001018724-26-000004` |
| 3 | `get_income_statement` | try structured data first | **total** revenue only |
| 4 | **`search_filing_text`** | XBRL has no "AWS segment" tag | 5 chunks, segment table |

Step 3 → 4 is the pivot. XBRL has a standard tag for total revenue and **none for AWS** —
segment breakdowns are disclosed in the notes, written for humans. That is what `metric=null`
was signalling back at the parser.

The model composed the call itself:

```python
query="AWS segment net sales", ticker="AMZN",
section="financial_statements_and_notes", filing_type="10-K"
```

It found **$128,725 million**. Claimed $115B. Difference **10.66%** against a 1.5% tolerance →
**REFUTES**, confidence 1.0. Python agreed, so no override fired.

**An instructive detail.** Ranking 40 real Amazon chunks by pure vector similarity for that
same query did *not* put the segment table on top. The live agent found it because it set
`section="financial_statements_and_notes"` first. **Your retrieval quality is leaning on the
agent's filter choices at least as much as on the embeddings** — which is measurable, and
unmeasured.

---

# PART V — EVIDENCE AND PROVENANCE

## 22. Why provenance is the whole product

For a verification system, "REFUTES" without evidence is an opinion. The audit trail must
answer: which passage, from which filing, found by which query?

FinVet captures this in three moves:

1. `_provenance_tool_names = {"search_filing_text", "corroborate_with_news"}` on the SEC agent
   marks which tools have their **full, untruncated** results stored.
2. `run_sec_agent` lifts the chunks into `state["rag_chunks_retrieved"]`, tagging each with the
   query that found it.
3. `_format_metadata` sets `data_sources["rag"]`, rendering the purple **RAG** badge and
   persisting to the audit trail's `data_sources` JSONB column.

## 23. The provenance bug — a case study in silent failure

**The cause.** Both provenance tools returned Pydantic models. LangChain stringifies a tool's
return value, and a `BaseModel` repr looks like `success=True chunks=[...]` — neither valid
JSON nor a Python literal. So `_parse_provenance` tried `json.loads`, then `ast.literal_eval`,
failed both, and fell back to `{"raw": "..."}`. Then `run_sec_agent` checked
`prov_result.get("success")`, got `None`, and skipped every chunk.

**The fix.** Return `.model_dump()` instead. The models stay for validation and documentation,
but what LangChain stringifies is now a Python dict literal that `ast.literal_eval` recovers
cleanly. Applied to `corroborate_with_news` too — it had the identical bug, which would have
silently discarded agent-to-agent evidence the same way.

**Verified live, not just in tests.** Re-running the same Apple Services claim:

| | before | after |
|---|---:|---|
| `rag_chunks_retrieved` downstream | 0 | **3** |
| `data_sources` keys | — | `['rag']` |
| RAG badge would render | no | **yes** |

The chunks now arriving downstream are the right ones — the segment table and the Services
net-sales footnote, each tagged with the query that found them
(`'Services segment net sales fiscal 2025'`), which is exactly what a reviewer needs.

**Why this class of bug is the dangerous one.** Nothing crashed. No test failed. The agent
retrieved correctly and reached the right verdict. Only the *evidence* vanished — and evidence
is the product. It is the same shape as the verdict-override gap found earlier this week: the
mechanism worked, the audit trail stayed silent.

> **Best practice.** Return plain dicts from tools, or assert the round-trip. Anything crossing
> a serialisation boundary needs a test that the *other side* can read it. And test for the
> presence of evidence, not merely the correctness of answers.

---

# PART VI — BEST PRACTICES, CONSOLIDATED

## 24. Ingestion

1. **Exploit document structure** before generic splitting. Section labels enable filters, and
   filters may matter more than embeddings.
2. **Chunk on semantic boundaries** in descending order: section → paragraph → sentence →
   characters. Never fixed character counts first.
3. **Always overlap** (~20%). Facts land on seams.
4. **Handle the pathological case** — a single oversized paragraph must degrade gracefully.
5. **Measure tokens with the tokenizer you actually embed with.** FinVet does not; harmless
   here, but note it.
6. **Assert shapes at boundaries.** Embedding bugs are silent.
7. **Batch embedding calls.** 50 at a time, not 988 round trips.
8. **Store metadata alongside vectors** so filters are cheap SQL.

## 25. Retrieval

9. **Hybrid beats either arm alone.** Semantic finds meaning; lexical finds exact tokens.
10. **Fuse by rank, not score.** RRF avoids incomparable scales.
11. **Degrade gracefully.** Losing the embedder must not disable keyword search — FinVet
    currently fails this.
12. **Set a minimum similarity floor.** `LIMIT 20` always returning 20 rows is a path to
    fabricated evidence.
13. **Use the same model for queries and documents**, always.
14. **Treat tuning constants as hypotheses.** `RRF_K = 60` is inherited, not measured.

## 26. Agentic design

15. **Docstrings are the model's only interface.** Include what the tool is *not* for.
16. **Let the model choose retrieval** — but measure whether it chose correctly. FinVet counts
    tool calls; nothing grades tool *selection*.
17. **Bound the loop, and instrument the bound.** Loop exhaustion should be a first-class
    metric, not prose that a classifier then fails to match.
18. **Separate evidence-gathering from deciding** (the anti-anchoring verdict call).
19. **Deterministic checks wherever the domain allows.** Let the model read; let arithmetic
    compute.

## 27. Evidence

20. **Capture provenance before truncation.**
21. **Test that evidence survives to the surface**, not merely that answers are right.
22. **Return dicts from tools**, or test the serialisation round-trip.

---

# PART VII — FINVET'S OPEN RAG ISSUES

**Already fixed in this worktree** (verified 2026-08-24):

| Issue | Fix |
|---|---|
| Embedder outage killed the whole query | `_embed_single` wrapped; degrades to keyword-only (`service.py:218-225`) |
| `available` was `bool(api_key)` — healthy when broken | now a live corpus-count probe, deliberately *not* an embedder check (`service.py:72`) |
| Section boundaries never closed | sections now terminate correctly; `legal_proceedings` is 3 chunks, not 96 KB |
| Provenance discarded (Pydantic repr) | tools return `.model_dump()`; covered by `tests/unit/test_tool_provenance.py` |

**Still open:**

| # | Issue | Impact | Where |
|---|---|---|---|
| 1 | No distance threshold on the vector arm | `LIMIT 20` always returns 20 rows, however irrelevant — a path to fabricated evidence | vector SQL |
| 2 | `RRF_K = 60` unmeasured | inherited default; unknown whether right for filings | `constants.py:55` |
| 3 | Task prefixes unused | tested, inconclusive — needs a gold set to settle | `service.py:33` |
| 4 | `cl100k_base` sizing a nomic model | cosmetic mismatch, harmless | `parser.py:75` |
| 5 | Tool *selection* ungraded | call counts recorded, correctness of choice is not | agent bench |
| 6 | **No retrieval evaluation at all** | quality entirely unknown | — |

**Number 8 is the one that matters.** Everything above is a hypothesis until retrieval is
measured. That requires a gold set of narrative claims — each with a claim, an expected
verdict, *and* the passage that should have been retrieved — which is what makes Ragas context
precision and recall computable.

---

*Compiled 2026-08-24. Code references: `finvet-v2.0.9/.claude/worktrees/rag-fix`.*

---

# APPENDIX — Complete file inventory for the agentic RAG path

Root: `~/Projects/Active/finvet-v2.0.9/.claude/worktrees/rag-fix`

## 1. Ingestion — offline: HTML → sections → chunks → vectors

| File | Lines | Role |
|---|---:|---|
| `src/finvet/rag/parser.py` | 331 | `_ITEM_PATTERN`, `parse_filing_html()`, `chunk_sections()`, `_get_overlap()`, `_count_tokens()` |
| `src/finvet/rag/service.py` | 350 | `_embed_texts()`, `_embed_single()`, `ingest_filing()`, `search()` — the hybrid retriever and RRF fusion |
| `src/finvet/rag/models.py` | 37 | `FilingChunk` — the SQLAlchemy row: `embedding vector(768)`, generated `tsv` |
| `src/finvet/rag/ingest.py` | 122 | CLI entry point: `python -m finvet.rag.ingest --dir data/filings` |
| `src/finvet/rag/__init__.py` | 14 | package exports |

## 2. Tools the agent may call

| File | Lines | Role |
|---|---:|---|
| `src/finvet/tools/filing_search.py` | 96 | **`search_filing_text`** — the RAG tool |
| `src/finvet/tools/corroborate.py` | 93 | **`corroborate_with_news`** — A2A, SEC agent → News agent |
| `src/finvet/tools/sec_tools.py` | 392 | the five XBRL tools + `SEC_TOOLS` list |
| `src/finvet/tools/memory_tools.py` | 82 | `search_past_verifications` |

## 3. The agent

| File | Lines | Role |
|---|---:|---|
| `src/finvet/agents/base.py` | 591 | ReAct loop, `_extract_verdict` (anti-anchoring), `_apply_override` (deterministic), `_parse_provenance`, `_extract_tool_info` |
| `src/finvet/agents/sec_agent/react_agent.py` | 45 | assembles the toolbox; sets `_provenance_tool_names` |
| `src/finvet/agents/prompts/sec_system.txt` | 73 | the SEC agent's system prompt |

## 4. Pipeline / orchestration

| File | Lines | Role |
|---|---:|---|
| `src/finvet/graph/workflow.py` | 338 | the 12-node LangGraph, routing, `_simple_consensus`, HITL |
| `src/finvet/graph/nodes/claim_parser.py` | 416 | claim → 7 fields; emits `metric=null`, which is what sends a claim down the RAG path |
| `src/finvet/graph/nodes/period_resolver.py` | 249 | "fiscal 2025" → dates; injected into SEC tool calls |
| `src/finvet/graph/nodes/domain_agents.py` | 120 | `run_sec_agent` — extracts RAG chunks into `rag_chunks_retrieved` |
| `src/finvet/graph/nodes/response_generator.py` | 403 | `_format_metadata` — sets `data_sources["rag"]`, drives the badge |

## 5. Configuration

| File | Lines | Role |
|---|---:|---|
| `src/finvet/config/constants.py` | 74 | `EMBEDDING_MODEL`, `EMBEDDING_DIMS`, `EMBED_BATCH_SIZE`, `RRF_K`, `RRF_ABSENT_RANK`, `AGENT_MAX_ITERATIONS`, tolerances |
| `src/finvet/config/settings.py` | 105 | `ollama_url`, `embedding_timeout_s`, LLM stage configs |
| `src/finvet/config/metrics.py` | 169 | `METRIC_WHITELIST` (22 SEC metrics), `METRIC_TO_CONCEPTS` |
| `src/finvet/config/database.py` | 98 | engine, session, `Base` |

## 6. Tests

| File | Lines | Role |
|---|---:|---|
| `tests/unit/test_tool_provenance.py` | 74 | pins the provenance round-trip — the only test touching the RAG path |

## 7. Data and storage

| Location | What |
|---|---|
| `data/filings/<TICKER>/<TICKER>_<FORM>_<PERIOD>.html` — **in the main tree, `finvet-v2.0.9/`, not this worktree** | source filings (AAPL, AMZN, MSFT, NVDA, TSLA) |
| Postgres `filing_chunks` | 988 rows; `vector(768)` + generated `tsvector`; HNSW + GIN + btree indexes |
| Ollama `http://localhost:11434` | serves `nomic-embed-text` |

## Reading order, if you want to follow one query end to end

1. `graph/nodes/claim_parser.py` — where `metric=null` is produced
2. `graph/workflow.py` — routing to the SEC agent
3. `agents/sec_agent/react_agent.py` — the toolbox
4. `agents/base.py` — the ReAct loop
5. `tools/filing_search.py` — the tool the model chooses
6. `rag/service.py` `search()` — embed, two arms, RRF
7. `graph/nodes/domain_agents.py` — chunks into state
8. `graph/nodes/response_generator.py` — chunks into the response
