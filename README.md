# relai

Semantic indexing layer over heterogeneous data sources.

relai doesn't store raw data. It creates **Views** — vector-searchable semantic representations of external objects — and **Claims** — relationships between views. Everything is searchable through local embeddings and vector search.

## How it works

```
External Data → View (semantic text) → Embedding → Vector Index
                                                         ↓
                                            Natural Language Query
```

An AI agent (or any code) creates Views by summarizing external data into natural language. relai embeds and indexes them. Search is purely semantic — no filters, no exact matching, just meaning.

Claims link views together. When a claim is created, relai automatically embeds it too, so relationships are discoverable through the same search.

## Install

```bash
bun install
```

On macOS, sqlite-vec requires Homebrew's SQLite:

```bash
brew install sqlite
```

## Quick start

### Library

```ts
import { Relai } from "relai";

const relai = new Relai();

// Index views — you decide what text represents the data
await relai.index({
  source: "crm",
  remoteId: "customers/42",
  type: "customer",
  text: "Customer ACME. Enterprise tier. Main contact hello@acme.com.",
});

await relai.index({
  source: "notes",
  remoteId: "acme-renewal.md",
  text: "Talked to ACME about renewal risk. They may churn Q3.",
});

// Create a relationship
await relai.claim(
  "view:notes:acme-renewal.md",
  "mentions",
  "view:crm:customers/42"
);

// Search — finds views and claims by meaning
const results = await relai.search("which customers might churn?");

await relai.dispose();
```

### CLI

```bash
# Download the embedding model (~300MB, runs locally)
bun run src/cli/relai.ts pull

# Download all models — embedding + reranker + query-expansion (generation)
bun run src/cli/relai.ts pull --all

# Index a view
bun run src/cli/relai.ts index \
  --source crm \
  --remoteId "customers/42" \
  --text "Customer ACME. Enterprise tier." \
  --type customer

# Search (hybrid: BM25 + vector + RRF + rerank + query expansion)
bun run src/cli/relai.ts search "enterprise customers"

# Search with flags
bun run src/cli/relai.ts search "enterprise customers" -k 10   # top-k (default 5)
bun run src/cli/relai.ts search "enterprise customers" --no-rerank   # skip cross-encoder rerank
bun run src/cli/relai.ts search "enterprise customers" --no-expand   # skip query expansion
bun run src/cli/relai.ts search "enterprise customers" --explain     # print pipeline diagnostics

# Create a claim
bun run src/cli/relai.ts claim view:notes:acme-renewal.md mentions view:crm:customers/42

# Query relationships
bun run src/cli/relai.ts related view:crm:customers/42

# Run the retrieval-quality benchmark (precision/recall/F1)
bun run src/cli/relai.ts bench
```

The `--no-rerank` and `--no-expand` flags trade result quality for latency: they
skip the optional reranker / query-expansion models, so search runs on the
vector + BM25 + RRF core alone (and never needs those extra model downloads).

## Core concepts

### View

A semantic representation of an external object. You provide the text — relai handles embedding and search.

```ts
{
  source: "crm",           // where the data comes from
  remoteId: "customers/42", // identifier in that system
  type: "customer",         // optional classification
  text: "Customer ACME..." // natural language — this is what gets embedded
}
```

Views are replaceable. Re-indexing the same `source:remoteId` replaces the existing view.

### Claim

A relationship between two views. Claims are automatically embedded and searchable.

```ts
await relai.claim("view:notes:acme.md", "mentions", "view:crm:customers/42");
```

## Architecture

- **Storage**: Single SQLite file (`~/.config/relai/relai.sqlite`)
- **Vectors**: [sqlite-vec](https://github.com/asg017/sqlite-vec) for KNN cosine similarity search
- **Keyword index**: SQLite FTS5 (BM25) over view/chunk text
- **Embeddings**: Local via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) — default model is `embeddinggemma-300M` (GGUF from HuggingFace)
- **Runtime**: Bun (also works with Node.js via better-sqlite3)

## Search architecture

Search is a multi-stage hybrid retrieval pipeline, not a single vector lookup.
Long views are chunked at index time; retrieval collapses chunks back to views
and reranks each view by its best-matching chunk.

```
query
  ├─ query expansion (typed lex / vec / hyde variants)   [optional, generation model]
  ├─ BM25 keyword search (FTS5)        ┐
  └─ vector search (sqlite-vec)        ┘→ Reciprocal Rank Fusion (RRF)
                                           ↓
                            collapse chunks → views (best chunk per view)
                                           ↓
                  cross-encoder rerank + position-aware blend   [optional, reranker model]
                                           ↓
                                        top-k views
```

Stages and their models:

- **BM25 + vector + RRF** — the always-on core. No LLM required; works even if
  the reranker / generation models are unavailable.
- **Cross-encoder rerank** — `Qwen3-Reranker-0.6B` (GGUF). Rescplits candidates
  scores candidates on query/document relevance, blended position-aware with the RRF ranking.
  Disable with `--no-rerank`.
- **Typed query expansion** — `Qwen2.5-0.5B-Instruct` (GGUF) generates `lex` /
  `vec` / `hyde` query variants routed through fusion. Disable with `--no-expand`.

The reranker and generation models are lazy-loaded and optional: pull them with
`relai pull --all`, or skip them per-search with `--no-rerank` / `--no-expand`.
Model URIs are overridable via `RELAI_EMBED_MODEL`, `RELAI_RERANK_MODEL`, and
`RELAI_GENERATE_MODEL`.

## Benchmark

A small labeled eval corpus under `bench/` measures retrieval quality
(precision@k / recall@k / F1@k) so pipeline changes are measured, not guessed:

```bash
bun run bench        # or: bun run src/cli/relai.ts bench
```

## Tests

```bash
bun test
```

## License

MIT
