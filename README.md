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

# Index a view
bun run src/cli/relai.ts index \
  --source crm \
  --remoteId "customers/42" \
  --text "Customer ACME. Enterprise tier." \
  --type customer

# Search
bun run src/cli/relai.ts search "enterprise customers"

# Create a claim
bun run src/cli/relai.ts claim view:notes:acme-renewal.md mentions view:crm:customers/42

# Query relationships
bun run src/cli/relai.ts related view:crm:customers/42
```

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
- **Embeddings**: Local via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) — default model is `embeddinggemma-300M` (GGUF from HuggingFace)
- **Runtime**: Bun (also works with Node.js via better-sqlite3)

## Tests

```bash
bun test
```

## License

MIT
