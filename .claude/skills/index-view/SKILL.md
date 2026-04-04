---
name: index-view
description: Index a semantic view of external data into relai so it becomes searchable
argument-hint: [source] [remoteId] [text]
---

Create a semantic view and index it into relai.

## CLI

```bash
bun run src/cli/relai.ts index --source $0 --remoteId $1 --text $2
```

## Library

```ts
import { Relai } from "./src/index.js";

const relai = new Relai();
await relai.index({
  source: "sqlite",
  remoteId: "customers/42",
  type: "customer",
  text: "Customer ACME. Email hello@acme.com. Enterprise tier.",
});
await relai.dispose();
```

## Parameters

- **source**: Where the data comes from (`sqlite`, `files`, `api`, `rss`, etc.)
- **remoteId**: Identifier in that source system (`customers/42`, `notes/acme.md`)
- **text**: Natural language summary used for embedding and search
- **type** (optional): Semantic classification (`customer`, `note`, `ticket`)

## How it works

1. A View is created with id `view:{source}:{remoteId}`
2. The text is embedded using the local embeddinggemma-300M model
3. The view and its vector are stored in SQLite + sqlite-vec
4. Re-indexing the same source+remoteId replaces the existing view (idempotent)

## Tips for writing view text

- Write natural language, not raw data dumps
- Include key entities, names, and concepts someone might search for
- Summarize rather than copy — the source of truth stays external
- Multiple views can reference the same source object with different perspectives
