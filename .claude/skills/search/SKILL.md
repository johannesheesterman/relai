---
name: search
description: Semantic search across all indexed views using a natural language query
argument-hint: [query] [k]
---

Search indexed views using natural language.

## CLI

```bash
bun run src/cli/relai.ts search "$0" -k ${1:-5}
```

## Library

```ts
import { Relai } from "./src/index.js";

const relai = new Relai();
const views = await relai.search("renewal risk", 5);
for (const view of views) {
  console.log(view.id, view.source, view.text);
}
await relai.dispose();
```

## Parameters

- **query**: Natural language search query
- **k** (default: 5): Maximum number of results

## How it works

1. The query is embedded with a query-specific prefix (asymmetric search)
2. sqlite-vec performs KNN cosine similarity search against all indexed vectors
3. Matching views are hydrated from the view store and returned sorted by score
