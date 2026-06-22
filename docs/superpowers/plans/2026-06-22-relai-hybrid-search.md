# Relai Hybrid Search Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade relai's single-stage vector search into a multi-stage hybrid retrieval engine (BM25 + vector + RRF fusion + cross-encoder reranking + chunking + typed query expansion), modeled on tobi/qmd, optimized for maximum result quality.

**Architecture:** relai keeps its current building blocks — `embeddinggemma-300M-Q8_0` embeddings via `node-llama-cpp`, SQLite + `sqlite-vec` storage — and adds a retrieval pipeline on top. Each stage is an independent, separately-testable module (`fts-index`, `fusion`, `chunker`, `reranker`, `query-expansion`) wired together by a rewritten `Relai.search()`. A benchmark harness (precision@k / F1) is built first so every later stage is measured, not guessed. The existing claims-as-searchable-views feature is preserved throughout.

**Tech Stack:** TypeScript, Bun (`bun test`, `bun:sqlite`) with Node/`better-sqlite3` fallback, `sqlite-vec` (vec0), SQLite FTS5, `node-llama-cpp` (embeddings + ranking context + chat session), GGUF models from HuggingFace.

## Global Constraints

- **Test runner:** `bun test`. Tests live in `test/*.test.ts`, import from `../src/*.js`, use `import { describe, test, expect, beforeEach } from "bun:test"`.
- **In-memory DB for tests:** `openDatabase(":memory:")`. Never hit a real model in unit tests — inject fake `Embedder` / `Reranker` / `QueryExpander` via constructor options.
- **No new heavyweight required deps.** Reranker and generation models are pulled at runtime via `node-llama-cpp`'s `resolveModelFile` (same mechanism as the existing embedder), cached under `~/.cache/relai/models/`. They must be lazy-loaded and optional — search must still work (vector + BM25 + RRF) if rerank/expansion models are unavailable.
- **ESM imports use `.js` extensions** (e.g. `import { foo } from "./bar.js"`) even though source is `.ts` — match existing code.
- **Backward compatibility:** existing `Relai.index()`, `Relai.claim()`, `Relai.related()`, and `Relai.search(query, k)` signatures must keep working. New behavior is additive via an options object.
- **Deterministic IDs preserved:** views remain `view:{source}:{remoteId}`; claim views remain `view:claim:{claimId}`.
- **Model URIs (defaults, env-overridable):**
  - Embed: `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf` (existing, `RELAI_EMBED_MODEL`)
  - Rerank: `hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf` (`RELAI_RERANK_MODEL`)
  - Generation (expansion): `hf:ggml-org/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf` (`RELAI_GENERATE_MODEL`)

---

## File Structure

**New files:**
- `src/fts-index.ts` — FTS5 keyword index over view text. `createFtsIndex(db)` → `{ upsert, remove, search }`.
- `src/fusion.ts` — pure functions: `reciprocalRankFusion()` and `positionAwareBlend()`. No I/O, no deps.
- `src/chunker.ts` — pure function `chunkText(text, opts)` → chunks with char positions. Code-fence-aware.
- `src/chunk-store.ts` — SQLite table mapping chunk ids → (view_id, pos, text). `createChunkStore(db)`.
- `src/reranker.ts` — `createReranker(opts)` → `{ rank, dispose }` over `node-llama-cpp` ranking context.
- `src/query-expansion.ts` — `createQueryExpander(opts)` → `{ expand, dispose }` producing typed `lex`/`vec`/`hyde` variants.
- `src/search-pipeline.ts` — orchestrates the stages. `hybridSearch(deps, query, options)`.
- `bench/dataset.json` — small labeled eval corpus (docs + queries + relevant ids).
- `bench/run.ts` — runs `hybridSearch` over the dataset, prints precision@k / recall / F1.
- Test files mirroring each module under `test/`.

**Modified files:**
- `src/types.ts` — new interfaces (`FtsIndex`, `ChunkStore`, `Reranker`, `QueryExpander`, `SearchOptions`, `SearchResult`, `RankedItem`, `Chunk`, `ExpandedQuery`).
- `src/vector-index.ts` — generalize `upsert`/`search` to operate on arbitrary string ids (chunk ids), add `remove()`.
- `src/relai.ts` — inject new stores/stages; rewrite `index()` to chunk, rewrite `search()` to call `hybridSearch`.
- `src/embedder.ts` — add `embedMany()` batch helper (used for multi-chunk indexing).
- `src/cli/relai.ts` — surface new search flags (`--no-rerank`, `--no-expand`, `--explain`, `-k`) and a `bench` command.
- `package.json` — add `"bench": "bun run bench/run.ts"` script.

---

## Phase 0 — Benchmark harness (build first, measure everything)

### Task 1: Eval dataset + scoring functions

**Files:**
- Create: `bench/dataset.json`
- Create: `bench/score.ts`
- Test: `test/score.test.ts`

**Interfaces:**
- Produces: `precisionAtK(retrieved: string[], relevant: string[], k: number): number`, `recallAtK(retrieved: string[], relevant: string[], k: number): number`, `f1AtK(retrieved: string[], relevant: string[], k: number): number`. All operate on ordered arrays of view ids.

- [ ] **Step 1: Write the failing test**

```typescript
// test/score.test.ts
import { describe, test, expect } from "bun:test";
import { precisionAtK, recallAtK, f1AtK } from "../bench/score.js";

describe("scoring", () => {
  test("precision@k counts relevant in top-k", () => {
    expect(precisionAtK(["a", "b", "c"], ["a", "c"], 2)).toBeCloseTo(0.5); // a relevant, b not → 1/2
  });
  test("recall@k counts found relevant over total relevant", () => {
    expect(recallAtK(["a", "b", "c"], ["a", "c", "z"], 3)).toBeCloseTo(2 / 3);
  });
  test("f1 combines precision and recall", () => {
    expect(f1AtK(["a", "b"], ["a"], 2)).toBeCloseTo(2 * (0.5 * 1) / (0.5 + 1));
  });
  test("empty relevant yields 0", () => {
    expect(precisionAtK(["a"], [], 1)).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/score.test.ts`
Expected: FAIL — `Cannot find module '../bench/score.js'`

- [ ] **Step 3: Write minimal implementation**

```typescript
// bench/score.ts
export function precisionAtK(retrieved: string[], relevant: string[], k: number): number {
  const top = retrieved.slice(0, k);
  if (top.length === 0) return 0;
  const rel = new Set(relevant);
  const hits = top.filter((id) => rel.has(id)).length;
  return hits / top.length;
}

export function recallAtK(retrieved: string[], relevant: string[], k: number): number {
  if (relevant.length === 0) return 0;
  const top = new Set(retrieved.slice(0, k));
  const hits = relevant.filter((id) => top.has(id)).length;
  return hits / relevant.length;
}

export function f1AtK(retrieved: string[], relevant: string[], k: number): number {
  const p = precisionAtK(retrieved, relevant, k);
  const r = recallAtK(retrieved, relevant, k);
  if (p + r === 0) return 0;
  return (2 * p * r) / (p + r);
}
```

- [ ] **Step 4: Create the dataset**

Create `bench/dataset.json` with this shape (start with ~15 docs and ~8 queries spanning both short records and long documents so chunking effects show up; expand over time):

```json
{
  "docs": [
    { "source": "tickets", "remoteId": "1", "type": "ticket", "text": "Customer reports the export to CSV button does nothing on the reports page." },
    { "source": "docs", "remoteId": "billing", "type": "doc", "text": "Billing overview. Invoices are generated monthly. To export invoices as CSV, open Reports, choose a date range, and click Export. Refunds are processed within 5 business days. VAT is applied per the customer's billing country..." }
  ],
  "queries": [
    { "query": "csv export broken", "relevant": ["view:tickets:1", "view:docs:billing"] }
  ]
}
```

- [ ] **Step 5: Run tests to verify pass**

Run: `bun test test/score.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add bench/score.ts bench/dataset.json test/score.test.ts
git commit -m "test: add benchmark scoring (precision/recall/f1) and eval dataset"
```

### Task 2: Benchmark runner skeleton (vector-only baseline)

**Files:**
- Create: `bench/run.ts`
- Modify: `package.json` (add `bench` script)

**Interfaces:**
- Consumes: `Relai` from `../src/relai.js`, scoring from `./score.js`.
- Produces: a CLI that indexes `dataset.json` docs into a temp in-file DB, runs each query through `relai.search`, and prints mean precision@5 / recall@5 / f1@5.

- [ ] **Step 1: Write the runner**

```typescript
// bench/run.ts
import { Relai } from "../src/relai.js";
import { precisionAtK, recallAtK, f1AtK } from "./score.js";
import { readFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

type Dataset = {
  docs: { source: string; remoteId: string; type?: string; text: string }[];
  queries: { query: string; relevant: string[] }[];
};

async function main() {
  const data: Dataset = JSON.parse(
    readFileSync(new URL("./dataset.json", import.meta.url), "utf8")
  );
  const dbPath = join(tmpdir(), `relai-bench-${process.pid}.sqlite`);
  const relai = new Relai({ dbPath });

  for (const d of data.docs) await relai.index(d);

  const k = 5;
  let p = 0, r = 0, f = 0;
  for (const q of data.queries) {
    const results = await relai.search(q.query, k);
    const ids = results.map((v) => v.id);
    p += precisionAtK(ids, q.relevant, k);
    r += recallAtK(ids, q.relevant, k);
    f += f1AtK(ids, q.relevant, k);
  }
  const n = data.queries.length;
  console.log(`queries=${n}  P@${k}=${(p / n).toFixed(3)}  R@${k}=${(r / n).toFixed(3)}  F1@${k}=${(f / n).toFixed(3)}`);
  await relai.dispose();
}

main();
```

- [ ] **Step 2: Add bench script to package.json**

In `package.json` `"scripts"`, add: `"bench": "bun run bench/run.ts"`.

- [ ] **Step 3: Run the baseline**

Run: `bun run bench`
Expected: a line like `queries=8  P@5=0.xxx  R@5=0.xxx  F1@5=0.xxx`. **Record these numbers** — this is the vector-only baseline every later phase is compared against.

- [ ] **Step 4: Commit**

```bash
git add bench/run.ts package.json
git commit -m "feat: add benchmark runner with vector-only baseline"
```

---

## Phase 1 — Hybrid retrieval (BM25 + vector + RRF). Highest ROI, no LLM.

### Task 3: FTS5 keyword index

**Files:**
- Create: `src/fts-index.ts`
- Modify: `src/types.ts` (add `FtsIndex`, `RankedItem`)
- Test: `test/fts-index.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export type RankedItem = { id: string; score: number };
  export interface FtsIndex {
    upsert(id: string, text: string): void;
    remove(id: string): void;
    search(query: string, k: number): RankedItem[];
  }
  export function createFtsIndex(db: Database): FtsIndex;
  ```
  Scores normalized to `[0,1)`, higher = better.

- [ ] **Step 1: Add types**

In `src/types.ts` append:

```typescript
export type RankedItem = { id: string; score: number };

export interface FtsIndex {
  upsert(id: string, text: string): void;
  remove(id: string): void;
  search(query: string, k: number): RankedItem[];
}
```

- [ ] **Step 2: Write the failing test**

```typescript
// test/fts-index.test.ts
import { describe, test, expect, beforeEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createFtsIndex } from "../src/fts-index.js";
import type { FtsIndex } from "../src/types.js";

let db: Database;
let fts: FtsIndex;

beforeEach(() => {
  db = openDatabase(":memory:");
  fts = createFtsIndex(db);
});

describe("FtsIndex", () => {
  test("finds exact keyword matches", () => {
    fts.upsert("a", "the quarterly revenue report for ACME");
    fts.upsert("b", "a recipe for chocolate cake");
    const results = fts.search("revenue report", 5);
    expect(results[0]?.id).toBe("a");
    expect(results[0]?.score).toBeGreaterThan(0);
  });

  test("returns empty for no match", () => {
    fts.upsert("a", "hello world");
    expect(fts.search("zzz", 5)).toEqual([]);
  });

  test("upsert replaces text", () => {
    fts.upsert("a", "first version about cats");
    fts.upsert("a", "second version about dogs");
    expect(fts.search("cats", 5)).toEqual([]);
    expect(fts.search("dogs", 5)[0]?.id).toBe("a");
  });

  test("remove deletes from index", () => {
    fts.upsert("a", "find me");
    fts.remove("a");
    expect(fts.search("find", 5)).toEqual([]);
  });

  test("k limits results", () => {
    for (let i = 0; i < 10; i++) fts.upsert(`v${i}`, "common token here");
    expect(fts.search("common", 3)).toHaveLength(3);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/fts-index.test.ts`
Expected: FAIL — `Cannot find module '../src/fts-index.js'`

- [ ] **Step 4: Write implementation**

```typescript
// src/fts-index.ts
import type { Database } from "./db.js";
import type { FtsIndex, RankedItem } from "./types.js";

// FTS5 query syntax is strict; sanitize user input into a safe OR-of-terms query.
function toMatchQuery(query: string): string {
  const terms = query
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0)
    .map((t) => `"${t}"`);
  return terms.join(" OR ");
}

export function createFtsIndex(db: Database): FtsIndex {
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS views_fts USING fts5(
      id UNINDEXED,
      text
    )
  `);

  const removeStmt = db.prepare(`DELETE FROM views_fts WHERE id = ?`);
  const insertStmt = db.prepare(`INSERT INTO views_fts (id, text) VALUES (?, ?)`);

  return {
    upsert(id: string, text: string) {
      removeStmt.run(id);
      insertStmt.run(id, text);
    },

    remove(id: string) {
      removeStmt.run(id);
    },

    search(query: string, k: number): RankedItem[] {
      const match = toMatchQuery(query);
      if (match.length === 0) return [];
      // bm25() is negative; lower (more negative) = better. Order ascending.
      const rows = db
        .prepare(
          `SELECT id, bm25(views_fts) AS score
           FROM views_fts
           WHERE views_fts MATCH ?
           ORDER BY score ASC
           LIMIT ?`
        )
        .all(match, k) as { id: string; score: number }[];
      return rows.map((row) => {
        const mag = Math.abs(row.score);
        // Map negative bm25 → stable (0,1), higher = better.
        return { id: row.id, score: mag / (1 + mag) };
      });
    },
  };
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/fts-index.test.ts`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/fts-index.ts src/types.ts test/fts-index.test.ts
git commit -m "feat: add FTS5 keyword index (BM25) over view text"
```

### Task 4: RRF fusion (pure function)

**Files:**
- Create: `src/fusion.ts`
- Test: `test/fusion.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export type FusedItem = { id: string; rrfScore: number; rank: number };
  export function reciprocalRankFusion(
    lists: RankedItem[][],
    weights?: number[],
    k?: number
  ): FusedItem[];
  ```
  `rank` is 1-indexed position in the fused output. Default `k=60`. Applies qmd's top-rank bonus (best contributing rank 0 → +0.05, rank ≤ 2 → +0.02).

- [ ] **Step 1: Write the failing test**

```typescript
// test/fusion.test.ts
import { describe, test, expect } from "bun:test";
import { reciprocalRankFusion } from "../src/fusion.js";

describe("reciprocalRankFusion", () => {
  test("doc ranked high in both lists wins", () => {
    const bm25 = [{ id: "a", score: 1 }, { id: "b", score: 0.5 }];
    const vec = [{ id: "a", score: 1 }, { id: "c", score: 0.5 }];
    const fused = reciprocalRankFusion([bm25, vec]);
    expect(fused[0]?.id).toBe("a");
    expect(fused[0]?.rank).toBe(1);
  });

  test("weights bias toward a list", () => {
    const original = [{ id: "x", score: 1 }];
    const expansion = [{ id: "y", score: 1 }];
    const fused = reciprocalRankFusion([original, expansion], [2.0, 1.0]);
    expect(fused[0]?.id).toBe("x");
  });

  test("missing/empty lists are tolerated", () => {
    const fused = reciprocalRankFusion([[], [{ id: "a", score: 1 }]]);
    expect(fused[0]?.id).toBe("a");
  });

  test("top-rank bonus boosts a rank-0 contributor", () => {
    // 'a' is rank 0 in list1 (gets +0.05); 'b' is rank 1 in both (no bonus).
    const l1 = [{ id: "a", score: 1 }, { id: "b", score: 0.9 }];
    const l2 = [{ id: "z", score: 1 }, { id: "b", score: 0.9 }];
    const fused = reciprocalRankFusion([l1, l2]);
    expect(fused[0]?.id).toBe("a");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/fusion.test.ts`
Expected: FAIL — `Cannot find module '../src/fusion.js'`

- [ ] **Step 3: Write implementation**

```typescript
// src/fusion.ts
import type { RankedItem } from "./types.js";

export type FusedItem = { id: string; rrfScore: number; rank: number };

export function reciprocalRankFusion(
  lists: RankedItem[][],
  weights: number[] = [],
  k: number = 60
): FusedItem[] {
  const scores = new Map<string, { rrfScore: number; topRank: number }>();

  for (let listIdx = 0; listIdx < lists.length; listIdx++) {
    const list = lists[listIdx];
    if (!list) continue;
    const weight = weights[listIdx] ?? 1.0;
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      if (!item) continue;
      const contribution = weight / (k + rank + 1);
      const existing = scores.get(item.id);
      if (existing) {
        existing.rrfScore += contribution;
        existing.topRank = Math.min(existing.topRank, rank);
      } else {
        scores.set(item.id, { rrfScore: contribution, topRank: rank });
      }
    }
  }

  for (const entry of scores.values()) {
    if (entry.topRank === 0) entry.rrfScore += 0.05;
    else if (entry.topRank <= 2) entry.rrfScore += 0.02;
  }

  return Array.from(scores.entries())
    .map(([id, e]) => ({ id, rrfScore: e.rrfScore }))
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map((e, i) => ({ ...e, rank: i + 1 }));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/fusion.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fusion.ts test/fusion.test.ts
git commit -m "feat: add reciprocal rank fusion with top-rank bonus"
```

### Task 5: Generalize vector index to arbitrary ids + remove()

**Files:**
- Modify: `src/vector-index.ts`
- Modify: `src/types.ts` (`VectorIndex` interface)
- Test: `test/vector-index.test.ts` (add cases)

**Interfaces:**
- Produces (updated):
  ```typescript
  export interface VectorIndex {
    upsert(id: string, vector: number[]): Promise<void>;
    remove(id: string): Promise<void>;
    search(vector: number[], k: number): Promise<{ id: string; score: number }[]>;
  }
  ```
  Note: result key renamed `viewId` → `id` because ids may now be chunk ids (`view:...#3`). Callers map chunk id → view id by stripping `#<n>`.

- [ ] **Step 1: Update the interface in types.ts**

Replace the `VectorIndex` interface in `src/types.ts` with:

```typescript
export interface VectorIndex {
  upsert(id: string, vector: number[]): Promise<void>;
  remove(id: string): Promise<void>;
  search(vector: number[], k: number): Promise<RankedItem[]>;
}
```

(`RankedItem` is `{ id: string; score: number }`, already defined in Task 3.)

- [ ] **Step 2: Update existing test expectations + add remove test**

In `test/vector-index.test.ts`, change every `results[0].viewId` to `results[0].id`, and add:

```typescript
  test("remove deletes a vector", async () => {
    const v1 = normalize([1, 0, 0, 0]);
    await index.upsert("a", v1);
    await index.remove("a");
    expect(await index.search(v1, 5)).toEqual([]);
  });
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/vector-index.test.ts`
Expected: FAIL — `index.remove is not a function` and/or `viewId` assertions now read `undefined`.

- [ ] **Step 4: Update implementation**

In `src/vector-index.ts`, rename the column usage and add `remove`. The vec0 table keeps column name `view_id` (no migration needed) but now stores chunk ids:

```typescript
  return {
    async upsert(id: string, vector: number[]) {
      ensureTable(vector.length);
      const embedding = new Float32Array(vector);
      db.prepare(`DELETE FROM vectors WHERE view_id = ?`).run(id);
      db.prepare(
        `INSERT INTO vectors (view_id, embedding) VALUES (?, ?)`
      ).run(id, embedding);
    },

    async remove(id: string) {
      if (!tableReady) return;
      db.prepare(`DELETE FROM vectors WHERE view_id = ?`).run(id);
    },

    async search(vector: number[], k: number) {
      if (!tableReady) return [];
      const embedding = new Float32Array(vector);
      const rows = db
        .prepare(
          `SELECT view_id, distance FROM vectors WHERE embedding MATCH ? AND k = ?`
        )
        .all(embedding, k);
      return rows.map((row: any) => ({
        id: row.view_id,
        score: 1 - row.distance,
      }));
    },
  };
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/vector-index.test.ts`
Expected: PASS (all original + remove test)

- [ ] **Step 6: Fix the one caller in relai.ts**

In `src/relai.ts` `search()`, the line building `scoreMap` reads `m.viewId`. Change it to `m.id`:

```typescript
    const scoreMap = new Map(matches.map((m) => [m.id, m.score]));
```

And in the same method change `matches.map((m) => m.viewId)` to `matches.map((m) => m.id)`. Run `bun test` to confirm `test/relai.test.ts` still passes.

- [ ] **Step 7: Commit**

```bash
git add src/vector-index.ts src/types.ts test/vector-index.test.ts src/relai.ts
git commit -m "refactor: generalize vector index to arbitrary ids, add remove()"
```

### Task 6: Wire BM25 + vector + RRF into Relai (no chunking yet)

**Files:**
- Modify: `src/relai.ts` (index writes to FTS; search fuses BM25 + vector)
- Test: `test/relai.test.ts` (add hybrid-search test)

**Interfaces:**
- Consumes: `createFtsIndex` (Task 3), `reciprocalRankFusion` (Task 4), generalized `VectorIndex` (Task 5).
- Produces: `Relai.search(query, k)` now returns the RRF-fused union of BM25 and vector hits. Existing behavior (returns `View[]` ordered best-first) preserved.

- [ ] **Step 1: Write the failing test**

```typescript
// add to test/relai.test.ts
test("hybrid search finds exact keyword match that vector search alone may rank lower", async () => {
  await relai.index({ source: "docs", remoteId: "1", text: "general notes about the system architecture and design" });
  await relai.index({ source: "docs", remoteId: "2", text: "the SKU-99XZ part number appears only here" });
  const results = await relai.search("SKU-99XZ", 5);
  expect(results[0]?.id).toBe("view:docs:2");
});
```

(Use whatever `beforeEach` setup the existing `test/relai.test.ts` uses — an in-memory `Relai`. If it doesn't have one, construct `new Relai({ dbPath: ":memory:" })` in `beforeEach` and `dispose` in `afterEach`. Note: a real embedder loads a model; if existing tests already do this they accept the model-download cost — match their setup exactly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/relai.test.ts`
Expected: FAIL — exact SKU not reliably ranked first by vector-only search (or fails because FTS isn't populated).

- [ ] **Step 3: Add FtsIndex to Relai and populate on index()**

In `src/relai.ts`:
- import `createFtsIndex` and `reciprocalRankFusion`, `type FtsIndex`, `type FusedItem`.
- add field `private ftsIndex: FtsIndex;` and in constructor `this.ftsIndex = createFtsIndex(this.db);`
- in `index()`, after `await this.viewStore.put(view);` add `this.ftsIndex.upsert(view.id, view.text);`
- in `claim()`, after `await this.viewStore.put(claimView);` add `this.ftsIndex.upsert(claimView.id, text);`

- [ ] **Step 4: Rewrite search() to fuse**

Replace the body of `search()`:

```typescript
  async search(query: string, k: number = 5): Promise<View[]> {
    const vector = await this.embedder.embedQuery(query);
    const [vecHits, ftsHits] = [
      await this.vectorIndex.search(vector, Math.max(k * 4, 20)),
      this.ftsIndex.search(query, Math.max(k * 4, 20)),
    ];
    const fused = reciprocalRankFusion([vecHits, ftsHits]);
    const topIds = fused.slice(0, k).map((f) => f.id);
    const views = await this.viewStore.getMany(topIds);
    const order = new Map(topIds.map((id, i) => [id, i]));
    return views.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
  }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/relai.test.ts`
Expected: PASS

- [ ] **Step 6: Run the benchmark, compare to baseline**

Run: `bun run bench`
Expected: P@5 / R@5 / F1@5 **≥ the Phase 0 baseline**. Record the new numbers in the commit message. If a metric regressed, investigate before committing (most likely the FTS `toMatchQuery` is over-broad — tighten to AND for multi-term, or reduce candidate width).

- [ ] **Step 7: Commit**

```bash
git add src/relai.ts test/relai.test.ts
git commit -m "feat: hybrid BM25+vector search via RRF fusion

bench: P@5 X.xxx→Y.yyy, F1@5 X.xxx→Y.yyy vs vector-only baseline"
```

---

## Phase 2 — Cross-encoder reranking.

### Task 7: Reranker module (node-llama-cpp ranking context)

**Files:**
- Create: `src/reranker.ts`
- Modify: `src/types.ts` (`Reranker` interface)
- Test: `test/reranker.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export interface Reranker {
    rank(query: string, documents: string[]): Promise<number[]>; // score per doc, same order
    dispose(): Promise<void>;
  }
  export function createReranker(opts?: { model?: string; cacheDir?: string }): Reranker;
  ```

- [ ] **Step 1: Add the interface to types.ts**

```typescript
export interface Reranker {
  rank(query: string, documents: string[]): Promise<number[]>;
  dispose(): Promise<void>;
}
```

- [ ] **Step 2: Write the failing test (loads the real rerank model — tag as integration)**

```typescript
// test/reranker.test.ts
import { describe, test, expect, afterAll } from "bun:test";
import { createReranker } from "../src/reranker.js";

const reranker = createReranker();
afterAll(() => reranker.dispose());

describe("Reranker (integration — downloads Qwen3-Reranker)", () => {
  test("scores a relevant doc above an irrelevant one", async () => {
    const scores = await reranker.rank("how do I export invoices to CSV", [
      "Open Reports, pick a date range, and click Export to download a CSV of invoices.",
      "The history of medieval pottery glazing techniques in northern Europe.",
    ]);
    expect(scores).toHaveLength(2);
    expect(scores[0]).toBeGreaterThan(scores[1]!);
  }, 120_000);
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/reranker.test.ts`
Expected: FAIL — `Cannot find module '../src/reranker.js'`

- [ ] **Step 4: Write implementation**

```typescript
// src/reranker.ts
import { getLlama, resolveModelFile, type Llama, type LlamaModel, type LlamaRankingContext } from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import type { Reranker } from "./types.js";

const DEFAULT_RERANK_MODEL =
  process.env.RELAI_RERANK_MODEL ??
  "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, "relai", "models")
  : join(homedir(), ".cache", "relai", "models");

export function createReranker(opts?: { model?: string; cacheDir?: string }): Reranker {
  const modelUri = opts?.model ?? DEFAULT_RERANK_MODEL;
  const cacheDir = opts?.cacheDir ?? MODEL_CACHE_DIR;

  let llama: Llama | null = null;
  let model: LlamaModel | null = null;
  let context: LlamaRankingContext | null = null;
  let loadPromise: Promise<void> | null = null;

  async function ensureLoaded() {
    if (context) return;
    if (loadPromise) return loadPromise;
    loadPromise = (async () => {
      if (!existsSync(cacheDir)) mkdirSync(cacheDir, { recursive: true });
      const modelPath = await resolveModelFile(modelUri, cacheDir);
      llama = await getLlama();
      model = await llama.loadModel({ modelPath });
      context = await model.createRankingContext();
    })();
    return loadPromise;
  }

  return {
    async rank(query: string, documents: string[]): Promise<number[]> {
      if (documents.length === 0) return [];
      await ensureLoaded();
      // rankAll returns a relevance score per document, in input order.
      return context!.rankAll(query, documents);
    },

    async dispose() {
      if (context) { await context.dispose(); context = null; }
      if (model) { await model.dispose(); model = null; }
      if (llama) { await llama.dispose(); llama = null; }
      loadPromise = null;
    },
  };
}
```

> Note for implementer: confirm the exact node-llama-cpp ranking API on the installed `^3.17.1` (`model.createRankingContext()` / `context.rankAll(query, docs)`). If the method is named `rank` only, loop over documents. Verify via `node -e "import('node-llama-cpp').then(m=>console.log(Object.keys(m)))"` or the package's `.d.ts`. Adjust the two call sites accordingly; keep the `Reranker` interface unchanged.

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/reranker.test.ts`
Expected: PASS (downloads model on first run; allow time)

- [ ] **Step 6: Commit**

```bash
git add src/reranker.ts src/types.ts test/reranker.test.ts
git commit -m "feat: add cross-encoder reranker (Qwen3-Reranker-0.6B)"
```

### Task 8: Position-aware blend (pure function)

**Files:**
- Modify: `src/fusion.ts` (add `positionAwareBlend`)
- Test: `test/fusion.test.ts` (add cases)

**Interfaces:**
- Produces:
  ```typescript
  export function positionAwareBlend(
    items: { id: string; rrfRank: number; rerankScore: number }[]
  ): RankedItem[];
  ```
  Weight by rrf rank: rank ≤ 3 → `0.75*rrf + 0.25*rerank`; rank ≤ 10 → `0.60/0.40`; else `0.40/0.60`. `rrf` term is `1/rrfRank`. Returns sorted desc.

- [ ] **Step 1: Write the failing test**

```typescript
// add to test/fusion.test.ts
import { positionAwareBlend } from "../src/fusion.js";

describe("positionAwareBlend", () => {
  test("top retrieval rank is protected from reranker disagreement", () => {
    // 'a' at rrfRank 1 with low rerank; 'b' at rrfRank 11 with high rerank.
    const blended = positionAwareBlend([
      { id: "a", rrfRank: 1, rerankScore: 0.1 },
      { id: "b", rrfRank: 11, rerankScore: 0.95 },
    ]);
    // a: 0.75*1 + 0.25*0.1 = 0.775 ; b: 0.40*(1/11) + 0.60*0.95 ≈ 0.606
    expect(blended[0]?.id).toBe("a");
  });

  test("strong reranker rescues a mid-rank doc", () => {
    const blended = positionAwareBlend([
      { id: "a", rrfRank: 4, rerankScore: 0.05 },
      { id: "b", rrfRank: 5, rerankScore: 0.99 },
    ]);
    expect(blended[0]?.id).toBe("b");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/fusion.test.ts`
Expected: FAIL — `positionAwareBlend is not a function`

- [ ] **Step 3: Add implementation to src/fusion.ts**

```typescript
export function positionAwareBlend(
  items: { id: string; rrfRank: number; rerankScore: number }[]
): RankedItem[] {
  return items
    .map((it) => {
      let w: number;
      if (it.rrfRank <= 3) w = 0.75;
      else if (it.rrfRank <= 10) w = 0.6;
      else w = 0.4;
      const rrfScore = 1 / it.rrfRank;
      const score = w * rrfScore + (1 - w) * it.rerankScore;
      return { id: it.id, score };
    })
    .sort((a, b) => b.score - a.score);
}
```

(Add `import type { RankedItem } from "./types.js";` at top of `src/fusion.ts` if not already present.)

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/fusion.test.ts`
Expected: PASS (all fusion tests)

- [ ] **Step 5: Commit**

```bash
git add src/fusion.ts test/fusion.test.ts
git commit -m "feat: add position-aware RRF/rerank blend"
```

### Task 9: Search pipeline module (fuse → rerank → blend)

**Files:**
- Create: `src/search-pipeline.ts`
- Modify: `src/types.ts` (`SearchOptions`, `SearchResult`)
- Test: `test/search-pipeline.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export type SearchOptions = {
    k?: number;
    candidateLimit?: number; // default 40
    rerank?: boolean;        // default true
    minScore?: number;       // default 0
  };
  export type SearchDeps = {
    embedQuery: (q: string) => Promise<number[]>;
    vectorSearch: (vec: number[], k: number) => Promise<RankedItem[]>;
    ftsSearch: (q: string, k: number) => RankedItem[];
    getText: (id: string) => string | undefined; // full text for a candidate id (best chunk later)
    reranker?: Reranker;
  };
  export function hybridSearch(deps: SearchDeps, query: string, options?: SearchOptions): Promise<RankedItem[]>;
  ```
  Returns ranked **ids** with final scores. The caller (`Relai`) hydrates to `View[]`. Designed for injecting fakes in tests (no model needed).

- [ ] **Step 1: Add types to src/types.ts**

```typescript
export type SearchOptions = {
  k?: number;
  candidateLimit?: number;
  rerank?: boolean;
  minScore?: number;
};
```

- [ ] **Step 2: Write the failing test (all deps faked, no models)**

```typescript
// test/search-pipeline.test.ts
import { describe, test, expect } from "bun:test";
import { hybridSearch } from "../src/search-pipeline.js";

describe("hybridSearch", () => {
  const deps = {
    embedQuery: async () => [1, 0, 0],
    vectorSearch: async () => [
      { id: "a", score: 0.9 },
      { id: "b", score: 0.8 },
    ],
    ftsSearch: () => [
      { id: "b", score: 0.7 },
      { id: "c", score: 0.6 },
    ],
    getText: (id: string) => `text for ${id}`,
  };

  test("without reranker, returns RRF-fused ids", async () => {
    const out = await hybridSearch(deps, "q", { rerank: false, k: 3 });
    expect(out.map((r) => r.id)).toContain("b"); // b appears in both lists → fused high
    expect(out.length).toBeLessThanOrEqual(3);
  });

  test("with reranker, blends and can promote a doc", async () => {
    const reranker = {
      rank: async (_q: string, docs: string[]) =>
        docs.map((d) => (d.includes("c") ? 0.99 : 0.01)),
      dispose: async () => {},
    };
    const out = await hybridSearch({ ...deps, reranker }, "q", { rerank: true, k: 3 });
    expect(out[0]?.id).toBe("c"); // reranker strongly favors c
  });

  test("rerank=false ignores reranker", async () => {
    let called = false;
    const reranker = {
      rank: async (_q: string, docs: string[]) => { called = true; return docs.map(() => 1); },
      dispose: async () => {},
    };
    await hybridSearch({ ...deps, reranker }, "q", { rerank: false });
    expect(called).toBe(false);
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/search-pipeline.test.ts`
Expected: FAIL — `Cannot find module '../src/search-pipeline.js'`

- [ ] **Step 4: Write implementation**

```typescript
// src/search-pipeline.ts
import type { RankedItem, SearchOptions, Reranker } from "./types.js";
import { reciprocalRankFusion, positionAwareBlend } from "./fusion.js";

export type SearchDeps = {
  embedQuery: (q: string) => Promise<number[]>;
  vectorSearch: (vec: number[], k: number) => Promise<RankedItem[]>;
  ftsSearch: (q: string, k: number) => RankedItem[];
  getText: (id: string) => string | undefined;
  reranker?: Reranker;
};

export async function hybridSearch(
  deps: SearchDeps,
  query: string,
  options: SearchOptions = {}
): Promise<RankedItem[]> {
  const k = options.k ?? 5;
  const candidateLimit = options.candidateLimit ?? 40;
  const doRerank = (options.rerank ?? true) && !!deps.reranker;
  const minScore = options.minScore ?? 0;

  const width = Math.max(candidateLimit, k * 4);
  const vector = await deps.embedQuery(query);
  const [vecHits, ftsHits] = [
    await deps.vectorSearch(vector, width),
    deps.ftsSearch(query, width),
  ];

  const fused = reciprocalRankFusion([vecHits, ftsHits]);
  const candidates = fused.slice(0, candidateLimit);

  if (!doRerank) {
    return candidates
      .slice(0, k)
      .map((c) => ({ id: c.id, score: c.rrfScore }))
      .filter((r) => r.score >= minScore);
  }

  const texts = candidates.map((c) => deps.getText(c.id) ?? "");
  const rerankScores = await deps.reranker!.rank(query, texts);

  const blended = positionAwareBlend(
    candidates.map((c, i) => ({
      id: c.id,
      rrfRank: c.rank,
      rerankScore: rerankScores[i] ?? 0,
    }))
  );

  return blended.filter((r) => r.score >= minScore).slice(0, k);
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/search-pipeline.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/search-pipeline.ts src/types.ts test/search-pipeline.test.ts
git commit -m "feat: add hybrid search pipeline (fuse -> rerank -> blend)"
```

### Task 10: Wire pipeline + reranker into Relai

**Files:**
- Modify: `src/relai.ts`
- Modify: `src/types.ts` (extend `RelaiConfig` is in relai.ts; add reranker injection)
- Test: `test/relai.test.ts`

**Interfaces:**
- Consumes: `hybridSearch` (Task 9), `createReranker` (Task 7).
- Produces: `Relai.search(query, k?, options?)` where `options: SearchOptions`. Reranker is lazy and optional; if disabled or unavailable, falls back to RRF-only. `RelaiConfig` gains `rerankModel?: string` and `rerank?: boolean` (default true).

- [ ] **Step 1: Write the failing test**

```typescript
// add to test/relai.test.ts — verifies the options plumbing without forcing a model load
test("search accepts options and rerank can be disabled", async () => {
  await relai.index({ source: "docs", remoteId: "1", text: "alpha bravo charlie" });
  const out = await relai.search("alpha", 3, { rerank: false });
  expect(out[0]?.id).toBe("view:docs:1");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test test/relai.test.ts`
Expected: FAIL — `search` doesn't accept a third argument / type error.

- [ ] **Step 3: Update Relai**

In `src/relai.ts`:
- import `createReranker`, `hybridSearch`, `type Reranker`, `type SearchOptions`, `type SearchDeps`.
- extend `RelaiConfig`: `rerankModel?: string; rerank?: boolean;`
- add fields `private reranker: Reranker | null = null;` and `private rerankEnabled: boolean;`
- in constructor: `this.rerankEnabled = config?.rerank ?? true;` and store `this.rerankModel = config?.rerankModel;`
- add a lazy getter:

```typescript
  private getReranker(): Reranker | undefined {
    if (!this.rerankEnabled) return undefined;
    if (!this.reranker) this.reranker = createReranker({ model: this.rerankModel });
    return this.reranker;
  }
```

- replace `search()`:

```typescript
  async search(query: string, k: number = 5, options: SearchOptions = {}): Promise<View[]> {
    const rerank = options.rerank ?? this.rerankEnabled;
    const deps: SearchDeps = {
      embedQuery: (q) => this.embedder.embedQuery(q),
      vectorSearch: (vec, n) => this.vectorIndex.search(vec, n),
      ftsSearch: (q, n) => this.ftsIndex.search(q, n),
      getText: (id) => this.textCache.get(id),
      reranker: rerank ? this.getReranker() : undefined,
    };
    // Hydrate a text lookup for candidates. For now, full view text (chunking added in Phase 3).
    const ranked = await hybridSearch(deps, query, { ...options, k, rerank });
    const views = await this.viewStore.getMany(ranked.map((r) => r.id));
    const order = new Map(ranked.map((r, i) => [r.id, i]));
    return views.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
  }
```

- `getText` needs candidate text. Simplest correct approach for this phase: fetch texts for candidate ids inside `hybridSearch` is not possible (it's pure). Instead, pre-load: before calling `hybridSearch`, we don't know candidates yet. Resolve by having `getText` read from `viewStore` synchronously is not possible (async). **Implementation choice:** change `SearchDeps.getText` to `getText: (ids: string[]) => Promise<Map<string,string>>`? That complicates the pure pipeline. Instead, keep `getText` synchronous and back it with a small in-memory cache populated at index time:

In `index()` and `claim()`, after putting the view, also `this.textCache.set(view.id, view.text)`. Add field `private textCache = new Map<string, string>();`. On startup, lazily warm it: add a private method `ensureTextCache()` that, on first search, runs `SELECT id, text FROM views` and fills the map. Call it at the top of `search()`:

```typescript
  private textCacheWarmed = false;
  private warmTextCache() {
    if (this.textCacheWarmed) return;
    const rows = this.db.prepare(`SELECT id, text FROM views`).all() as { id: string; text: string }[];
    for (const r of rows) this.textCache.set(r.id, r.text);
    this.textCacheWarmed = true;
  }
```

Call `this.warmTextCache();` as the first line of `search()`.

- in `dispose()`, add `if (this.reranker) await this.reranker.dispose();`

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test test/relai.test.ts`
Expected: PASS

- [ ] **Step 5: Benchmark with rerank on**

Add a `rerank` toggle to `bench/run.ts` (read `process.env.BENCH_RERANK`); run `BENCH_RERANK=1 bun run bench` and compare to Phase 1. Record numbers.

- [ ] **Step 6: Commit**

```bash
git add src/relai.ts src/types.ts test/relai.test.ts bench/run.ts
git commit -m "feat: wire reranking pipeline into Relai.search with options

bench (rerank on): P@5 ...→..., F1@5 ...→... vs Phase 1"
```

---

## Phase 3 — Chunking (decide empirically; relai's data is mixed-length).

### Task 11: Code-fence-aware chunker (pure function)

**Files:**
- Create: `src/chunker.ts`
- Modify: `src/types.ts` (`Chunk`)
- Test: `test/chunker.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export type Chunk = { text: string; pos: number };
  export function chunkText(text: string, opts?: { maxChars?: number }): Chunk[];
  ```
  Default `maxChars = 3600` (≈900 tokens at ~4 chars/token). Splits on paragraph (`\n\n`) then sentence boundaries; never splits inside a fenced code block (```` ``` ````). Short text → single chunk at pos 0.

- [ ] **Step 1: Add type**

```typescript
export type Chunk = { text: string; pos: number };
```

- [ ] **Step 2: Write the failing test**

```typescript
// test/chunker.test.ts
import { describe, test, expect } from "bun:test";
import { chunkText } from "../src/chunker.js";

describe("chunkText", () => {
  test("short text → one chunk at pos 0", () => {
    const chunks = chunkText("hello world");
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toEqual({ text: "hello world", pos: 0 });
  });

  test("long text splits into multiple chunks under the limit", () => {
    const para = "Sentence about the topic. ".repeat(40); // ~1040 chars
    const text = (para + "\n\n").repeat(8); // ~8.3k chars
    const chunks = chunkText(text, { maxChars: 1500 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(1600);
  });

  test("does not split inside a fenced code block", () => {
    const code = "```\n" + "const x = 1;\n".repeat(50) + "```";
    const text = "Intro paragraph.\n\n" + code + "\n\nOutro paragraph.";
    const chunks = chunkText(text, { maxChars: 200 });
    const codeChunk = chunks.find((c) => c.text.includes("const x = 1;"));
    // The whole code block stays inside a single chunk.
    expect(codeChunk?.text.match(/const x = 1;/g)?.length).toBe(50);
  });

  test("pos is the character offset into the original text", () => {
    const text = "AAAA\n\nBBBB";
    const chunks = chunkText(text, { maxChars: 4 });
    expect(chunks[1]?.pos).toBe(text.indexOf("BBBB"));
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/chunker.test.ts`
Expected: FAIL — `Cannot find module '../src/chunker.js'`

- [ ] **Step 4: Write implementation**

```typescript
// src/chunker.ts
import type { Chunk } from "./types.js";

// Split into atomic blocks: fenced code blocks stay whole; prose splits on blank lines.
function toBlocks(text: string): { text: string; pos: number }[] {
  const blocks: { text: string; pos: number }[] = [];
  const fence = /```[\s\S]*?```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = fence.exec(text)) !== null) {
    if (m.index > last) {
      for (const b of splitProse(text.slice(last, m.index), last)) blocks.push(b);
    }
    blocks.push({ text: m[0], pos: m.index });
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    for (const b of splitProse(text.slice(last), last)) blocks.push(b);
  }
  return blocks;
}

function splitProse(slice: string, base: number): { text: string; pos: number }[] {
  const out: { text: string; pos: number }[] = [];
  const re = /\n\n+/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(slice)) !== null) {
    const piece = slice.slice(last, m.index);
    if (piece.trim()) out.push({ text: piece, pos: base + last });
    last = m.index + m[0].length;
  }
  const tail = slice.slice(last);
  if (tail.trim()) out.push({ text: tail, pos: base + last });
  return out;
}

export function chunkText(text: string, opts?: { maxChars?: number }): Chunk[] {
  const maxChars = opts?.maxChars ?? 3600;
  if (text.length <= maxChars) return [{ text, pos: 0 }];

  const blocks = toBlocks(text);
  const chunks: Chunk[] = [];
  let buf = "";
  let bufPos = -1;

  for (const block of blocks) {
    if (buf === "") {
      buf = block.text;
      bufPos = block.pos;
    } else if (buf.length + 2 + block.text.length <= maxChars) {
      buf += "\n\n" + block.text;
    } else {
      chunks.push({ text: buf, pos: bufPos });
      buf = block.text;
      bufPos = block.pos;
    }
  }
  if (buf !== "") chunks.push({ text: buf, pos: bufPos });
  return chunks.length > 0 ? chunks : [{ text, pos: 0 }];
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/chunker.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/chunker.ts src/types.ts test/chunker.test.ts
git commit -m "feat: add code-fence-aware text chunker"
```

### Task 12: Chunk store + multi-chunk indexing

**Files:**
- Create: `src/chunk-store.ts`
- Modify: `src/types.ts` (`ChunkStore`), `src/embedder.ts` (`embedMany`), `src/relai.ts` (chunk on index), `src/search-pipeline.ts` (map chunk id → view id)
- Test: `test/chunk-store.test.ts`, `test/relai.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export interface ChunkStore {
    putChunks(viewId: string, chunks: Chunk[]): void; // replaces all chunks for a view
    removeByView(viewId: string): void;
    getText(chunkId: string): string | undefined;
    allChunkTexts(): { id: string; viewId: string; text: string }[];
  }
  export function createChunkStore(db: Database): ChunkStore;
  ```
  Chunk id format: `${viewId}#${index}`. Helper `viewIdOf(chunkId)` strips `#<n>`.

- [ ] **Step 1: Add interface + helper to types.ts**

```typescript
export interface ChunkStore {
  putChunks(viewId: string, chunks: Chunk[]): void;
  removeByView(viewId: string): void;
  getText(chunkId: string): string | undefined;
  allChunkTexts(): { id: string; viewId: string; text: string }[];
}
```

Add a tiny pure helper (new file `src/ids.ts` or top of `chunk-store.ts`):

```typescript
export function viewIdOf(chunkId: string): string {
  const hash = chunkId.lastIndexOf("#");
  return hash === -1 ? chunkId : chunkId.slice(0, hash);
}
export function chunkId(viewId: string, index: number): string {
  return `${viewId}#${index}`;
}
```

- [ ] **Step 2: Write the failing test for chunk-store**

```typescript
// test/chunk-store.test.ts
import { describe, test, expect, beforeEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createChunkStore } from "../src/chunk-store.js";

let db: Database;
beforeEach(() => { db = openDatabase(":memory:"); });

describe("ChunkStore", () => {
  test("putChunks then getText by chunk id", () => {
    const cs = createChunkStore(db);
    cs.putChunks("view:docs:1", [{ text: "alpha", pos: 0 }, { text: "beta", pos: 5 }]);
    expect(cs.getText("view:docs:1#0")).toBe("alpha");
    expect(cs.getText("view:docs:1#1")).toBe("beta");
  });

  test("putChunks replaces previous chunks for a view", () => {
    const cs = createChunkStore(db);
    cs.putChunks("view:docs:1", [{ text: "old", pos: 0 }]);
    cs.putChunks("view:docs:1", [{ text: "new", pos: 0 }]);
    expect(cs.getText("view:docs:1#0")).toBe("new");
  });

  test("removeByView clears all chunks", () => {
    const cs = createChunkStore(db);
    cs.putChunks("view:docs:1", [{ text: "x", pos: 0 }, { text: "y", pos: 2 }]);
    cs.removeByView("view:docs:1");
    expect(cs.getText("view:docs:1#0")).toBeUndefined();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/chunk-store.test.ts`
Expected: FAIL — `Cannot find module '../src/chunk-store.js'`

- [ ] **Step 4: Write chunk-store implementation**

```typescript
// src/chunk-store.ts
import type { Database } from "./db.js";
import type { Chunk, ChunkStore } from "./types.js";
import { chunkId } from "./ids.js";

export function createChunkStore(db: Database): ChunkStore {
  db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      view_id TEXT NOT NULL,
      pos INTEGER NOT NULL,
      text TEXT NOT NULL
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_view ON chunks(view_id)`);

  const removeStmt = db.prepare(`DELETE FROM chunks WHERE view_id = ?`);
  const insertStmt = db.prepare(
    `INSERT INTO chunks (id, view_id, pos, text) VALUES (?, ?, ?, ?)`
  );
  const getStmt = db.prepare(`SELECT text FROM chunks WHERE id = ?`);

  return {
    putChunks(viewId: string, chunks: Chunk[]) {
      removeStmt.run(viewId);
      chunks.forEach((c, i) => insertStmt.run(chunkId(viewId, i), viewId, c.pos, c.text));
    },
    removeByView(viewId: string) {
      removeStmt.run(viewId);
    },
    getText(id: string) {
      const row = getStmt.get(id) as { text: string } | undefined;
      return row?.text;
    },
    allChunkTexts() {
      return db.prepare(`SELECT id, view_id, text FROM chunks`).all() as {
        id: string; view_id: string; text: string;
      }[] as any;
    },
  };
}
```

(Fix the `allChunkTexts` mapping to rename `view_id`→`viewId`:)

```typescript
    allChunkTexts() {
      const rows = db.prepare(`SELECT id, view_id, text FROM chunks`).all() as
        { id: string; view_id: string; text: string }[];
      return rows.map((r) => ({ id: r.id, viewId: r.view_id, text: r.text }));
    },
```

- [ ] **Step 5: Run chunk-store test**

Run: `bun test test/chunk-store.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 6: Add embedMany to embedder**

In `src/types.ts` add to `Embedder`: `embedMany(texts: string[]): Promise<number[][]>;`
In `src/embedder.ts` add to the returned object:

```typescript
    async embedMany(texts: string[]): Promise<number[][]> {
      const out: number[][] = [];
      for (const t of texts) out.push(await embedText(formatDocForEmbedding(t)));
      return out;
    },
```

- [ ] **Step 7: Write failing test for chunked indexing in relai**

```typescript
// add to test/relai.test.ts
test("a long view is chunked and a query matching only a late section still retrieves it", async () => {
  const longText =
    "Intro about onboarding.\n\n".repeat(20) +
    "The secret passphrase is XYZZY-PLUGH.\n\n" +
    "Closing remarks about offboarding.\n\n".repeat(20);
  await relai.index({ source: "docs", remoteId: "long", text: longText });
  const results = await relai.search("XYZZY-PLUGH passphrase", 5, { rerank: false });
  expect(results[0]?.id).toBe("view:docs:long");
});
```

- [ ] **Step 8: Run to verify it fails (or is flaky on whole-doc embedding)**

Run: `bun test test/relai.test.ts`
Expected: FAIL or unstable — whole-doc embedding dilutes the late section. (FTS may still catch it; if the test passes by luck, strengthen it by making the passphrase a semantic-only paraphrase. The goal is the chunked vector path.)

- [ ] **Step 9: Update Relai.index() / claim() to chunk + embed per chunk**

In `src/relai.ts`:
- import `createChunkStore`, `chunkText`, `viewIdOf`, `type ChunkStore`.
- add field `private chunkStore: ChunkStore;` and construct `this.chunkStore = createChunkStore(this.db);`
- rewrite the embed/upsert portion of `index()`:

```typescript
  async index(input: ViewInput): Promise<View> {
    const view: View = { ...input, id: `view:${input.source}:${input.remoteId}` };
    await this.viewStore.put(view);
    this.ftsIndex.upsert(view.id, view.text);
    await this.indexChunks(view.id, view.text);
    return view;
  }

  private async indexChunks(viewId: string, text: string) {
    const chunks = chunkText(text);
    this.chunkStore.putChunks(viewId, chunks);
    // Remove any stale chunk vectors for this view, then add fresh ones.
    // (Vectors keyed by chunk id; a re-index with fewer chunks must not leave orphans.)
    await this.vectorIndex.remove(viewId); // legacy single-vector id, if present
    const vectors = await this.embedder.embedMany(chunks.map((c) => c.text));
    for (let i = 0; i < chunks.length; i++) {
      await this.vectorIndex.upsert(`${viewId}#${i}`, vectors[i]!);
    }
    this.textCache.set(viewId, text);
  }
```

- in `claim()`, replace the embed/upsert lines with `await this.indexChunks(claimView.id, text);`
- update the search hydration to collapse chunk ids → view ids. In `search()`, after getting `ranked` (chunk ids), dedup by view id keeping best rank:

```typescript
    const seen = new Set<string>();
    const viewOrder: string[] = [];
    for (const r of ranked) {
      const vid = viewIdOf(r.id);
      if (seen.has(vid)) continue;
      seen.add(vid);
      viewOrder.push(vid);
    }
    const views = await this.viewStore.getMany(viewOrder);
    const order = new Map(viewOrder.map((id, i) => [id, i]));
    return views.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
```

- update `SearchDeps.getText` to read chunk text: `getText: (id) => this.chunkStore.getText(id) ?? this.textCache.get(viewIdOf(id))`.
- the FTS index is per-view (not per-chunk), so `ftsSearch` returns view ids while vector returns chunk ids. RRF treats them as different ids — **fix:** normalize FTS ids to a synthetic chunk id `#0`, OR (simpler, recommended) normalize **both** sides to view ids before fusion. Add a `mapToViewId` step in `hybridSearch`:

In `src/search-pipeline.ts`, before fusion, collapse each list to best-per-view using `viewIdOf`. Add a helper param `idToGroup?: (id: string) => string` to `SearchDeps`; default identity. In Relai, pass `viewIdOf`. Then fuse on group ids, and rerank using the best chunk text per view.

  Concretely, add to `hybridSearch`:

```typescript
  const group = deps.idToGroup ?? ((id: string) => id);
  const collapse = (list: RankedItem[]) => {
    const best = new Map<string, RankedItem>();
    for (const it of list) {
      const g = group(it.id);
      const cur = best.get(g);
      if (!cur || it.score > cur.score) best.set(g, { id: it.id, score: it.score });
    }
    return Array.from(best.values()).sort((a, b) => b.score - a.score);
  };
  const vecCollapsed = collapse(vecHits);   // ids are best chunk per view
  const ftsCollapsed = collapse(ftsHits);   // ids are view ids
```

  But fusion must align on the same id space. Re-key both to `group(id)` for fusion, while remembering the representative chunk id for rerank text:

```typescript
  const repChunk = new Map<string, string>(); // viewId -> best chunk id (for rerank text)
  for (const it of vecCollapsed) repChunk.set(group(it.id), it.id);
  const toGrouped = (l: RankedItem[]) => l.map((it) => ({ id: group(it.id), score: it.score }));
  const fused = reciprocalRankFusion([toGrouped(vecCollapsed), toGrouped(ftsCollapsed)]);
  const candidates = fused.slice(0, candidateLimit);
  const texts = candidates.map((c) => deps.getText(repChunk.get(c.id) ?? c.id) ?? "");
```

  Add `idToGroup?: (id: string) => string;` to `SearchDeps` type. Update tests in Task 9 if they break (they use plain ids → identity grouping, still pass).

- [ ] **Step 10: Run all tests**

Run: `bun test`
Expected: PASS (chunker, chunk-store, pipeline, relai including the new long-doc test)

- [ ] **Step 11: Benchmark — is chunking net-positive?**

Run: `bun run bench` and `BENCH_RERANK=1 bun run bench`. Compare to Phase 2.
**Decision gate:** relai's data is mixed-length. If chunking improves F1@5 on long docs without hurting short records, keep it. If it regresses short-record precision (over-splitting tiny views), gate chunking behind a length threshold: in `indexChunks`, if `text.length <= maxChars` you already get a single chunk — that's the desired no-op for short views, so a regression would point at fusion id-grouping; debug there. Record the decision and numbers in the commit.

- [ ] **Step 12: Commit**

```bash
git add src/chunk-store.ts src/ids.ts src/types.ts src/embedder.ts src/relai.ts src/search-pipeline.ts test/chunk-store.test.ts test/relai.test.ts
git commit -m "feat: per-chunk indexing and retrieval with view-level collapse

bench: F1@5 ...→... (chunking decision: kept/threshold-gated)"
```

---

## Phase 4 — Typed query expansion + strong-signal short-circuit.

### Task 13: Query expander module

**Files:**
- Create: `src/query-expansion.ts`
- Modify: `src/types.ts` (`QueryExpander`, `ExpandedQuery`)
- Test: `test/query-expansion.test.ts`

**Interfaces:**
- Produces:
  ```typescript
  export type ExpandedQuery = { type: "lex" | "vec" | "hyde"; query: string };
  export interface QueryExpander {
    expand(query: string): Promise<ExpandedQuery[]>;
    dispose(): Promise<void>;
  }
  export function createQueryExpander(opts?: { model?: string; cacheDir?: string }): QueryExpander;
  ```
  `lex` → keyword reformulation (routed to FTS); `vec` → semantic paraphrase (routed to vector); `hyde` → a 1-2 sentence hypothetical answer (routed to vector). Returns `[]` on model/parse failure (graceful — original query still runs).

- [ ] **Step 1: Add types**

```typescript
export type ExpandedQuery = { type: "lex" | "vec" | "hyde"; query: string };
export interface QueryExpander {
  expand(query: string): Promise<ExpandedQuery[]>;
  dispose(): Promise<void>;
}
```

- [ ] **Step 2: Write the failing test (integration — loads generation model)**

```typescript
// test/query-expansion.test.ts
import { describe, test, expect, afterAll } from "bun:test";
import { createQueryExpander } from "../src/query-expansion.js";

const expander = createQueryExpander();
afterAll(() => expander.dispose());

describe("QueryExpander (integration — downloads generation model)", () => {
  test("produces typed variants for a query", async () => {
    const out = await expander.expand("how to cancel my subscription");
    expect(out.length).toBeGreaterThan(0);
    for (const e of out) {
      expect(["lex", "vec", "hyde"]).toContain(e.type);
      expect(e.query.length).toBeGreaterThan(0);
    }
  }, 120_000);
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/query-expansion.test.ts`
Expected: FAIL — `Cannot find module '../src/query-expansion.js'`

- [ ] **Step 4: Write implementation**

```typescript
// src/query-expansion.ts
import { getLlama, resolveModelFile, type Llama, type LlamaModel, type LlamaContext } from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import type { QueryExpander, ExpandedQuery } from "./types.js";

const DEFAULT_GENERATE_MODEL =
  process.env.RELAI_GENERATE_MODEL ??
  "hf:ggml-org/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf";

const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, "relai", "models")
  : join(homedir(), ".cache", "relai", "models");

const SYSTEM = `You expand a search query into alternative queries to improve retrieval.
Return ONLY a JSON array. Each element: {"type":"lex|vec|hyde","query":"..."}.
- "lex": a keyword-only reformulation (synonyms, key nouns).
- "vec": a natural-language paraphrase of the intent.
- "hyde": a one-sentence hypothetical answer document.
Return 3 elements total, one of each type. No prose, no markdown.`;

export function createQueryExpander(opts?: { model?: string; cacheDir?: string }): QueryExpander {
  const modelUri = opts?.model ?? DEFAULT_GENERATE_MODEL;
  const cacheDir = opts?.cacheDir ?? MODEL_CACHE_DIR;

  let llama: Llama | null = null;
  let model: LlamaModel | null = null;
  let ctx: LlamaContext | null = null;
  let loadPromise: Promise<void> | null = null;

  async function ensureLoaded() {
    if (ctx) return;
    if (loadPromise) return loadPromise;
    loadPromise = (async () => {
      if (!existsSync(cacheDir)) mkdirSync(cacheDir, { recursive: true });
      const modelPath = await resolveModelFile(modelUri, cacheDir);
      llama = await getLlama();
      model = await llama.loadModel({ modelPath });
      ctx = await model.createContext();
    })();
    return loadPromise;
  }

  function parse(raw: string): ExpandedQuery[] {
    try {
      const start = raw.indexOf("[");
      const end = raw.lastIndexOf("]");
      if (start === -1 || end === -1) return [];
      const arr = JSON.parse(raw.slice(start, end + 1));
      if (!Array.isArray(arr)) return [];
      return arr
        .filter((e) => e && typeof e.query === "string" && ["lex", "vec", "hyde"].includes(e.type))
        .map((e) => ({ type: e.type, query: e.query.trim() }))
        .filter((e) => e.query.length > 0);
    } catch {
      return [];
    }
  }

  return {
    async expand(query: string): Promise<ExpandedQuery[]> {
      try {
        await ensureLoaded();
        const { LlamaChatSession } = await import("node-llama-cpp");
        const session = new LlamaChatSession({ contextSequence: ctx!.getSequence() });
        const raw = await session.prompt(`${SYSTEM}\n\nQuery: ${query}`, { temperature: 0.2, maxTokens: 256 });
        return parse(raw);
      } catch {
        return [];
      }
    },
    async dispose() {
      if (ctx) { await ctx.dispose(); ctx = null; }
      if (model) { await model.dispose(); model = null; }
      if (llama) { await llama.dispose(); llama = null; }
      loadPromise = null;
    },
  };
}
```

> Note for implementer: verify `LlamaChatSession` / `ctx.getSequence()` against installed `node-llama-cpp@^3.17.1` (same check as Task 7). The `parse()` JSON-extraction is the resilience layer — small models stray from format, so `expand()` must never throw; returning `[]` degrades to original-query-only search.

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/query-expansion.test.ts`
Expected: PASS (downloads model first run)

- [ ] **Step 6: Commit**

```bash
git add src/query-expansion.ts src/types.ts test/query-expansion.test.ts
git commit -m "feat: add typed query expansion (lex/vec/hyde)"
```

### Task 14: Route expansions through the pipeline + strong-signal short-circuit

**Files:**
- Modify: `src/search-pipeline.ts`
- Modify: `src/relai.ts`
- Modify: `src/types.ts` (`SearchOptions.expand`)
- Test: `test/search-pipeline.test.ts`

**Interfaces:**
- Consumes: `QueryExpander` (Task 13).
- Produces: `hybridSearch` accepts `deps.expander?` and `options.expand?: boolean` (default true when expander present). Original query lists get RRF weight `2.0`; expansion lists `1.0`. `lex` variants route to `ftsSearch`; `vec`/`hyde` route to `vectorSearch`. Strong-signal short-circuit: if top FTS score ≥ 0.85 and gap to runner-up ≥ 0.15, skip expansion.

- [ ] **Step 1: Add option type**

In `src/types.ts` extend `SearchOptions`: add `expand?: boolean;`

- [ ] **Step 2: Write failing test (faked expander, no model)**

```typescript
// add to test/search-pipeline.test.ts
test("expansion adds routed lists and weights original higher", async () => {
  const calls: { fts: string[]; vec: number } = { fts: [], vec: 0 };
  const deps = {
    embedQuery: async () => [1, 0, 0],
    vectorSearch: async () => { calls.vec++; return [{ id: "a", score: 0.5 }]; },
    ftsSearch: (q: string) => { calls.fts.push(q); return [{ id: "b", score: 0.5 }]; },
    getText: (id: string) => id,
    expander: {
      expand: async () => [
        { type: "lex" as const, query: "keyword variant" },
        { type: "vec" as const, query: "semantic variant" },
      ],
      dispose: async () => {},
    },
  };
  const out = await hybridSearch(deps, "orig", { rerank: false, expand: true, k: 5 });
  expect(calls.fts).toContain("orig");          // original FTS
  expect(calls.fts).toContain("keyword variant"); // lex routed to FTS
  expect(calls.vec).toBeGreaterThanOrEqual(2);    // original + vec/hyde routed to vector
  expect(out.length).toBeGreaterThan(0);
});

test("strong FTS signal skips expansion", async () => {
  let expandCalled = false;
  const deps = {
    embedQuery: async () => [1, 0, 0],
    vectorSearch: async () => [{ id: "a", score: 0.5 }],
    ftsSearch: () => [{ id: "b", score: 0.95 }, { id: "c", score: 0.5 }], // gap 0.45 ≥ 0.15, top ≥ 0.85
    getText: (id: string) => id,
    expander: { expand: async () => { expandCalled = true; return []; }, dispose: async () => {} },
  };
  await hybridSearch(deps, "exact", { rerank: false, expand: true });
  expect(expandCalled).toBe(false);
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `bun test test/search-pipeline.test.ts`
Expected: FAIL — `deps.expander` not used; expansion lists not added.

- [ ] **Step 4: Update hybridSearch**

Add to `SearchDeps`: `expander?: QueryExpander;`. Rework the retrieval section:

```typescript
  const STRONG_MIN = 0.85, STRONG_GAP = 0.15;

  const vector = await deps.embedQuery(query);
  const baseVec = await deps.vectorSearch(vector, width);
  const baseFts = deps.ftsSearch(query, width);

  const top = baseFts[0]?.score ?? 0;
  const second = baseFts[1]?.score ?? 0;
  const strongSignal = baseFts.length > 0 && top >= STRONG_MIN && (top - second) >= STRONG_GAP;

  const doExpand = (options.expand ?? true) && !!deps.expander && !strongSignal;
  const expansions = doExpand ? await deps.expander!.expand(query) : [];

  // weighted lists: original first (weight 2.0), expansions after (1.0)
  const lists: RankedItem[][] = [baseVec, baseFts];
  const weights: number[] = [2.0, 2.0];
  for (const e of expansions) {
    if (e.type === "lex") {
      lists.push(deps.ftsSearch(e.query, width));
    } else {
      const ev = await deps.embedQuery(e.query);
      lists.push(await deps.vectorSearch(ev, width));
    }
    weights.push(1.0);
  }
```

Then collapse each list with `group` (as in Task 12) before fusion, pass `weights` into `reciprocalRankFusion(grouped, weights)`. Keep the rest (rerank/blend) unchanged.

- [ ] **Step 5: Run test to verify it passes**

Run: `bun test test/search-pipeline.test.ts`
Expected: PASS (all pipeline tests, including the two new ones)

- [ ] **Step 6: Wire expander into Relai**

In `src/relai.ts`: add `createQueryExpander` import, `RelaiConfig.expand?: boolean` / `generateModel?: string`, lazy `getExpander()` (mirror `getReranker()`), pass `expander: options.expand ?? this.expandEnabled ? this.getExpander() : undefined` into `deps`, and dispose it in `dispose()`.

- [ ] **Step 7: Run full suite + benchmark**

Run: `bun test` then `BENCH_RERANK=1 bun run bench`.
Expected: all tests pass; record final P@5/R@5/F1@5 with the full pipeline vs every prior phase.

- [ ] **Step 8: Commit**

```bash
git add src/search-pipeline.ts src/relai.ts src/types.ts test/search-pipeline.test.ts
git commit -m "feat: route typed query expansion through pipeline + strong-signal short-circuit

bench (full pipeline): P@5 ...→..., F1@5 ...→... vs vector-only baseline"
```

---

## Phase 5 — CLI surface + docs.

### Task 15: CLI flags and bench command

**Files:**
- Modify: `src/cli/relai.ts`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Relai.search(query, k, options)`.
- Produces: `relai search <query> [-k N] [--no-rerank] [--no-expand] [--explain]` and `relai bench`.

- [ ] **Step 1: Read the current CLI**

Read `src/cli/relai.ts` to match its arg-parsing style (it currently dispatches `pull`/`index`/`search`/`claim`/`related`).

- [ ] **Step 2: Add flags to the search command**

Parse `-k <n>` (default 5), `--no-rerank` → `{ rerank: false }`, `--no-expand` → `{ expand: false }`. Pass options into `relai.search(query, k, options)`. Keep output format identical to today (print id + a snippet of text per result), one result per line.

- [ ] **Step 3: Add a `bench` subcommand**

`case "bench":` → dynamically import and run `bench/run.ts`'s `main`, or `Bun.spawn(["bun", "run", "bench/run.ts"])`. Keep it thin.

- [ ] **Step 4: Manual smoke test**

```bash
bun run src/cli/relai.ts index --source docs --remote-id 1 --text "csv export from reports page"
bun run src/cli/relai.ts search "export csv" -k 3 --no-rerank
```
Expected: prints `view:docs:1` among results.

- [ ] **Step 5: Update README**

Document the new pipeline (BM25 + vector + RRF + rerank + expansion), the new model downloads (`relai pull` should pull rerank + generation models too — update the `pull` command to accept `--all`), and the new search flags. Add a short "Search architecture" section.

- [ ] **Step 6: Commit**

```bash
git add src/cli/relai.ts README.md
git commit -m "feat: expose hybrid-search flags and bench command in CLI; document pipeline"
```

---

## Self-Review Notes

- **Spec coverage:** qmd's four stages map to Phases 1 (BM25+RRF), 2 (rerank+blend), 3 (chunking), 4 (typed expansion + strong-signal). qmd's benchmark harness → Phase 0. qmd's "rerank the best chunk, not the full body" → Task 12 step 9 (`repChunk` text selection). Claims-as-views preserved: every `claim()` change re-uses `indexChunks`/`ftsIndex.upsert` so claim views stay first-class searchable.
- **Mixed-length data (user's answer):** handled empirically — chunker is a no-op for short text (≤ maxChars → single chunk), and Task 12 step 11 is an explicit decision gate measured by the benchmark.
- **Max-quality priority (user's answer):** all four stages included; rerank + expansion are on by default, flag-gated off only for latency-sensitive callers.
- **Graceful degradation:** reranker and expander are lazy/optional; failures return `[]` and search falls back to RRF. Verified by the `rerank:false` / strong-signal / parse-failure paths.
- **Open verification item for implementer:** exact `node-llama-cpp@^3.17.1` API names for ranking context (`createRankingContext`/`rankAll`) and chat session (`LlamaChatSession`/`getSequence`). Flagged inline in Tasks 7 and 13 — confirm against the installed `.d.ts` before writing, adjust call sites only, interfaces stay fixed.
```
