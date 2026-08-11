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

  test("reranker reorders peers outside the top-3 protected band", async () => {
    // Eight candidates so several land outside the rank<=3 protected band
    // (positionAwareBlend pins w=0.75 for the top 3 RRF docs). Within the
    // unprotected band the reranker should be able to lift a doc above a peer
    // that started ahead of it on RRF.
    const ids = ["a", "b", "c", "d", "e", "f", "g", "h"];
    const wide = {
      embedQuery: async () => [1, 0, 0],
      // identical lists so RRF order is exactly a,b,c,d,e,f,g,h
      vectorSearch: async () => ids.map((id, i) => ({ id, score: 1 - i * 0.1 })),
      ftsSearch: () => ids.map((id, i) => ({ id, score: 1 - i * 0.1 })),
      getText: (id: string) => `text for ${id}`,
    };
    // "g" starts at RRF rank 7, "e" at rank 5 — both outside the protected
    // top-3. Give g a strong reranker score and e a weak one.
    const reranker = {
      rank: async (_q: string, docs: string[]) =>
        docs.map((d) => (d.includes("g") ? 0.99 : d.includes("e") ? 0.0 : 0.3)),
      dispose: async () => {},
    };
    const out = await hybridSearch({ ...wide, reranker }, "q", {
      rerank: true,
      k: 8,
      candidateLimit: 8,
    });
    const order = out.map((r) => r.id);
    // The reranker genuinely promoted g above its lower-reranked peer e,
    // despite e having the stronger (lower) RRF rank.
    expect(order.indexOf("g")).toBeLessThan(order.indexOf("e"));
  });

  test("top RRF doc is retained despite reranker disagreement (intended protection)", async () => {
    // positionAwareBlend intentionally floors the top-3 RRF docs at w=0.75
    // (matching qmd) and the #1 RRF hit additionally enjoys the largest
    // 1/rrfRank base, so a confident keyword/vector consensus is not overturned
    // by the reranker. Here "b" appears in BOTH base lists and is the #1 RRF
    // hit; even with the worst possible reranker score (0.0) it must stay ahead
    // of "c", which gets the best reranker score (0.99) but starts at RRF #3.
    const reranker = {
      rank: async (_q: string, docs: string[]) =>
        docs.map((d) => (d.includes("c") ? 0.99 : 0.0)),
      dispose: async () => {},
    };
    const out = await hybridSearch({ ...deps, reranker }, "q", { rerank: true, k: 3 });
    const order = out.map((r) => r.id);
    // b (RRF #1) is protected and stays ranked ahead of c despite the reranker.
    // b blend = 0.75*(1/1) + 0.25*0 = 0.75; c blend = 0.75*(1/3) + 0.25*0.99 ≈ 0.497.
    expect(order.indexOf("b")).toBeLessThan(order.indexOf("c"));
    expect(order[0]).toBe("b");
  });

  test("search survives a failing reranker (graceful degradation -> RRF)", async () => {
    const reranker = {
      rank: async () => {
        throw new Error("model unavailable");
      },
      dispose: async () => {},
    };
    // Must not throw; falls back to the RRF-fused ranking (same as rerank=false).
    const out = await hybridSearch({ ...deps, reranker }, "q", { rerank: true, k: 3 });
    expect(out.map((r) => r.id)).toContain("b"); // b is fused high (both lists)
    expect(out.length).toBeGreaterThan(0);
    expect(out.length).toBeLessThanOrEqual(3);
  });

  test("search survives a failing expander (graceful degradation)", async () => {
    const expander = {
      expand: async () => {
        throw new Error("expansion model unavailable");
      },
      dispose: async () => {},
    };
    const out = await hybridSearch({ ...deps, expander }, "q", {
      rerank: false,
      expand: true,
      k: 3,
    });
    expect(out.map((r) => r.id)).toContain("b");
    expect(out.length).toBeGreaterThan(0);
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
    expect(calls.fts).toContain("orig");            // original FTS
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
});
