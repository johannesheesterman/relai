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
    const ids = out.map((r) => r.id);
    // reranker strongly favors c, promoting it above the lower-reranked doc a
    expect(ids.indexOf("c")).toBeLessThan(ids.indexOf("a"));
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
