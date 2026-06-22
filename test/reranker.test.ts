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
