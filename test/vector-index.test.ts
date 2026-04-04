import { describe, test, expect, beforeEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createVectorIndex } from "../src/vector-index.js";
import type { VectorIndex } from "../src/types.js";

let db: Database;
let index: VectorIndex;

beforeEach(() => {
  db = openDatabase(":memory:");
  index = createVectorIndex(db);
});

describe("VectorIndex", () => {
  const dim = 4;

  function normalize(v: number[]): number[] {
    const mag = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return v.map((x) => x / mag);
  }

  test("upsert and search returns results", async () => {
    const v1 = normalize([1, 0, 0, 0]);
    const v2 = normalize([0, 1, 0, 0]);
    await index.upsert("a", v1);
    await index.upsert("b", v2);

    const results = await index.search(v1, 2);
    expect(results).toHaveLength(2);
    expect(results[0].viewId).toBe("a");
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  test("search on empty index returns empty", async () => {
    const results = await index.search([1, 0, 0, 0], 5);
    expect(results).toEqual([]);
  });

  test("upsert replaces existing vector", async () => {
    const v1 = normalize([1, 0, 0, 0]);
    const v2 = normalize([0, 1, 0, 0]);
    await index.upsert("a", v1);
    await index.upsert("a", v2);

    const results = await index.search(v2, 1);
    expect(results).toHaveLength(1);
    expect(results[0].viewId).toBe("a");
    expect(results[0].score).toBeCloseTo(1, 1);
  });

  test("k limits results", async () => {
    for (let i = 0; i < 10; i++) {
      const v = [0, 0, 0, 0];
      v[i % dim] = 1;
      await index.upsert(`v${i}`, normalize(v));
    }
    const results = await index.search(normalize([1, 0, 0, 0]), 3);
    expect(results).toHaveLength(3);
  });
});
