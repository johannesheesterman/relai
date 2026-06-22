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
