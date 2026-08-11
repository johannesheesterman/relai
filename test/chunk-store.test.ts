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
