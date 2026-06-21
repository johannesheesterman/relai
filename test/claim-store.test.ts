import { describe, test, expect, beforeEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createClaimStore } from "../src/claim-store.js";
import { ref, type ClaimStore } from "../src/types.js";

let db: Database;
let store: ClaimStore;

beforeEach(() => {
  db = openDatabase(":memory:");
  store = createClaimStore(db);
});

describe("ClaimStore", () => {
  test("put and match by subject", async () => {
    await store.put({
      subject: "v1",
      predicate: "mentions",
      object: ref("v2"),
      createdBy: "agent:test",
    });
    const results = await store.match({ subject: "v1" });
    expect(results).toHaveLength(1);
    expect(results[0].predicate).toBe("mentions");
    expect(results[0].object).toEqual(ref("v2"));
    expect(results[0].createdBy).toBe("agent:test");
  });

  test("match by predicate and object", async () => {
    await store.put({ subject: "v1", predicate: "mentions", object: ref("v2") });
    const results = await store.match({
      predicate: "mentions",
      object: ref("v2"),
    });
    expect(results).toHaveLength(1);
    expect(results[0].subject).toBe("v1");
  });

  test("stores literal objects", async () => {
    await store.put({ subject: "v1", predicate: "priority", object: "high" });
    const results = await store.match({ subject: "v1", predicate: "priority" });
    expect(results).toHaveLength(1);
    expect(results[0].object).toBe("high");
  });

  test("match returns empty for unknown subject", async () => {
    const results = await store.match({ subject: "unknown" });
    expect(results).toEqual([]);
  });

  test("multiple claims from same subject", async () => {
    await store.put({ subject: "v1", predicate: "mentions", object: ref("v2") });
    await store.put({ subject: "v1", predicate: "related_to", object: ref("v3") });
    const results = await store.match({ subject: "v1" });
    expect(results).toHaveLength(2);
  });

  test("put replaces a claim with the same subject/predicate/object", async () => {
    await store.put({
      subject: "v1",
      predicate: "mentions",
      object: ref("v2"),
      createdBy: "a",
    });
    await store.put({
      subject: "v1",
      predicate: "mentions",
      object: ref("v2"),
      createdBy: "b",
    });
    const results = await store.match({ subject: "v1" });
    expect(results).toHaveLength(1);
    expect(results[0].createdBy).toBe("b");
  });

  test("delete removes matching claims", async () => {
    await store.put({ subject: "v1", predicate: "mentions", object: ref("v2") });
    await store.put({ subject: "v1", predicate: "related_to", object: ref("v3") });

    await store.delete({ subject: "v1", predicate: "mentions" });

    const results = await store.match({ subject: "v1" });
    expect(results).toHaveLength(1);
    expect(results[0].predicate).toBe("related_to");
  });

  test("delete without a pattern is refused", async () => {
    await store.put({ subject: "v1", predicate: "mentions", object: ref("v2") });
    await expect(store.delete({})).rejects.toThrow();
    expect(await store.match({ subject: "v1" })).toHaveLength(1);
  });
});
