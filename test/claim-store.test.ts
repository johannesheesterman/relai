import { describe, test, expect, beforeEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createClaimStore } from "../src/claim-store.js";
import type { Claim, ClaimStore } from "../src/types.js";

let db: Database;
let store: ClaimStore;

beforeEach(() => {
  db = openDatabase(":memory:");
  store = createClaimStore(db);
});

describe("ClaimStore", () => {
  test("put and from", async () => {
    const claim: Claim = {
      id: "c1",
      from: "v1",
      type: "mentions",
      to: "v2",
      createdBy: "agent:test",
    };
    await store.put(claim);
    const results = await store.from("v1");
    expect(results).toHaveLength(1);
    expect(results[0].type).toBe("mentions");
    expect(results[0].to).toBe("v2");
    expect(results[0].createdBy).toBe("agent:test");
  });

  test("put and to", async () => {
    await store.put({ id: "c1", from: "v1", type: "mentions", to: "v2" });
    const results = await store.to("v2");
    expect(results).toHaveLength(1);
    expect(results[0].from).toBe("v1");
  });

  test("from returns empty for unknown viewId", async () => {
    const results = await store.from("unknown");
    expect(results).toEqual([]);
  });

  test("multiple claims from same view", async () => {
    await store.put({ id: "c1", from: "v1", type: "mentions", to: "v2" });
    await store.put({ id: "c2", from: "v1", type: "related_to", to: "v3" });
    const results = await store.from("v1");
    expect(results).toHaveLength(2);
  });

  test("put replaces existing claim", async () => {
    await store.put({ id: "c1", from: "v1", type: "mentions", to: "v2" });
    await store.put({ id: "c1", from: "v1", type: "same_as", to: "v2" });
    const results = await store.from("v1");
    expect(results).toHaveLength(1);
    expect(results[0].type).toBe("same_as");
  });
});
