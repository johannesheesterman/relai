import { describe, test, expect, beforeEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createViewStore } from "../src/view-store.js";
import type { View, ViewStore } from "../src/types.js";

let db: Database;
let store: ViewStore;

beforeEach(() => {
  db = openDatabase(":memory:");
  store = createViewStore(db);
});

describe("ViewStore", () => {
  test("put and getMany", async () => {
    const view: View = {
      id: "v1",
      source: "files",
      remoteId: "test.md",
      text: "Hello world",
    };
    await store.put(view);
    const results = await store.getMany(["v1"]);
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe("v1");
    expect(results[0].source).toBe("files");
    expect(results[0].remoteId).toBe("test.md");
    expect(results[0].text).toBe("Hello world");
  });

  test("put with links", async () => {
    const view: View = {
      id: "v1",
      source: "files",
      remoteId: "a.md",
      text: "doc a",
      links: [{ type: "references", to: "v2" }],
    };
    await store.put(view);
    const [result] = await store.getMany(["v1"]);
    expect(result.links).toEqual([{ type: "references", to: "v2" }]);
  });

  test("put replaces existing", async () => {
    await store.put({ id: "v1", source: "files", remoteId: "a.md", text: "old" });
    await store.put({ id: "v1", source: "files", remoteId: "a.md", text: "new" });
    const [result] = await store.getMany(["v1"]);
    expect(result.text).toBe("new");
  });

  test("getMany with empty ids returns empty", async () => {
    const results = await store.getMany([]);
    expect(results).toEqual([]);
  });

  test("getMany with missing ids returns partial", async () => {
    await store.put({ id: "v1", source: "files", remoteId: "a.md", text: "hi" });
    const results = await store.getMany(["v1", "v2"]);
    expect(results).toHaveLength(1);
  });

  test("type field is optional", async () => {
    await store.put({
      id: "v1",
      source: "sqlite",
      remoteId: "customers/1",
      type: "customer",
      text: "Customer ACME",
    });
    const [result] = await store.getMany(["v1"]);
    expect(result.type).toBe("customer");
  });
});
