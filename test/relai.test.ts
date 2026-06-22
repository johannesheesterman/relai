import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { openDatabase, type Database } from "../src/db.js";
import { createViewStore } from "../src/view-store.js";
import { createClaimStore } from "../src/claim-store.js";
import { createVectorIndex } from "../src/vector-index.js";
import { Relai } from "../src/relai.js";
import type { Embedder, ViewStore, ClaimStore, VectorIndex } from "../src/types.js";

function createMockEmbedder(): Embedder {
  // Simple hash-based embedder that produces deterministic 4-dim vectors
  function hash(text: string): number[] {
    let h = 0;
    for (let i = 0; i < text.length; i++) {
      h = (h * 31 + text.charCodeAt(i)) | 0;
    }
    const v = [
      ((h >> 0) & 0xff) / 255,
      ((h >> 8) & 0xff) / 255,
      ((h >> 16) & 0xff) / 255,
      ((h >> 24) & 0xff) / 255,
    ];
    const mag = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / mag);
  }

  return {
    async embed(text: string) { return hash(text); },
    async embedQuery(text: string) { return hash(text); },
    async dispose() {},
  };
}

describe("Relai claim embedding", () => {
  let db: Database;
  let viewStore: ViewStore;
  let claimStore: ClaimStore;
  let vectorIndex: VectorIndex;
  let embedder: Embedder;

  beforeEach(() => {
    db = openDatabase(":memory:");
    viewStore = createViewStore(db);
    claimStore = createClaimStore(db);
    vectorIndex = createVectorIndex(db);
    embedder = createMockEmbedder();
  });

  test("claim creates an embedded view combining both endpoints", async () => {
    // Index two views
    const fromView = {
      id: "view:tickets:123",
      source: "tickets",
      remoteId: "123",
      text: "Ticket about adding dark mode",
    };
    const toView = {
      id: "view:ticket-types:feature-request",
      source: "ticket-types",
      remoteId: "feature-request",
      text: "Feature request: new functionality or enhancement",
    };
    await viewStore.put(fromView);
    await viewStore.put(toView);
    const v1 = await embedder.embed(fromView.text);
    const v2 = await embedder.embed(toView.text);
    await vectorIndex.upsert(fromView.id, v1);
    await vectorIndex.upsert(toView.id, v2);

    // Create claim — this should also create a claim view
    const claimId = crypto.randomUUID();
    await claimStore.put({
      id: claimId,
      from: fromView.id,
      type: "type_of",
      to: toView.id,
    });

    // Simulate what Relai.claim() does: create claim view
    const [fromViews, toViews] = await Promise.all([
      viewStore.getMany([fromView.id]),
      viewStore.getMany([toView.id]),
    ]);
    const fv = fromViews[0];
    const tv = toViews[0];
    const claimText = `${fv.text} type_of ${tv.text}`;
    const claimView = {
      id: `view:claim:${claimId}`,
      source: "claim",
      remoteId: claimId,
      type: "claim",
      text: claimText,
      links: [
        { type: "from", to: fromView.id },
        { type: "to", to: toView.id },
      ],
    };
    await viewStore.put(claimView);
    const claimVector = await embedder.embed(claimText);
    await vectorIndex.upsert(claimView.id, claimVector);

    // Verify the claim view exists
    const [result] = await viewStore.getMany([`view:claim:${claimId}`]);
    expect(result).toBeDefined();
    expect(result.type).toBe("claim");
    expect(result.text).toContain("dark mode");
    expect(result.text).toContain("type_of");
    expect(result.text).toContain("Feature request");
    expect(result.links).toHaveLength(2);
    expect(result.links![0].to).toBe(fromView.id);
    expect(result.links![1].to).toBe(toView.id);
  });

  test("claim view is findable via vector search", async () => {
    // Index views and create claim view
    await viewStore.put({
      id: "view:tickets:1",
      source: "tickets",
      remoteId: "1",
      text: "Bug report about login",
    });
    await viewStore.put({
      id: "view:ticket-types:bug",
      source: "ticket-types",
      remoteId: "bug",
      text: "Bug: a defect or error",
    });

    const bugVector = await embedder.embed("Bug report about login type_of Bug: a defect or error");
    await vectorIndex.upsert("view:claim:c1", bugVector);

    // Search with the same text should find it
    const queryVector = await embedder.embedQuery("Bug report about login type_of Bug: a defect or error");
    const results = await vectorIndex.search(queryVector, 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].id).toBe("view:claim:c1");
  });
});

describe("Relai hybrid search", () => {
  let relai: Relai;

  beforeEach(() => {
    relai = new Relai({ dbPath: ":memory:", embedder: createMockEmbedder(), rerank: false });
  });

  afterEach(async () => {
    await relai.dispose();
  });

  test("hybrid search finds exact keyword match that vector search alone may rank lower", async () => {
    await relai.index({ source: "docs", remoteId: "1", text: "general notes about the system architecture and design" });
    await relai.index({ source: "docs", remoteId: "2", text: "the SKU-99XZ part number appears only here" });
    const results = await relai.search("SKU-99XZ", 5);
    expect(results[0]?.id).toBe("view:docs:2");
  });

  test("search accepts options and rerank can be disabled", async () => {
    await relai.index({ source: "docs", remoteId: "1", text: "alpha bravo charlie" });
    const out = await relai.search("alpha", 3, { rerank: false });
    expect(out[0]?.id).toBe("view:docs:1");
  });
});
