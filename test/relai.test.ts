import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { Relai } from "../src/relai.js";
import { createClaimStore } from "../src/claim-store.js";
import { openDatabase, type Database } from "../src/db.js";
import { ref, type Embedder, type VectorIndex } from "../src/types.js";

function createMockEmbedder(): Embedder {
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
    async embed(text: string) {
      return hash(text);
    },
    async embedQuery(text: string) {
      return hash(text);
    },
    async embedMany(texts: string[]) {
      return texts.map((t) => hash(t));
    },
    async dispose() {},
  };
}

function createMockVectorIndex(): VectorIndex {
  const ids: string[] = [];

  return {
    async upsert(id: string) {
      if (!ids.includes(id)) ids.push(id);
    },
    async remove(id: string) {
      const i = ids.indexOf(id);
      if (i >= 0) ids.splice(i, 1);
    },
    async search(_vector: number[], k: number) {
      return ids.slice(0, k).map((id, index) => ({
        id,
        score: 1 - index / 100,
      }));
    },
  };
}

describe("Relai core claims", () => {
  let tmpDir: string | undefined;

  afterEach(() => {
    if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
    tmpDir = undefined;
  });

  function createRelai(): Relai {
    tmpDir = mkdtempSync(join(tmpdir(), "relai-test-"));
    return new Relai({
      dbPath: join(tmpDir, "relai.sqlite"),
      embedder: createMockEmbedder(),
      vectorIndex: createMockVectorIndex(),
      rerank: false,
      expand: false,
    });
  }

  test("stores literal properties and reference relationships as claims", async () => {
    const relai = createRelai();
    try {
      await relai.put("hubspot:ticket:1", "Support ticket for Acme");
      await relai.claim("hubspot:ticket:1", "priority", "high");
      await relai.claim("hubspot:ticket:1", "customer", ref("customer:acme"));

      const claims = await relai.match({ subject: "hubspot:ticket:1" });

      expect(claims).toHaveLength(2);
      expect(claims).toContainEqual({
        subject: "hubspot:ticket:1",
        predicate: "priority",
        object: "high",
      });
      expect(claims).toContainEqual({
        subject: "hubspot:ticket:1",
        predicate: "customer",
        object: ref("customer:acme"),
      });
    } finally {
      await relai.dispose();
    }
  });

  test("cross-system customer anchor connects HubSpot, Jira, and ClickUp", async () => {
    const relai = createRelai();
    try {
      await relai.put("customer:acme", "Customer Acme");
      await relai.put("hubspot:ticket:1", "HubSpot ticket for Acme");
      await relai.put("jira:issue:APP-1", "Jira issue for Acme");
      await relai.put("clickup:task:abc", "ClickUp task for Acme");

      await relai.claim("hubspot:ticket:1", "customer", ref("customer:acme"));
      await relai.claim("jira:issue:APP-1", "customer", ref("customer:acme"));
      await relai.claim("clickup:task:abc", "customer", ref("customer:acme"));

      const claims = await relai.match({
        predicate: "customer",
        object: ref("customer:acme"),
      });

      expect(claims.map((claim) => claim.subject).sort()).toEqual([
        "clickup:task:abc",
        "hubspot:ticket:1",
        "jira:issue:APP-1",
      ]);
    } finally {
      await relai.dispose();
    }
  });

  test("describe returns outgoing and incoming claims", async () => {
    const relai = createRelai();
    try {
      await relai.put("customer:acme", "Customer Acme");
      await relai.put("jira:issue:APP-1", "Jira issue for Acme");
      await relai.claim("customer:acme", "name", "Acme");
      await relai.claim("jira:issue:APP-1", "customer", ref("customer:acme"));

      const description = await relai.describe("customer:acme");

      expect(description.thing?.id).toBe("customer:acme");
      expect(description.claims).toEqual([
        { subject: "customer:acme", predicate: "name", object: "Acme" },
      ]);
      expect(description.incoming).toEqual([
        {
          subject: "jira:issue:APP-1",
          predicate: "customer",
          object: ref("customer:acme"),
        },
      ]);
    } finally {
      await relai.dispose();
    }
  });

  test("put stores searchable text", async () => {
    const relai = createRelai();
    try {
      await relai.put("customer:acme", "Customer Acme export issue");

      const results = await relai.search("Acme", 1);

      expect(results[0]?.id).toBe("customer:acme");
    } finally {
      await relai.dispose();
    }
  });
});

describe("Relai hybrid search", () => {
  let relai: Relai;

  beforeEach(() => {
    relai = new Relai({
      dbPath: ":memory:",
      embedder: createMockEmbedder(),
      rerank: false,
      expand: false,
    });
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

  test("a long view is chunked and a query matching only a late section still retrieves it", async () => {
    const longText =
      "Intro about onboarding.\n\n".repeat(20) +
      "The secret passphrase is XYZZY-PLUGH.\n\n" +
      "Closing remarks about offboarding.\n\n".repeat(20);
    await relai.index({ source: "docs", remoteId: "long", text: longText });
    const results = await relai.search("XYZZY-PLUGH passphrase", 5, { rerank: false });
    expect(results[0]?.id).toBe("view:docs:long");
  });
});

describe("claim store migration", () => {
  let db: Database;

  afterEach(() => {
    db.close();
  });

  test("migrates legacy from/type/to claims into reference claims", async () => {
    db = openDatabase(":memory:");
    db.exec(`
      CREATE TABLE claims (
        id TEXT PRIMARY KEY,
        "from" TEXT NOT NULL,
        type TEXT NOT NULL,
        "to" TEXT NOT NULL,
        created_by TEXT
      )
    `);
    db.prepare(
      `INSERT INTO claims (id, "from", type, "to") VALUES (?, ?, ?, ?)`
    ).run("c1", "hubspot:ticket:1", "customer", "customer:acme");

    const store = createClaimStore(db);
    const claims = await store.match({
      predicate: "customer",
      object: ref("customer:acme"),
    });

    expect(claims).toEqual([
      {
        subject: "hubspot:ticket:1",
        predicate: "customer",
        object: ref("customer:acme"),
      },
    ]);
  });
});
