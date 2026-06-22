import type { Database } from "./db.js";
import { loadSqliteVec } from "./db.js";
import type { VectorIndex } from "./types.js";

export function createVectorIndex(db: Database): VectorIndex {
  loadSqliteVec(db);

  let tableReady = !!db
    .prepare(
      `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'`
    )
    .get();

  function ensureTable(dims: number) {
    if (tableReady) return;
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
        view_id TEXT PRIMARY KEY,
        embedding float[${dims}] distance_metric=cosine
      )
    `);
    tableReady = true;
  }

  return {
    async upsert(id: string, vector: number[]) {
      ensureTable(vector.length);
      const embedding = new Float32Array(vector);
      db.prepare(`DELETE FROM vectors WHERE view_id = ?`).run(id);
      db.prepare(
        `INSERT INTO vectors (view_id, embedding) VALUES (?, ?)`
      ).run(id, embedding);
    },

    async remove(id: string) {
      if (!tableReady) return;
      db.prepare(`DELETE FROM vectors WHERE view_id = ?`).run(id);
    },

    async search(vector: number[], k: number) {
      if (!tableReady) return [];
      const embedding = new Float32Array(vector);
      const rows = db
        .prepare(
          `SELECT view_id, distance FROM vectors WHERE embedding MATCH ? AND k = ?`
        )
        .all(embedding, k);
      return rows.map((row: any) => ({
        id: row.view_id,
        score: 1 - row.distance,
      }));
    },
  };
}
