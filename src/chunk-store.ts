// src/chunk-store.ts
import type { Database } from "./db.js";
import type { Chunk, ChunkStore } from "./types.js";
import { chunkId } from "./ids.js";

export function createChunkStore(db: Database): ChunkStore {
  db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      view_id TEXT NOT NULL,
      pos INTEGER NOT NULL,
      text TEXT NOT NULL
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_view ON chunks(view_id)`);

  const removeStmt = db.prepare(`DELETE FROM chunks WHERE view_id = ?`);
  const insertStmt = db.prepare(
    `INSERT INTO chunks (id, view_id, pos, text) VALUES (?, ?, ?, ?)`
  );
  const getStmt = db.prepare(`SELECT text FROM chunks WHERE id = ?`);

  return {
    putChunks(viewId: string, chunks: Chunk[]) {
      removeStmt.run(viewId);
      chunks.forEach((c, i) =>
        insertStmt.run(chunkId(viewId, i), viewId, c.pos, c.text)
      );
    },
    removeByView(viewId: string) {
      removeStmt.run(viewId);
    },
    getText(id: string) {
      const row = getStmt.get(id) as { text: string } | undefined;
      return row?.text;
    },
    allChunkTexts() {
      const rows = db.prepare(`SELECT id, view_id, text FROM chunks`).all() as {
        id: string;
        view_id: string;
        text: string;
      }[];
      return rows.map((r) => ({ id: r.id, viewId: r.view_id, text: r.text }));
    },
  };
}
