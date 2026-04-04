import type { Database } from "./db.js";
import type { View, ViewStore } from "./types.js";

export function createViewStore(db: Database): ViewStore {
  db.exec(`
    CREATE TABLE IF NOT EXISTS views (
      id TEXT PRIMARY KEY,
      source TEXT NOT NULL,
      remote_id TEXT NOT NULL,
      type TEXT,
      text TEXT NOT NULL,
      links TEXT
    )
  `);
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_views_source_remote ON views(source, remote_id)`
  );

  const putStmt = db.prepare(`
    INSERT OR REPLACE INTO views (id, source, remote_id, type, text, links)
    VALUES (?, ?, ?, ?, ?, ?)
  `);

  return {
    async put(view: View) {
      putStmt.run(
        view.id,
        view.source,
        view.remoteId,
        view.type ?? null,
        view.text,
        view.links ? JSON.stringify(view.links) : null
      );
    },

    async getMany(ids: string[]): Promise<View[]> {
      if (ids.length === 0) return [];
      const placeholders = ids.map(() => "?").join(",");
      const rows = db
        .prepare(`SELECT * FROM views WHERE id IN (${placeholders})`)
        .all(...ids);
      return rows.map((row: any) => ({
        id: row.id,
        source: row.source,
        remoteId: row.remote_id,
        type: row.type ?? undefined,
        text: row.text,
        links: row.links ? JSON.parse(row.links) : undefined,
      }));
    },
  };
}
