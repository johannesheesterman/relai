import type { Database } from "./db.js";
import type { FtsIndex, RankedItem } from "./types.js";

// FTS5 query syntax is strict; sanitize user input into a safe OR-of-terms query.
function toMatchQuery(query: string): string {
  const terms = query
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0)
    .map((t) => `"${t}"`);
  return terms.join(" OR ");
}

export function createFtsIndex(db: Database): FtsIndex {
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS views_fts USING fts5(
      id UNINDEXED,
      text
    )
  `);

  const removeStmt = db.prepare(`DELETE FROM views_fts WHERE id = ?`);
  const insertStmt = db.prepare(`INSERT INTO views_fts (id, text) VALUES (?, ?)`);

  return {
    upsert(id: string, text: string) {
      removeStmt.run(id);
      insertStmt.run(id, text);
    },

    remove(id: string) {
      removeStmt.run(id);
    },

    search(query: string, k: number): RankedItem[] {
      const match = toMatchQuery(query);
      if (match.length === 0) return [];
      // bm25() is negative; lower (more negative) = better. Order ascending.
      const rows = db
        .prepare(
          `SELECT id, bm25(views_fts) AS score
           FROM views_fts
           WHERE views_fts MATCH ?
           ORDER BY score ASC
           LIMIT ?`
        )
        .all(match, k) as { id: string; score: number }[];
      return rows.map((row) => {
        const mag = Math.abs(row.score);
        // Map negative bm25 → stable (0,1), higher = better.
        return { id: row.id, score: mag / (1 + mag) };
      });
    },
  };
}
