import type { Database } from "./db.js";
import type { Claim, ClaimStore } from "./types.js";

export function createClaimStore(db: Database): ClaimStore {
  db.exec(`
    CREATE TABLE IF NOT EXISTS claims (
      id TEXT PRIMARY KEY,
      "from" TEXT NOT NULL,
      type TEXT NOT NULL,
      "to" TEXT NOT NULL,
      created_by TEXT
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_claims_from ON claims("from")`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_claims_to ON claims("to")`);

  const putStmt = db.prepare(`
    INSERT OR REPLACE INTO claims (id, "from", type, "to", created_by)
    VALUES (?, ?, ?, ?, ?)
  `);

  const fromStmt = db.prepare(`SELECT * FROM claims WHERE "from" = ?`);
  const toStmt = db.prepare(`SELECT * FROM claims WHERE "to" = ?`);

  function rowToClaim(row: any): Claim {
    return {
      id: row.id,
      from: row.from,
      type: row.type,
      to: row.to,
      createdBy: row.created_by ?? undefined,
    };
  }

  return {
    async put(claim: Claim) {
      putStmt.run(
        claim.id,
        claim.from,
        claim.type,
        claim.to,
        claim.createdBy ?? null
      );
    },

    async from(viewId: string): Promise<Claim[]> {
      return fromStmt.all(viewId).map(rowToClaim);
    },

    async to(viewId: string): Promise<Claim[]> {
      return toStmt.all(viewId).map(rowToClaim);
    },
  };
}
