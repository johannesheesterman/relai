import type { Database } from "./db.js";
import type { Claim, ClaimObject, ClaimPattern, ClaimStore } from "./types.js";
import { isRef, ref } from "./types.js";

function encodeObject(object: ClaimObject): { value: string; isRef: number } {
  if (isRef(object)) return { value: object.ref, isRef: 1 };
  return { value: JSON.stringify(object), isRef: 0 };
}

function decodeObject(value: string, isRefValue: number): ClaimObject {
  if (isRefValue === 1) return ref(value);
  return JSON.parse(value) as ClaimObject;
}

function ensureClaimsTable(db: Database): void {
  const existing = db
    .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'claims'`)
    .get();

  if (existing) {
    const columns = db.prepare(`PRAGMA table_info(claims)`).all();
    const columnNames = new Set(columns.map((column: any) => column.name));

    if (columnNames.has("from") && columnNames.has("to")) {
      db.exec(`
        ALTER TABLE claims RENAME TO claims_legacy;

        CREATE TABLE claims (
          subject TEXT NOT NULL,
          predicate TEXT NOT NULL,
          object TEXT NOT NULL,
          object_is_ref INTEGER NOT NULL CHECK (object_is_ref IN (0, 1)),
          created_by TEXT,
          PRIMARY KEY (subject, predicate, object, object_is_ref)
        );

        INSERT OR IGNORE INTO claims (
          subject,
          predicate,
          object,
          object_is_ref,
          created_by
        )
        SELECT
          "from",
          type,
          "to",
          1,
          created_by
        FROM claims_legacy;

        DROP TABLE claims_legacy;
      `);
    }
  } else {
    db.exec(`
      CREATE TABLE claims (
        subject TEXT NOT NULL,
        predicate TEXT NOT NULL,
        object TEXT NOT NULL,
        object_is_ref INTEGER NOT NULL CHECK (object_is_ref IN (0, 1)),
        created_by TEXT,
        PRIMARY KEY (subject, predicate, object, object_is_ref)
      )
    `);
  }

  db.exec(`CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject)`);
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_claims_predicate_object ON claims(predicate, object, object_is_ref)`
  );
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_claims_object_ref ON claims(object) WHERE object_is_ref = 1`
  );
}

function buildWhere(pattern: ClaimPattern): {
  where: string;
  params: unknown[];
} {
  const clauses: string[] = [];
  const params: unknown[] = [];

  if (pattern.subject !== undefined) {
    clauses.push("subject = ?");
    params.push(pattern.subject);
  }

  if (pattern.predicate !== undefined) {
    clauses.push("predicate = ?");
    params.push(pattern.predicate);
  }

  if (pattern.object !== undefined) {
    const encoded = encodeObject(pattern.object);
    clauses.push("object = ?", "object_is_ref = ?");
    params.push(encoded.value, encoded.isRef);
  }

  return {
    where: clauses.length ? `WHERE ${clauses.join(" AND ")}` : "",
    params,
  };
}

function rowToClaim(row: any): Claim {
  return {
    subject: row.subject,
    predicate: row.predicate,
    object: decodeObject(row.object, row.object_is_ref),
    createdBy: row.created_by ?? undefined,
  };
}

export function createClaimStore(db: Database): ClaimStore {
  ensureClaimsTable(db);

  const putStmt = db.prepare(`
    INSERT OR REPLACE INTO claims (
      subject,
      predicate,
      object,
      object_is_ref,
      created_by
    )
    VALUES (?, ?, ?, ?, ?)
  `);

  return {
    async put(claim: Claim) {
      const encoded = encodeObject(claim.object);
      putStmt.run(
        claim.subject,
        claim.predicate,
        encoded.value,
        encoded.isRef,
        claim.createdBy ?? null
      );
    },

    async delete(pattern: ClaimPattern): Promise<void> {
      const { where, params } = buildWhere(pattern);
      if (!where) throw new Error("Refusing to delete all claims without a pattern.");
      db.prepare(`DELETE FROM claims ${where}`).run(...params);
    },

    async match(pattern: ClaimPattern): Promise<Claim[]> {
      const { where, params } = buildWhere(pattern);
      const rows = db
        .prepare(
          `SELECT subject, predicate, object, object_is_ref, created_by FROM claims ${where}`
        )
        .all(...params);
      return rows.map(rowToClaim);
    },
  };
}
