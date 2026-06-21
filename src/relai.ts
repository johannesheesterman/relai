import { openDatabase, type Database } from "./db.js";
import { createViewStore } from "./view-store.js";
import { createClaimStore } from "./claim-store.js";
import { createVectorIndex } from "./vector-index.js";
import { createEmbedder } from "./embedder.js";
import type {
  Claim,
  ClaimObject,
  ClaimPattern,
  ClaimStore,
  Description,
  Embedder,
  Thing,
  ThingId,
  ViewInput,
  ViewStore,
  VectorIndex,
} from "./types.js";
import { ref } from "./types.js";
import { homedir } from "os";
import { join } from "path";
import { mkdirSync, existsSync } from "fs";

export type RelaiConfig = {
  dbPath?: string;
  embedModel?: string;
  embedder?: Embedder;
  vectorIndex?: VectorIndex;
};

const DEFAULT_DB_DIR = join(homedir(), ".config", "relai");
const DEFAULT_DB_PATH = join(DEFAULT_DB_DIR, "relai.sqlite");

export class Relai {
  private db: Database;
  private viewStore: ViewStore;
  private claimStore: ClaimStore;
  private vectorIndex: VectorIndex;
  private embedder: Embedder;

  constructor(config?: RelaiConfig) {
    const dbPath = config?.dbPath ?? DEFAULT_DB_PATH;
    const dbDir = join(dbPath, "..");
    if (!existsSync(dbDir)) mkdirSync(dbDir, { recursive: true });

    this.db = openDatabase(dbPath);
    this.viewStore = createViewStore(this.db);
    this.claimStore = createClaimStore(this.db);
    this.vectorIndex = config?.vectorIndex ?? createVectorIndex(this.db);
    this.embedder = config?.embedder ?? createEmbedder({ model: config?.embedModel });
  }

  async put(
    idOrThing: ThingId | Thing,
    text?: string
  ): Promise<Thing> {
    const thing: Thing =
      typeof idOrThing === "string"
        ? { id: idOrThing, text: text ?? "" }
        : idOrThing;

    await this.viewStore.put(thing);
    const vector = await this.embedder.embed(thing.text);
    await this.vectorIndex.upsert(thing.id, vector);
    return thing;
  }

  async index(input: ViewInput): Promise<Thing> {
    return this.put({
      id: `view:${input.source}:${input.remoteId}`,
      source: input.source,
      remoteId: input.remoteId,
      type: input.type,
      text: input.text,
      links: input.links,
    });
  }

  async search(query: string, k: number = 5): Promise<Thing[]> {
    const vector = await this.embedder.embedQuery(query);
    const matches = await this.vectorIndex.search(vector, k);
    const views = await this.viewStore.getMany(matches.map((m) => m.viewId));

    const scoreMap = new Map(matches.map((m) => [m.viewId, m.score]));
    return views.sort(
      (a, b) => (scoreMap.get(b.id) ?? 0) - (scoreMap.get(a.id) ?? 0)
    );
  }

  async claim(
    subject: ThingId,
    predicate: string,
    object: ClaimObject,
    createdBy?: string
  ): Promise<Claim> {
    const claim: Claim = {
      subject,
      predicate,
      object,
      createdBy,
    };
    await this.claimStore.put(claim);
    return claim;
  }

  async unclaim(
    subject: ThingId,
    predicate?: string,
    object?: ClaimObject
  ): Promise<void> {
    await this.claimStore.delete({ subject, predicate, object });
  }

  async match(pattern: ClaimPattern): Promise<Claim[]> {
    return this.claimStore.match(pattern);
  }

  async describe(id: ThingId): Promise<Description> {
    const [thing] = await this.viewStore.getMany([id]);
    const [claims, incoming] = await Promise.all([
      this.claimStore.match({ subject: id }),
      this.claimStore.match({ object: ref(id) }),
    ]);

    return { thing, claims, incoming };
  }

  async related(viewId: string): Promise<Claim[]> {
    const description = await this.describe(viewId);
    return [...description.claims, ...description.incoming];
  }

  async dispose(): Promise<void> {
    await this.embedder.dispose();
    this.db.close();
  }
}
