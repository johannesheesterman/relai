import { openDatabase, type Database } from "./db.js";
import { createViewStore } from "./view-store.js";
import { createClaimStore } from "./claim-store.js";
import { createVectorIndex } from "./vector-index.js";
import { createEmbedder } from "./embedder.js";
import { createFtsIndex } from "./fts-index.js";
import { reciprocalRankFusion } from "./fusion.js";
import type {
  Claim,
  ClaimStore,
  Embedder,
  FtsIndex,
  View,
  ViewInput,
  ViewStore,
  VectorIndex,
} from "./types.js";
import { homedir } from "os";
import { join } from "path";
import { mkdirSync, existsSync } from "fs";

export type RelaiConfig = {
  dbPath?: string;
  embedModel?: string;
  embedder?: Embedder;
};

const DEFAULT_DB_DIR = join(homedir(), ".config", "relai");
const DEFAULT_DB_PATH = join(DEFAULT_DB_DIR, "relai.sqlite");

export class Relai {
  private db: Database;
  private viewStore: ViewStore;
  private claimStore: ClaimStore;
  private vectorIndex: VectorIndex;
  private ftsIndex: FtsIndex;
  private embedder: Embedder;

  constructor(config?: RelaiConfig) {
    const dbPath = config?.dbPath ?? DEFAULT_DB_PATH;
    const dbDir = join(dbPath, "..");
    if (!existsSync(dbDir)) mkdirSync(dbDir, { recursive: true });

    this.db = openDatabase(dbPath);
    this.viewStore = createViewStore(this.db);
    this.claimStore = createClaimStore(this.db);
    this.vectorIndex = createVectorIndex(this.db);
    this.ftsIndex = createFtsIndex(this.db);
    this.embedder = config?.embedder ?? createEmbedder({ model: config?.embedModel });
  }

  async index(input: ViewInput): Promise<View> {
    const view: View = {
      ...input,
      id: `view:${input.source}:${input.remoteId}`,
    };
    await this.viewStore.put(view);
    this.ftsIndex.upsert(view.id, view.text);
    const vector = await this.embedder.embed(view.text);
    await this.vectorIndex.upsert(view.id, vector);
    return view;
  }

  async search(query: string, k: number = 5): Promise<View[]> {
    const vector = await this.embedder.embedQuery(query);
    const [vecHits, ftsHits] = [
      await this.vectorIndex.search(vector, Math.max(k * 4, 20)),
      this.ftsIndex.search(query, Math.max(k * 4, 20)),
    ];
    const fused = reciprocalRankFusion([vecHits, ftsHits]);
    const topIds = fused.slice(0, k).map((f) => f.id);
    const views = await this.viewStore.getMany(topIds);
    const order = new Map(topIds.map((id, i) => [id, i]));
    return views.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
  }

  async claim(
    from: string,
    type: string,
    to: string,
    createdBy?: string
  ): Promise<Claim> {
    const claim: Claim = {
      id: crypto.randomUUID(),
      from,
      type,
      to,
      createdBy,
    };
    await this.claimStore.put(claim);

    const [fromViews, toViews] = await Promise.all([
      this.viewStore.getMany([from]),
      this.viewStore.getMany([to]),
    ]);
    const fromView = fromViews[0];
    const toView = toViews[0];

    if (fromView && toView) {
      const text = `${fromView.text} ${type} ${toView.text}`;
      const claimView: View = {
        id: `view:claim:${claim.id}`,
        source: "claim",
        remoteId: claim.id,
        type: "claim",
        text,
        links: [
          { type: "from", to: from },
          { type: "to", to: to },
        ],
      };
      await this.viewStore.put(claimView);
      this.ftsIndex.upsert(claimView.id, text);
      const vector = await this.embedder.embed(text);
      await this.vectorIndex.upsert(claimView.id, vector);
    }

    return claim;
  }

  async related(viewId: string): Promise<Claim[]> {
    const [fromClaims, toClaims] = await Promise.all([
      this.claimStore.from(viewId),
      this.claimStore.to(viewId),
    ]);
    return [...fromClaims, ...toClaims];
  }

  async dispose(): Promise<void> {
    await this.embedder.dispose();
    this.db.close();
  }
}
