import { openDatabase, type Database } from "./db.js";
import { createViewStore } from "./view-store.js";
import { createClaimStore } from "./claim-store.js";
import { createVectorIndex } from "./vector-index.js";
import { createEmbedder } from "./embedder.js";
import { createFtsIndex } from "./fts-index.js";
import { createReranker } from "./reranker.js";
import { createChunkStore } from "./chunk-store.js";
import { chunkText } from "./chunker.js";
import { viewIdOf } from "./ids.js";
import { hybridSearch, type SearchDeps } from "./search-pipeline.js";
import type {
  Claim,
  ClaimStore,
  ChunkStore,
  Embedder,
  FtsIndex,
  Reranker,
  SearchOptions,
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
  rerankModel?: string;
  rerank?: boolean;
};

const DEFAULT_DB_DIR = join(homedir(), ".config", "relai");
const DEFAULT_DB_PATH = join(DEFAULT_DB_DIR, "relai.sqlite");

export class Relai {
  private db: Database;
  private viewStore: ViewStore;
  private claimStore: ClaimStore;
  private vectorIndex: VectorIndex;
  private ftsIndex: FtsIndex;
  private chunkStore: ChunkStore;
  private embedder: Embedder;
  private reranker: Reranker | null = null;
  private rerankEnabled: boolean;
  private rerankModel?: string;
  private textCache = new Map<string, string>();
  private textCacheWarmed = false;

  constructor(config?: RelaiConfig) {
    const dbPath = config?.dbPath ?? DEFAULT_DB_PATH;
    const dbDir = join(dbPath, "..");
    if (!existsSync(dbDir)) mkdirSync(dbDir, { recursive: true });

    this.db = openDatabase(dbPath);
    this.viewStore = createViewStore(this.db);
    this.claimStore = createClaimStore(this.db);
    this.vectorIndex = createVectorIndex(this.db);
    this.ftsIndex = createFtsIndex(this.db);
    this.chunkStore = createChunkStore(this.db);
    this.embedder = config?.embedder ?? createEmbedder({ model: config?.embedModel });
    this.rerankEnabled = config?.rerank ?? true;
    this.rerankModel = config?.rerankModel;
  }

  private getReranker(): Reranker | undefined {
    if (!this.rerankEnabled) return undefined;
    if (!this.reranker) this.reranker = createReranker({ model: this.rerankModel });
    return this.reranker;
  }

  private warmTextCache() {
    if (this.textCacheWarmed) return;
    const rows = this.db.prepare(`SELECT id, text FROM views`).all() as {
      id: string;
      text: string;
    }[];
    for (const r of rows) this.textCache.set(r.id, r.text);
    this.textCacheWarmed = true;
  }

  async index(input: ViewInput): Promise<View> {
    const view: View = {
      ...input,
      id: `view:${input.source}:${input.remoteId}`,
    };
    await this.viewStore.put(view);
    this.ftsIndex.upsert(view.id, view.text);
    await this.indexChunks(view.id, view.text);
    return view;
  }

  private async indexChunks(viewId: string, text: string) {
    const chunks = chunkText(text);
    this.chunkStore.putChunks(viewId, chunks);
    // Remove any stale chunk vectors for this view, then add fresh ones.
    // (Vectors keyed by chunk id; a re-index with fewer chunks must not leave orphans.)
    await this.vectorIndex.remove(viewId); // legacy single-vector id, if present
    const vectors = await this.embedder.embedMany(chunks.map((c) => c.text));
    for (let i = 0; i < chunks.length; i++) {
      await this.vectorIndex.upsert(`${viewId}#${i}`, vectors[i]!);
    }
    this.textCache.set(viewId, text);
  }

  async search(
    query: string,
    k: number = 5,
    options: SearchOptions = {}
  ): Promise<View[]> {
    this.warmTextCache();
    const rerank = options.rerank ?? this.rerankEnabled;
    const deps: SearchDeps = {
      embedQuery: (q) => this.embedder.embedQuery(q),
      vectorSearch: (vec, n) => this.vectorIndex.search(vec, n),
      ftsSearch: (q, n) => this.ftsIndex.search(q, n),
      getText: (id) => this.chunkStore.getText(id) ?? this.textCache.get(viewIdOf(id)),
      idToGroup: viewIdOf,
      reranker: rerank ? this.getReranker() : undefined,
    };
    const ranked = await hybridSearch(deps, query, { ...options, k, rerank });

    // Collapse any chunk ids to view ids, keeping best rank, deduped.
    const seen = new Set<string>();
    const viewOrder: string[] = [];
    for (const r of ranked) {
      const vid = viewIdOf(r.id);
      if (seen.has(vid)) continue;
      seen.add(vid);
      viewOrder.push(vid);
    }
    const views = await this.viewStore.getMany(viewOrder);
    const order = new Map(viewOrder.map((id, i) => [id, i]));
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
      await this.indexChunks(claimView.id, text);
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
    if (this.reranker) await this.reranker.dispose();
    await this.embedder.dispose();
    this.db.close();
  }
}
