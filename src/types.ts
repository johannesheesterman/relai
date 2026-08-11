export type ThingId = string;

export type Ref = { ref: ThingId };

export type ClaimObject = Ref | string | number | boolean | null;

export type Thing = {
  id: ThingId;
  text: string;
  type?: string;
  source?: string;
  remoteId?: string;
  links?: { type: string; to: string }[];
};

export type View = Thing;

export type ViewInput = {
  source: string;
  remoteId: string;
  type?: string;
  text: string;
  links?: { type: string; to: string }[];
};

export type Claim = {
  subject: ThingId;
  predicate: string;
  object: ClaimObject;
  createdBy?: string;
};

export type ClaimPattern = {
  subject?: ThingId;
  predicate?: string;
  object?: ClaimObject;
};

export type Description = {
  thing?: Thing;
  claims: Claim[];
  incoming: Claim[];
};

export function ref(id: ThingId): Ref {
  return { ref: id };
}

export function isRef(value: ClaimObject): value is Ref {
  return (
    typeof value === "object" &&
    value !== null &&
    "ref" in value &&
    typeof value.ref === "string"
  );
}

export interface Embedder {
  embed(text: string): Promise<number[]>;
  embedQuery(text: string): Promise<number[]>;
  embedMany(texts: string[]): Promise<number[][]>;
  dispose(): Promise<void>;
}

export interface VectorIndex {
  upsert(id: string, vector: number[]): Promise<void>;
  remove(id: string): Promise<void>;
  search(vector: number[], k: number): Promise<RankedItem[]>;
}

export interface ViewStore {
  put(view: Thing): Promise<void>;
  getMany(ids: string[]): Promise<Thing[]>;
}

export interface ClaimStore {
  put(claim: Claim): Promise<void>;
  delete(pattern: ClaimPattern): Promise<void>;
  match(pattern: ClaimPattern): Promise<Claim[]>;
}

export type RankedItem = { id: string; score: number };

export interface FtsIndex {
  upsert(id: string, text: string): void;
  remove(id: string): void;
  search(query: string, k: number): RankedItem[];
}

export interface Reranker {
  rank(query: string, documents: string[]): Promise<number[]>;
  dispose(): Promise<void>;
}

export type SearchOptions = {
  k?: number;
  candidateLimit?: number;
  rerank?: boolean;
  expand?: boolean;
  minScore?: number;
};

export type Chunk = { text: string; pos: number };

export interface ChunkStore {
  putChunks(viewId: string, chunks: Chunk[]): void;
  removeByView(viewId: string): void;
  getText(chunkId: string): string | undefined;
  allChunkTexts(): { id: string; viewId: string; text: string }[];
}

export type ExpandedQuery = { type: "lex" | "vec" | "hyde"; query: string };

export interface QueryExpander {
  expand(query: string): Promise<ExpandedQuery[]>;
  dispose(): Promise<void>;
}
