export type View = {
  id: string;
  source: string;
  remoteId: string;
  type?: string;
  text: string;
  links?: { type: string; to: string }[];
};

export type Claim = {
  id: string;
  from: string;
  type: string;
  to: string;
  createdBy?: string;
};

export type ViewInput = {
  source: string;
  remoteId: string;
  type?: string;
  text: string;
  links?: { type: string; to: string }[];
};

export interface Embedder {
  embed(text: string): Promise<number[]>;
  embedQuery(text: string): Promise<number[]>;
  dispose(): Promise<void>;
}

export interface VectorIndex {
  upsert(id: string, vector: number[]): Promise<void>;
  remove(id: string): Promise<void>;
  search(vector: number[], k: number): Promise<RankedItem[]>;
}

export interface ViewStore {
  put(view: View): Promise<void>;
  getMany(ids: string[]): Promise<View[]>;
}

export interface ClaimStore {
  put(claim: Claim): Promise<void>;
  from(viewId: string): Promise<Claim[]>;
  to(viewId: string): Promise<Claim[]>;
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
  minScore?: number;
};
