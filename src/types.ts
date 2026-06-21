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
  dispose(): Promise<void>;
}

export interface VectorIndex {
  upsert(viewId: string, vector: number[]): Promise<void>;
  search(
    vector: number[],
    k: number
  ): Promise<{ viewId: string; score: number }[]>;
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
