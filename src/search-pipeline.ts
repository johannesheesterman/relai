// src/search-pipeline.ts
import type { RankedItem, SearchOptions, Reranker } from "./types.js";
import { reciprocalRankFusion, positionAwareBlend } from "./fusion.js";

export type SearchDeps = {
  embedQuery: (q: string) => Promise<number[]>;
  vectorSearch: (vec: number[], k: number) => Promise<RankedItem[]>;
  ftsSearch: (q: string, k: number) => RankedItem[];
  getText: (id: string) => string | undefined;
  reranker?: Reranker;
};

export async function hybridSearch(
  deps: SearchDeps,
  query: string,
  options: SearchOptions = {}
): Promise<RankedItem[]> {
  const k = options.k ?? 5;
  const candidateLimit = options.candidateLimit ?? 40;
  const doRerank = (options.rerank ?? true) && !!deps.reranker;
  const minScore = options.minScore ?? 0;

  const width = Math.max(candidateLimit, k * 4);
  const vector = await deps.embedQuery(query);
  const [vecHits, ftsHits] = [
    await deps.vectorSearch(vector, width),
    deps.ftsSearch(query, width),
  ];

  const fused = reciprocalRankFusion([vecHits, ftsHits]);
  const candidates = fused.slice(0, candidateLimit);

  if (!doRerank) {
    return candidates
      .slice(0, k)
      .map((c) => ({ id: c.id, score: c.rrfScore }))
      .filter((r) => r.score >= minScore);
  }

  const texts = candidates.map((c) => deps.getText(c.id) ?? "");
  const rerankScores = await deps.reranker!.rank(query, texts);

  const blended = positionAwareBlend(
    candidates.map((c, i) => ({
      id: c.id,
      rrfRank: c.rank,
      rerankScore: rerankScores[i] ?? 0,
    }))
  );

  return blended.filter((r) => r.score >= minScore).slice(0, k);
}
