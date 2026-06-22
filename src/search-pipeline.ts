// src/search-pipeline.ts
import type { RankedItem, SearchOptions, Reranker } from "./types.js";
import { reciprocalRankFusion, positionAwareBlend } from "./fusion.js";

export type SearchDeps = {
  embedQuery: (q: string) => Promise<number[]>;
  vectorSearch: (vec: number[], k: number) => Promise<RankedItem[]>;
  ftsSearch: (q: string, k: number) => RankedItem[];
  getText: (id: string) => string | undefined;
  idToGroup?: (id: string) => string;
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

  // Collapse chunk ids to view (group) ids before fusion so vector hits (chunk
  // ids) and FTS hits (view ids) align on the same id space. Keep the best
  // chunk per group, and remember the representative chunk id for rerank text.
  const group = deps.idToGroup ?? ((id: string) => id);
  const collapse = (list: RankedItem[]) => {
    const best = new Map<string, RankedItem>();
    for (const it of list) {
      const g = group(it.id);
      const cur = best.get(g);
      if (!cur || it.score > cur.score) best.set(g, { id: it.id, score: it.score });
    }
    return Array.from(best.values()).sort((a, b) => b.score - a.score);
  };
  const vecCollapsed = collapse(vecHits);
  const ftsCollapsed = collapse(ftsHits);

  const repChunk = new Map<string, string>(); // groupId -> best chunk id (for rerank text)
  for (const it of ftsCollapsed) repChunk.set(group(it.id), it.id);
  for (const it of vecCollapsed) repChunk.set(group(it.id), it.id);

  const toGrouped = (l: RankedItem[]) =>
    l.map((it) => ({ id: group(it.id), score: it.score }));
  const fused = reciprocalRankFusion([
    toGrouped(vecCollapsed),
    toGrouped(ftsCollapsed),
  ]);
  const candidates = fused.slice(0, candidateLimit);

  if (!doRerank) {
    return candidates
      .slice(0, k)
      .map((c) => ({ id: c.id, score: c.rrfScore }))
      .filter((r) => r.score >= minScore);
  }

  const texts = candidates.map(
    (c) => deps.getText(repChunk.get(c.id) ?? c.id) ?? ""
  );
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
