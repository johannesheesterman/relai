// src/search-pipeline.ts
import type {
  RankedItem,
  SearchOptions,
  Reranker,
  QueryExpander,
  ExpandedQuery,
} from "./types.js";
import { reciprocalRankFusion, positionAwareBlend } from "./fusion.js";

export type SearchDeps = {
  embedQuery: (q: string) => Promise<number[]>;
  vectorSearch: (vec: number[], k: number) => Promise<RankedItem[]>;
  ftsSearch: (q: string, k: number) => RankedItem[];
  getText: (id: string) => string | undefined;
  idToGroup?: (id: string) => string;
  reranker?: Reranker;
  expander?: QueryExpander;
};

const STRONG_MIN = 0.85;
const STRONG_GAP = 0.15;

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
  const baseVec = await deps.vectorSearch(vector, width);
  const baseFts = deps.ftsSearch(query, width);

  // Strong-signal short-circuit: when the top FTS hit dominates, the keyword
  // match is unambiguous and expansion only adds noise — skip it.
  const top = baseFts[0]?.score ?? 0;
  const second = baseFts[1]?.score ?? 0;
  const strongSignal =
    baseFts.length > 0 && top >= STRONG_MIN && top - second >= STRONG_GAP;

  const doExpand =
    (options.expand ?? true) && !!deps.expander && !strongSignal;
  // Graceful degradation: a missing/failed expansion model must never break
  // search — fall back to no expansions (vector + BM25 + RRF still run).
  let expansions: ExpandedQuery[] = [];
  if (doExpand) {
    try {
      expansions = await deps.expander!.expand(query);
    } catch {
      expansions = [];
    }
  }

  // Original lists carry weight 2.0; routed expansion lists carry 1.0.
  const lists: RankedItem[][] = [baseVec, baseFts];
  const weights: number[] = [2.0, 2.0];
  for (const e of expansions) {
    if (e.type === "lex") {
      lists.push(deps.ftsSearch(e.query, width));
    } else {
      const ev = await deps.embedQuery(e.query);
      lists.push(await deps.vectorSearch(ev, width));
    }
    weights.push(1.0);
  }

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
  const collapsedLists = lists.map(collapse);

  const repChunk = new Map<string, string>(); // groupId -> best chunk id (for rerank text)
  // Apply later lists first so the original vector list (index 0) wins ties and
  // its real chunk ids (carrying `#n`) become the representatives for rerank text.
  for (let i = collapsedLists.length - 1; i >= 0; i--) {
    for (const it of collapsedLists[i]!) repChunk.set(group(it.id), it.id);
  }

  const toGrouped = (l: RankedItem[]) =>
    l.map((it) => ({ id: group(it.id), score: it.score }));
  const fused = reciprocalRankFusion(
    collapsedLists.map(toGrouped),
    weights
  );
  const candidates = fused.slice(0, candidateLimit);

  // Shared RRF-only fallback: returned when rerank is disabled OR when the
  // reranker is unavailable/throws, so search always degrades gracefully to
  // vector + BM25 + RRF instead of failing.
  const rrfFallback = () =>
    candidates
      .map((c) => ({ id: c.id, score: c.rrfScore }))
      .filter((r) => r.score >= minScore)
      .slice(0, k);

  if (!doRerank) return rrfFallback();

  let rerankScores: number[];
  try {
    const texts = candidates.map(
      (c) => deps.getText(repChunk.get(c.id) ?? c.id) ?? ""
    );
    rerankScores = await deps.reranker!.rank(query, texts);
  } catch {
    // Graceful degradation: a missing/failed rerank model must never throw out
    // of search — fall back to the RRF-fused ranking.
    return rrfFallback();
  }

  const blended = positionAwareBlend(
    candidates.map((c, i) => ({
      id: c.id,
      rrfRank: c.rank,
      rerankScore: rerankScores[i] ?? 0,
    }))
  );

  return blended.filter((r) => r.score >= minScore).slice(0, k);
}
