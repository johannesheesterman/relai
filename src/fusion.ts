// src/fusion.ts
import type { RankedItem } from "./types.js";

export type FusedItem = { id: string; rrfScore: number; rank: number };

export function reciprocalRankFusion(
  lists: RankedItem[][],
  weights: number[] = [],
  k: number = 60
): FusedItem[] {
  const scores = new Map<string, { rrfScore: number; topRank: number }>();

  for (let listIdx = 0; listIdx < lists.length; listIdx++) {
    const list = lists[listIdx];
    if (!list) continue;
    const weight = weights[listIdx] ?? 1.0;
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      if (!item) continue;
      const contribution = weight / (k + rank + 1);
      const existing = scores.get(item.id);
      if (existing) {
        existing.rrfScore += contribution;
        existing.topRank = Math.min(existing.topRank, rank);
      } else {
        scores.set(item.id, { rrfScore: contribution, topRank: rank });
      }
    }
  }

  for (const entry of scores.values()) {
    if (entry.topRank === 0) entry.rrfScore += 0.05;
    else if (entry.topRank <= 2) entry.rrfScore += 0.02;
  }

  return Array.from(scores.entries())
    .map(([id, e]) => ({ id, rrfScore: e.rrfScore }))
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map((e, i) => ({ ...e, rank: i + 1 }));
}

export function positionAwareBlend(
  items: { id: string; rrfRank: number; rerankScore: number }[]
): RankedItem[] {
  return items
    .map((it) => {
      let w: number;
      if (it.rrfRank <= 3) w = 0.75;
      else if (it.rrfRank <= 10) w = 0.6;
      else w = 0.4;
      const rrfScore = 1 / it.rrfRank;
      const score = w * rrfScore + (1 - w) * it.rerankScore;
      return { id: it.id, score };
    })
    .sort((a, b) => b.score - a.score);
}
