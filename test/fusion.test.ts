// test/fusion.test.ts
import { describe, test, expect } from "bun:test";
import { reciprocalRankFusion } from "../src/fusion.js";

describe("reciprocalRankFusion", () => {
  test("doc ranked high in both lists wins", () => {
    const bm25 = [{ id: "a", score: 1 }, { id: "b", score: 0.5 }];
    const vec = [{ id: "a", score: 1 }, { id: "c", score: 0.5 }];
    const fused = reciprocalRankFusion([bm25, vec]);
    expect(fused[0]?.id).toBe("a");
    expect(fused[0]?.rank).toBe(1);
  });

  test("weights bias toward a list", () => {
    const original = [{ id: "x", score: 1 }];
    const expansion = [{ id: "y", score: 1 }];
    const fused = reciprocalRankFusion([original, expansion], [2.0, 1.0]);
    expect(fused[0]?.id).toBe("x");
  });

  test("missing/empty lists are tolerated", () => {
    const fused = reciprocalRankFusion([[], [{ id: "a", score: 1 }]]);
    expect(fused[0]?.id).toBe("a");
  });

  test("top-rank bonus boosts a rank-0 contributor", () => {
    // 'a' is rank 0 in list1 (gets +0.05); 'b' is rank 1 in both (no bonus).
    const l1 = [{ id: "a", score: 1 }, { id: "b", score: 0.9 }];
    const l2 = [{ id: "z", score: 1 }, { id: "b", score: 0.9 }];
    const fused = reciprocalRankFusion([l1, l2]);
    expect(fused[0]?.id).toBe("a");
  });
});
