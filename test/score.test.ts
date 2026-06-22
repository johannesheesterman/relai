import { describe, test, expect } from "bun:test";
import { precisionAtK, recallAtK, f1AtK } from "../bench/score.js";

describe("scoring", () => {
  test("precision@k counts relevant in top-k", () => {
    expect(precisionAtK(["a", "b", "c"], ["a", "c"], 2)).toBeCloseTo(0.5); // a relevant, b not → 1/2
  });
  test("recall@k counts found relevant over total relevant", () => {
    expect(recallAtK(["a", "b", "c"], ["a", "c", "z"], 3)).toBeCloseTo(2 / 3);
  });
  test("f1 combines precision and recall", () => {
    expect(f1AtK(["a", "b"], ["a"], 2)).toBeCloseTo(2 * (0.5 * 1) / (0.5 + 1));
  });
  test("empty relevant yields 0", () => {
    expect(precisionAtK(["a"], [], 1)).toBe(0);
  });
});
