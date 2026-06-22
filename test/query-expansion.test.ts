import { describe, test, expect, afterAll } from "bun:test";
import { createQueryExpander } from "../src/query-expansion.js";

const expander = createQueryExpander();
afterAll(() => expander.dispose());

describe("QueryExpander (integration — downloads generation model)", () => {
  test("produces typed variants for a query", async () => {
    const out = await expander.expand("how to cancel my subscription");
    expect(out.length).toBeGreaterThan(0);
    for (const e of out) {
      expect(["lex", "vec", "hyde"]).toContain(e.type);
      expect(e.query.length).toBeGreaterThan(0);
    }
  }, 120_000);
});
