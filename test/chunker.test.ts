// test/chunker.test.ts
import { describe, test, expect } from "bun:test";
import { chunkText } from "../src/chunker.js";

describe("chunkText", () => {
  test("short text → one chunk at pos 0", () => {
    const chunks = chunkText("hello world");
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toEqual({ text: "hello world", pos: 0 });
  });

  test("long text splits into multiple chunks under the limit", () => {
    const para = "Sentence about the topic. ".repeat(40); // ~1040 chars
    const text = (para + "\n\n").repeat(8); // ~8.3k chars
    const chunks = chunkText(text, { maxChars: 1500 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(1600);
  });

  test("does not split inside a fenced code block", () => {
    const code = "```\n" + "const x = 1;\n".repeat(50) + "```";
    const text = "Intro paragraph.\n\n" + code + "\n\nOutro paragraph.";
    const chunks = chunkText(text, { maxChars: 200 });
    const codeChunk = chunks.find((c) => c.text.includes("const x = 1;"));
    // The whole code block stays inside a single chunk.
    expect(codeChunk?.text.match(/const x = 1;/g)?.length).toBe(50);
  });

  test("pos is the character offset into the original text", () => {
    const text = "AAAA\n\nBBBB";
    const chunks = chunkText(text, { maxChars: 4 });
    expect(chunks[1]?.pos).toBe(text.indexOf("BBBB"));
  });
});
