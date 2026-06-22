// src/chunker.ts
import type { Chunk } from "./types.js";

// Split into atomic blocks: fenced code blocks stay whole; prose splits on blank lines.
function toBlocks(text: string): { text: string; pos: number }[] {
  const blocks: { text: string; pos: number }[] = [];
  const fence = /```[\s\S]*?```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = fence.exec(text)) !== null) {
    if (m.index > last) {
      for (const b of splitProse(text.slice(last, m.index), last)) blocks.push(b);
    }
    blocks.push({ text: m[0], pos: m.index });
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    for (const b of splitProse(text.slice(last), last)) blocks.push(b);
  }
  return blocks;
}

function splitProse(slice: string, base: number): { text: string; pos: number }[] {
  const out: { text: string; pos: number }[] = [];
  const re = /\n\n+/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(slice)) !== null) {
    const piece = slice.slice(last, m.index);
    if (piece.trim()) out.push({ text: piece, pos: base + last });
    last = m.index + m[0].length;
  }
  const tail = slice.slice(last);
  if (tail.trim()) out.push({ text: tail, pos: base + last });
  return out;
}

export function chunkText(text: string, opts?: { maxChars?: number }): Chunk[] {
  const maxChars = opts?.maxChars ?? 3600;
  if (text.length <= maxChars) return [{ text, pos: 0 }];

  const blocks = toBlocks(text);
  const chunks: Chunk[] = [];
  let buf = "";
  let bufPos = -1;

  for (const block of blocks) {
    if (buf === "") {
      buf = block.text;
      bufPos = block.pos;
    } else if (buf.length + 2 + block.text.length <= maxChars) {
      buf += "\n\n" + block.text;
    } else {
      chunks.push({ text: buf, pos: bufPos });
      buf = block.text;
      bufPos = block.pos;
    }
  }
  if (buf !== "") chunks.push({ text: buf, pos: bufPos });
  return chunks.length > 0 ? chunks : [{ text, pos: 0 }];
}
