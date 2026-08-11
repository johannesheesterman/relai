import {
  getLlama,
  resolveModelFile,
  LlamaChatSession,
  type Llama,
  type LlamaModel,
  type LlamaContext,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import type { QueryExpander, ExpandedQuery } from "./types.js";

const DEFAULT_GENERATE_MODEL =
  process.env.RELAI_GENERATE_MODEL ??
  "hf:ggml-org/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf";

const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, "relai", "models")
  : join(homedir(), ".cache", "relai", "models");

const SYSTEM = `You expand a search query into alternative queries to improve retrieval.
Return ONLY a JSON array. Each element: {"type":"lex|vec|hyde","query":"..."}.
- "lex": a keyword-only reformulation (synonyms, key nouns).
- "vec": a natural-language paraphrase of the intent.
- "hyde": a one-sentence hypothetical answer document.
Return 3 elements total, one of each type. No prose, no markdown.`;

export function createQueryExpander(opts?: {
  model?: string;
  cacheDir?: string;
}): QueryExpander {
  const modelUri = opts?.model ?? DEFAULT_GENERATE_MODEL;
  const cacheDir = opts?.cacheDir ?? MODEL_CACHE_DIR;

  let llama: Llama | null = null;
  let model: LlamaModel | null = null;
  let ctx: LlamaContext | null = null;
  let loadPromise: Promise<void> | null = null;

  async function ensureLoaded() {
    if (ctx) return;
    if (loadPromise) return loadPromise;
    loadPromise = (async () => {
      if (!existsSync(cacheDir)) mkdirSync(cacheDir, { recursive: true });
      const modelPath = await resolveModelFile(modelUri, cacheDir);
      llama = await getLlama();
      model = await llama.loadModel({ modelPath });
      ctx = await model.createContext();
    })();
    return loadPromise;
  }

  function parse(raw: string): ExpandedQuery[] {
    try {
      const start = raw.indexOf("[");
      const end = raw.lastIndexOf("]");
      if (start === -1 || end === -1) return [];
      const arr = JSON.parse(raw.slice(start, end + 1));
      if (!Array.isArray(arr)) return [];
      return arr
        .filter(
          (e) =>
            e &&
            typeof e.query === "string" &&
            ["lex", "vec", "hyde"].includes(e.type)
        )
        .map((e) => ({ type: e.type as ExpandedQuery["type"], query: e.query.trim() }))
        .filter((e) => e.query.length > 0);
    } catch {
      return [];
    }
  }

  return {
    async expand(query: string): Promise<ExpandedQuery[]> {
      try {
        await ensureLoaded();
        const session = new LlamaChatSession({
          contextSequence: ctx!.getSequence(),
        });
        const raw = await session.prompt(`${SYSTEM}\n\nQuery: ${query}`, {
          temperature: 0.2,
          maxTokens: 256,
        });
        return parse(raw);
      } catch {
        return [];
      }
    },

    async dispose() {
      if (ctx) {
        await ctx.dispose();
        ctx = null;
      }
      if (model) {
        await model.dispose();
        model = null;
      }
      if (llama) {
        await llama.dispose();
        llama = null;
      }
      loadPromise = null;
    },
  };
}
