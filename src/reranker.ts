import {
  getLlama,
  resolveModelFile,
  type Llama,
  type LlamaModel,
  type LlamaRankingContext,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import type { Reranker } from "./types.js";

const DEFAULT_RERANK_MODEL =
  process.env.RELAI_RERANK_MODEL ??
  "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";

const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, "relai", "models")
  : join(homedir(), ".cache", "relai", "models");

export function createReranker(opts?: {
  model?: string;
  cacheDir?: string;
}): Reranker {
  const modelUri = opts?.model ?? DEFAULT_RERANK_MODEL;
  const cacheDir = opts?.cacheDir ?? MODEL_CACHE_DIR;

  let llama: Llama | null = null;
  let model: LlamaModel | null = null;
  let context: LlamaRankingContext | null = null;
  let loadPromise: Promise<void> | null = null;

  async function ensureLoaded() {
    if (context) return;
    if (loadPromise) return loadPromise;
    loadPromise = (async () => {
      if (!existsSync(cacheDir)) mkdirSync(cacheDir, { recursive: true });
      const modelPath = await resolveModelFile(modelUri, cacheDir);
      llama = await getLlama();
      model = await llama.loadModel({ modelPath });
      context = await model.createRankingContext();
    })();
    return loadPromise;
  }

  return {
    async rank(query: string, documents: string[]): Promise<number[]> {
      if (documents.length === 0) return [];
      await ensureLoaded();
      // rankAll returns a relevance score per document, in input order.
      return context!.rankAll(query, documents);
    },

    async dispose() {
      if (context) {
        await context.dispose();
        context = null;
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
