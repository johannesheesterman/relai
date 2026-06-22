import {
  getLlama,
  resolveModelFile,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync } from "fs";
import type { Embedder } from "./types.js";

const DEFAULT_EMBED_MODEL =
  process.env.RELAI_EMBED_MODEL ??
  "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";

const MODEL_CACHE_DIR = process.env.XDG_CACHE_HOME
  ? join(process.env.XDG_CACHE_HOME, "relai", "models")
  : join(homedir(), ".cache", "relai", "models");

function formatDocForEmbedding(text: string): string {
  return `title: none | text: ${text}`;
}

function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

export async function pullModel(
  model?: string,
  cacheDir?: string
): Promise<string> {
  const modelUri = model ?? DEFAULT_EMBED_MODEL;
  const dir = cacheDir ?? MODEL_CACHE_DIR;
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  return resolveModelFile(modelUri, dir);
}

export function createEmbedder(options?: {
  model?: string;
  cacheDir?: string;
}): Embedder {
  const modelUri = options?.model ?? DEFAULT_EMBED_MODEL;
  const cacheDir = options?.cacheDir ?? MODEL_CACHE_DIR;

  let llama: Llama | null = null;
  let model: LlamaModel | null = null;
  let context: LlamaEmbeddingContext | null = null;
  let loadPromise: Promise<void> | null = null;

  async function ensureLoaded() {
    if (context) return;
    if (loadPromise) return loadPromise;
    loadPromise = (async () => {
      if (!existsSync(cacheDir)) mkdirSync(cacheDir, { recursive: true });
      const modelPath = await resolveModelFile(modelUri, cacheDir);
      llama = await getLlama();
      model = await llama.loadModel({ modelPath });
      context = await model.createEmbeddingContext();
    })();
    return loadPromise;
  }

  async function embedText(text: string): Promise<number[]> {
    await ensureLoaded();
    const result = await context!.getEmbeddingFor(text);
    return Array.from(result.vector);
  }

  return {
    async embed(text: string): Promise<number[]> {
      return embedText(formatDocForEmbedding(text));
    },

    async embedQuery(text: string): Promise<number[]> {
      return embedText(formatQueryForEmbedding(text));
    },

    async embedMany(texts: string[]): Promise<number[][]> {
      const out: number[][] = [];
      for (const t of texts) out.push(await embedText(formatDocForEmbedding(t)));
      return out;
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
