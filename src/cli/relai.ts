#!/usr/bin/env bun
import { Relai } from "../relai.js";
import { pullModel } from "../embedder.js";
import { ref, type ClaimObject, type SearchOptions } from "../types.js";

const RERANK_MODEL =
  process.env.RELAI_RERANK_MODEL ??
  "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const GENERATE_MODEL =
  process.env.RELAI_GENERATE_MODEL ??
  "hf:ggml-org/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf";

const args = process.argv.slice(2);
const command = args[0];

function flag(name: string): string | undefined {
  const idx = args.indexOf(name);
  if (idx === -1 || idx + 1 >= args.length) return undefined;
  return args[idx + 1];
}

function hasFlag(name: string): boolean {
  return args.includes(name);
}

function usage() {
  console.log(`relai - tiny semantic index

Usage:
  relai put <id> --text <text>                                      Store searchable text
  relai index --source <s> --remoteId <r> --text <t> [--type <t>]   Index a view
  relai search <query> [-k <n>] [--no-rerank] [--no-expand] [--explain]   Hybrid search
  relai claim <subject> <predicate> --ref <id>                      Add a reference claim
  relai claim <subject> <predicate> --value <json>                  Add a literal claim
  relai unclaim <subject> [predicate] [--ref <id>|--value <json>]   Remove claims
  relai match [--subject <id>] [--predicate <p>] [--ref <id>|--value <json>]
  relai describe <id>                                               Show outgoing and incoming claims
  relai related <id>                                                Show claims touching an id
  relai pull [--all]                                                Download models
  relai bench                                                       Run benchmark
`);
}

function createRelai() {
  return new Relai();
}

function claimObjectFromFlags(): ClaimObject | undefined {
  const refValue = flag("--ref");
  const rawValue = flag("--value");

  if (refValue && rawValue) {
    throw new Error("Use either --ref or --value, not both.");
  }

  if (refValue) return ref(refValue);
  if (rawValue !== undefined) return JSON.parse(rawValue) as ClaimObject;
  return undefined;
}

function formatObject(object: ClaimObject): string {
  if (typeof object === "object" && object !== null && "ref" in object) {
    return object.ref;
  }
  return JSON.stringify(object);
}

async function main() {
  switch (command) {
    case "put": {
      const id = args[1];
      const text = flag("--text");
      if (!id || !text) {
        console.error("Usage: relai put <id> --text <text>");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        const thing = await relai.put(id, text);
        console.log(`put ${thing.id}`);
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "index": {
      const source = flag("--source");
      const remoteId = flag("--remoteId");
      const text = flag("--text");
      const type = flag("--type");
      if (!source || !remoteId || !text) {
        console.error(
          "Usage: relai index --source <s> --remoteId <r> --text <t> [--type <t>]"
        );
        process.exit(1);
      }
      const relai = createRelai();
      try {
        const view = await relai.index({ source, remoteId, text, type });
        console.log(`indexed ${view.id}`);
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "search": {
      const query = args[1];
      if (!query) {
        console.error(
          "Usage: relai search <query> [-k <n>] [--no-rerank] [--no-expand] [--explain]"
        );
        process.exit(1);
      }
      const k = parseInt(flag("-k") ?? "5", 10);
      const options: SearchOptions = {};
      if (hasFlag("--no-rerank")) options.rerank = false;
      if (hasFlag("--no-expand")) options.expand = false;
      const explain = hasFlag("--explain");
      const relai = createRelai();
      try {
        const views = await relai.search(query, k, options);
        if (views.length === 0) {
          console.log("No results found.");
        } else {
          for (const v of views) {
            console.log(`[${v.id}] ${v.text.slice(0, 120)}`);
          }
        }
        if (explain) {
          console.error(
            `(rerank=${options.rerank !== false}, expand=${options.expand !== false}, k=${k}, results=${views.length})`
          );
        }
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "claim": {
      const subject = args[1];
      const predicate = args[2];
      const object = claimObjectFromFlags();
      if (!subject || !predicate || object === undefined) {
        console.error("Usage: relai claim <subject> <predicate> --ref <id>|--value <json>");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        await relai.claim(subject, predicate, object);
        console.log(`claimed ${subject} ${predicate} ${formatObject(object)}`);
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "unclaim": {
      const subject = args[1];
      const predicate = args[2];
      const object = claimObjectFromFlags();
      if (!subject) {
        console.error("Usage: relai unclaim <subject> [predicate] [--ref <id>|--value <json>]");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        await relai.unclaim(subject, predicate, object);
        console.log("unclaimed");
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "match": {
      const subject = flag("--subject");
      const predicate = flag("--predicate");
      const object = claimObjectFromFlags();
      const relai = createRelai();
      try {
        const claims = await relai.match({ subject, predicate, object });
        if (claims.length === 0) {
          console.log("No claims found.");
        } else {
          for (const c of claims) {
            console.log(`${c.subject} --[${c.predicate}]--> ${formatObject(c.object)}`);
          }
        }
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "describe": {
      const id = args[1];
      if (!id) {
        console.error("Usage: relai describe <id>");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        const description = await relai.describe(id);
        if (description.thing) {
          console.log(`[${description.thing.id}] ${description.thing.text}`);
        } else {
          console.log(`[${id}]`);
        }
        for (const c of description.claims) {
          console.log(`  ${c.predicate} -> ${formatObject(c.object)}`);
        }
        for (const c of description.incoming) {
          console.log(`  <- ${c.subject} --[${c.predicate}]`);
        }
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "related": {
      const viewId = args[1];
      if (!viewId) {
        console.error("Usage: relai related <id>");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        const claims = await relai.related(viewId);
        if (claims.length === 0) {
          console.log("No claims found.");
        } else {
          for (const c of claims) {
            console.log(`${c.subject} --[${c.predicate}]--> ${formatObject(c.object)}`);
          }
        }
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "pull": {
      console.log("Downloading embedding model...");
      const path = await pullModel();
      console.log(`Model ready: ${path}`);
      if (hasFlag("--all")) {
        console.log("Downloading reranker model...");
        const rerankPath = await pullModel(RERANK_MODEL);
        console.log(`Model ready: ${rerankPath}`);
        console.log("Downloading generation model...");
        const generatePath = await pullModel(GENERATE_MODEL);
        console.log(`Model ready: ${generatePath}`);
      }
      break;
    }

    case "bench": {
      const proc = Bun.spawn(["bun", "run", "bench/run.ts"], {
        stdout: "inherit",
        stderr: "inherit",
      });
      const code = await proc.exited;
      if (code !== 0) process.exit(code);
      break;
    }

    default:
      usage();
      break;
  }
}

main().catch((err) => {
  console.error(err.message ?? err);
  process.exit(1);
});
