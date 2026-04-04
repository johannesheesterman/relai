#!/usr/bin/env bun
import { Relai } from "../relai.js";
import { pullModel } from "../embedder.js";

const args = process.argv.slice(2);
const command = args[0];

function flag(name: string): string | undefined {
  const idx = args.indexOf(name);
  if (idx === -1 || idx + 1 >= args.length) return undefined;
  return args[idx + 1];
}

function usage() {
  console.log(`relai - semantic indexing layer

Usage:
  relai index --source <s> --remoteId <r> --text <t> [--type <t>]   Index a view
  relai search <query> [-k <n>]                                      Semantic search
  relai claim <from> <type> <to>                                     Create a claim
  relai related <viewId>                                             Show claims
  relai pull                                                         Download model
`);
}

function createRelai() {
  return new Relai();
}

async function main() {
  switch (command) {
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
        console.error("Usage: relai search <query> [-k <n>]");
        process.exit(1);
      }
      const k = parseInt(flag("-k") ?? "5", 10);
      const relai = createRelai();
      try {
        const views = await relai.search(query, k);
        if (views.length === 0) {
          console.log("No results found.");
        } else {
          for (const v of views) {
            console.log(`[${v.id}] ${v.text.slice(0, 120)}`);
          }
        }
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "claim": {
      const from = args[1];
      const type = args[2];
      const to = args[3];
      if (!from || !type || !to) {
        console.error("Usage: relai claim <from> <type> <to>");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        const claim = await relai.claim(from, type, to);
        console.log(`created ${claim.id}`);
      } finally {
        await relai.dispose();
      }
      break;
    }

    case "related": {
      const viewId = args[1];
      if (!viewId) {
        console.error("Usage: relai related <viewId>");
        process.exit(1);
      }
      const relai = createRelai();
      try {
        const claims = await relai.related(viewId);
        if (claims.length === 0) {
          console.log("No claims found.");
        } else {
          for (const c of claims) {
            console.log(`${c.from} --[${c.type}]--> ${c.to}`);
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
