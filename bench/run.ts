import { Relai } from "../src/relai.js";
import { precisionAtK, recallAtK, f1AtK } from "./score.js";
import { readFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

type Dataset = {
  docs: { source: string; remoteId: string; type?: string; text: string }[];
  queries: { query: string; relevant: string[] }[];
};

async function main() {
  const data: Dataset = JSON.parse(
    readFileSync(new URL("./dataset.json", import.meta.url), "utf8")
  );
  const dbPath = join(tmpdir(), `relai-bench-${process.pid}.sqlite`);
  const rerank = process.env.BENCH_RERANK === "1";
  const relai = new Relai({ dbPath, rerank });

  for (const d of data.docs) await relai.index(d);

  const k = 5;
  let p = 0, r = 0, f = 0;
  for (const q of data.queries) {
    const results = await relai.search(q.query, k, { rerank });
    const ids = results.map((v) => v.id);
    p += precisionAtK(ids, q.relevant, k);
    r += recallAtK(ids, q.relevant, k);
    f += f1AtK(ids, q.relevant, k);
  }
  const n = data.queries.length;
  console.log(`queries=${n}  P@${k}=${(p / n).toFixed(3)}  R@${k}=${(r / n).toFixed(3)}  F1@${k}=${(f / n).toFixed(3)}`);
  await relai.dispose();
}

main();
