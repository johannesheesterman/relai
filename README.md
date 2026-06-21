# relai

A tiny local index for AI-readable things.

relai does not store raw data. It stores:

- **Things**: ids with searchable text
- **Claims**: simple facts about things
- **Vectors**: embeddings for semantic search

The core loop is:

```text
put text
claim facts
search meaning
match structure
```

## Install

```bash
bun install
```

On macOS, sqlite-vec requires Homebrew's SQLite:

```bash
brew install sqlite
```

## Quick start

### Library

```ts
import { Relai, ref } from "relai";

const relai = new Relai();

await relai.put(
  "customer:acme",
  "Customer Acme. Enterprise account using Acme Analytics."
);

await relai.put(
  "hubspot:ticket:987654321",
  "Support ticket for Acme: monthly analytics CSV export fails."
);

await relai.put(
  "jira:issue:APP-123",
  "Jira issue APP-123: CSV export times out for large analytics reports."
);

await relai.put(
  "clickup:task:86abc",
  "ClickUp task for Acme: validate monthly reporting export workflow."
);

await relai.claim("customer:acme", "type", "Customer");
await relai.claim("customer:acme", "name", "Acme");

await relai.claim("hubspot:ticket:987654321", "customer", ref("customer:acme"));
await relai.claim("jira:issue:APP-123", "customer", ref("customer:acme"));
await relai.claim("clickup:task:86abc", "customer", ref("customer:acme"));
await relai.claim("hubspot:ticket:987654321", "causedBy", ref("jira:issue:APP-123"));

const results = await relai.search("Acme export problem");
const acmeItems = await relai.match({
  predicate: "customer",
  object: ref("customer:acme"),
});
const context = await relai.describe("customer:acme");

await relai.dispose();
```

### CLI

```bash
# Download the embedding model (~300MB, runs locally)
bun run src/cli/relai.ts pull

# Store searchable text
bun run src/cli/relai.ts put customer:acme \
  --text "Customer Acme. Enterprise account."

bun run src/cli/relai.ts put hubspot:ticket:987654321 \
  --text "Support ticket for Acme: CSV export fails."

# Add claims
bun run src/cli/relai.ts claim customer:acme type --value '"Customer"'
bun run src/cli/relai.ts claim hubspot:ticket:987654321 customer --ref customer:acme
bun run src/cli/relai.ts claim hubspot:ticket:987654321 priority --value '"high"'

# Search text
bun run src/cli/relai.ts search "Acme export problem"

# Match structure
bun run src/cli/relai.ts match --predicate customer --ref customer:acme

# Expand context
bun run src/cli/relai.ts describe customer:acme
```

## Core concepts

### Thing

A thing is anything with an id and searchable text:

```ts
await relai.put("jira:issue:APP-123", "CSV export job times out.");
```

Things can be external objects, like HubSpot tickets or Jira issues, or internal anchors, like `customer:acme`.

### Claim

A claim is a fact about a thing:

```text
subject predicate object
```

The object can be a literal value:

```ts
await relai.claim("hubspot:ticket:987654321", "priority", "high");
```

or a reference to another thing:

```ts
await relai.claim("hubspot:ticket:987654321", "customer", ref("customer:acme"));
```

Document properties and relationships are both claims.

## Architecture

- **Storage**: Single SQLite file (`~/.config/relai/relai.sqlite`)
- **Vectors**: [sqlite-vec](https://github.com/asg017/sqlite-vec) for KNN cosine similarity search
- **Embeddings**: Local via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp) — default model is `embeddinggemma-300M` (GGUF from HuggingFace)
- **Runtime**: Bun (also works with Node.js via better-sqlite3)

## Tests

```bash
bun test
```

## License

MIT
