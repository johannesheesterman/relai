---
name: claims
description: Create and query semantic relationships (claims) between views
argument-hint: [from] [type] [to]
---

Create or query relationships between views.

## Create a claim

```bash
bun run src/cli/relai.ts claim $0 $1 $2
```

## Query claims for a view

```bash
bun run src/cli/relai.ts related $0
```

## Library

```ts
import { Relai } from "./src/index.js";

const relai = new Relai();

// Create a claim
await relai.claim(
  "view:files:notes/acme-renewal.md",
  "mentions",
  "view:sqlite:customers/42",
  "agent:sales"  // optional createdBy
);

// Get all claims involving a view (both directions)
const claims = await relai.related("view:sqlite:customers/42");
for (const c of claims) {
  console.log(`${c.from} --[${c.type}]--> ${c.to}`);
}
await relai.dispose();
```

## Common claim types

- `mentions` — one view references another
- `same_as` — two views represent the same real-world entity
- `related_to` — general relationship
- `part_of` — hierarchical containment
- `caused_by` — causal relationship

Any string works as a claim type — these are conventions, not enforced.

## Key concepts

- Claims are directional: `from --[type]--> to`
- `related(viewId)` returns claims in both directions
- `createdBy` is optional — use it to track which agent created the claim

## Claims are searchable

When a claim is created, relai automatically generates an embedded view for it. The claim view's text combines both endpoint views' text with the claim type (e.g. `"Ticket about dark mode type_of Feature request: new functionality"`). This means:

- Searching "feature request" will find claim views linking tickets to the feature-request type
- Search results with `type: "claim"` have `links` pointing back to the `from` and `to` views
- No special query syntax needed — claims are found through the same semantic search as everything else
