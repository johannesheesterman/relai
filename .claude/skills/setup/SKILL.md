---
name: setup
description: Set up relai — install dependencies, download embedding model, configure database
disable-model-invocation: true
---

## Install dependencies

```bash
bun install
```

## macOS prerequisite

sqlite-vec needs Homebrew's SQLite:

```bash
brew install sqlite
```

## Download embedding model

Pre-download (~300MB):

```bash
bun run src/cli/relai.ts pull
```

Or it auto-downloads on first `index` or `search`.

## Configuration

**Database**: defaults to `~/.config/relai/relai.sqlite`. Override:

```ts
const relai = new Relai({ dbPath: "./my-project.sqlite" });
```

**Embedding model**: defaults to `embeddinggemma-300M`. Override:

```bash
RELAI_EMBED_MODEL="hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
```

Models are cached in `~/.cache/relai/models/`.

## Run tests

```bash
bun test
```
