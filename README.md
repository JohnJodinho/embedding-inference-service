---
sdk: docker
app_port: 7860
---

# Embedding Inference Service

Shared CPU embedding inference API served from the existing Hugging Face Space
`JohnAlbarkaIbrahim/SentimentSpace`.

## Model registry

| Alias | Model | Revision | Dimensions | Licence |
|-------|-------|----------|------------|---------|
| `minilm-384-v1` | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | 384 | Apache-2.0 |
| `bge-base-768-v1` | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | 768 | MIT |

Both revisions are immutable commit hashes. CPU inference only; each model is
loaded once at startup.

## Endpoints

- `GET /health/live` — process liveness (no auth)
- `GET /health/ready` — model + worker readiness (no auth)
- `GET /v1/models` — supported aliases and dimensions (auth required)
- `POST /v1/embeddings` — embed one or many texts (auth required)
- `GET /metrics` — process-local metrics (auth required)

For `/v1/embeddings`, BGE requires `input_type` to be `query` or `document`.
MiniLM accepts an omitted `input_type` or `text`; unsupported values return 400.

## Authentication

Two distinct client secrets are configured as Hugging Face Space Secrets:

- `RAG_API_KEY` — for the existing RAG application (MiniLM)
- `ZIT_API_KEY` — for the ZIT assistant (BGE)

Requests present `Authorization: Bearer <key>`. Health endpoints are unauthenticated.

## Deployment

This repository is synced to the Hugging Face Space `JohnAlbarkaIbrahim/SentimentScope`
via GitHub Actions on every push to `main`. The Hugging Face Docker build installs
pinned dependencies, downloads both pinned models, runs the full test suite, and
only deploys if all tests pass.
