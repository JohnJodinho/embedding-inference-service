---
sdk: docker
app_port: 7860
---

# Embedding Inference Service

Shared CPU embedding inference API served from the existing Hugging Face Space
`JohnAlbarkaIbrahim/SentimentScope`.

For consumer integrations, see the [API Integration Handbook](API_INTEGRATION_HANDBOOK.md)
for authentication, endpoints, input limits, request/response formats, examples,
and retry guidance.

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
- `GET /v1/models` — supported aliases, dimensions, and per-model input limits
  (auth required)
- `POST /v1/embeddings` — embed one or many texts (auth required)
- `GET /metrics` — process-local metrics (auth required)

For `/v1/embeddings`, BGE requires `input_type` to be `query` or `document`.
MiniLM accepts an omitted `input_type` or `text`; unsupported values return 400.
Input text is checked against both the character limit and the model tokenizer's
token limit (including special tokens and the BGE query instruction); over-limit
input is rejected rather than silently truncated.

Batch requests accept at most 8 texts for MiniLM and 4 for BGE by default.
These limits also fit within the scheduler's maximum inference batch size, so
large requests cannot bypass its batching and fairness bounds. Requests above
the configured per-model limit return `400 too_many_texts`; clients should split
large document collections into bounded batches and retry failed batches
independently. BGE's smaller default reflects its 768-dimensional output and
the CPU-only Space; tune it only after measuring memory, latency, and queue
pressure on the deployed Space. `/v1/models` reports the active text, token,
and character limits so clients can adapt without duplicating server settings.

## Authentication

Two distinct client secrets are configured as Hugging Face Space Secrets:

- `RAG_API_KEY` — for the existing RAG application (MiniLM)
- `ZIT_API_KEY` — for the ZIT assistant (BGE)

Both configured keys currently authenticate to all protected endpoints and both
model aliases; the names identify clients for rate limiting and logs, not
model-level access restrictions.

Requests present `Authorization: Bearer <key>`. Health endpoints are unauthenticated.
In `/docs`, click **Authorize** and enter either API key; Swagger UI adds the
`Bearer` prefix for requests to protected endpoints.

An embeddings request must include a JSON body, for example:

```bash
curl -X POST "https://<space-host>/v1/embeddings" \
  -H "Authorization: Bearer <RAG_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"minilm-384-v1","input":"hello world","normalize_embeddings":true}'
```

For BGE retrieval queries, set `input_type` explicitly:

```bash
curl -X POST "https://<space-host>/v1/embeddings" \
  -H "Authorization: Bearer <ZIT_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-base-768-v1","input":"What courses does ZIT offer?","input_type":"query","normalize_embeddings":true}'
```

The response contains one ordered vector per input, along with the model alias,
dimension and `usage.input_texts`. The API returns 400 for invalid input, 401
for invalid credentials, 413 for an oversized body, 415 for non-JSON bodies,
429 for rate limits, 503 for unavailable/full inference queues, and 504 for
request or inference timeouts.

The request timeout is 90 seconds total and the model-inference timeout is 60
seconds. Queue capacity, batch limits, and inference-thread counts can be tuned
with the variables in `.env.example`; an overloaded queue returns 503, and a
timed-out request returns 504. `/metrics` exposes per-model request, queue-wait,
and inference latency histograms plus queue-depth gauges. `/v1/models` and
`/v1/embeddings` publish typed request and response schemas in `/docs`.

## Deployment

This repository is synced to the Hugging Face Space `JohnAlbarkaIbrahim/SentimentScope`
via GitHub Actions on every push to `main`. The Space's `main` branch is an
intentional deployment mirror: the workflow force-replaces its tree with GitHub
`main` so no merges or stale Space files are retained. The token is sent through
a masked Git HTTP authorization header, not embedded in a remote URL.

The Hugging Face Docker build installs pinned runtime dependencies, downloads
both pinned models, runs all tests with real-model smoke checks (none are
skipped), and only deploys if all tests pass. Application-source changes do not
invalidate the cached model-download layer.

Runtime and test dependencies are declared in `pyproject.toml` and pinned
transitively in `uv.lock`. Update direct pins and regenerate the lock with
`uv lock --python 3.11` before pushing dependency changes; the Docker build installs the
frozen runtime graph and keeps test-only packages out of the final image.
