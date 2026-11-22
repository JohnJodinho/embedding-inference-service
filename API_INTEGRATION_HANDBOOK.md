# Embedding Inference Service: API Integration Handbook

This handbook is for application developers integrating the service. It covers
authentication, model selection, request and response formats, limits, retries,
and production-use examples. It does not describe deployment or internal
development workflows.

## 1. Service address

The current Hugging Face Space API base URL is:

```text
https://johnalbarkaibrahim-sentimentscope.hf.space
```

Keep the base URL configurable in your application rather than repeating it
throughout your code. Use HTTPS and do not send API keys over plain HTTP.

Interactive API documentation is available at `/docs`; the OpenAPI document is
at `/openapi.json`.

## 2. Authentication

Protected endpoints require an API key in the HTTP Bearer authorization header:

```http
Authorization: Bearer <API_KEY>
```

The service accepts the keys configured as Hugging Face Space Secrets:

- `RAG_API_KEY`
- `ZIT_API_KEY`

Both configured keys currently authenticate to all protected endpoints and
both model aliases. The key names identify the consuming client for
rate-limiting and logs; they do not restrict access to a particular model.
Keep each key on a trusted server, store it in your application's secret
manager or environment, and rotate it if exposed. Never put it in browser
JavaScript, a mobile app bundle, source control, or client-visible logs.

The health endpoints are public. Model discovery, inference, and metrics
endpoints require authentication. In Swagger UI at `/docs`, use **Authorize**
and enter the raw key; Swagger adds the Bearer scheme.

## 3. Endpoints

| Method | Path | Authentication | Purpose |
|---|---|---|---|
| `GET` | `/health/live` | None | Process liveness; returns plain-text `ok` |
| `GET` | `/health/ready` | None | Readiness; `200` when models and worker are ready, otherwise `503` |
| `GET` | `/v1/models` | Bearer key | List model aliases, dimensions, and active input limits |
| `POST` | `/v1/embeddings` | Bearer key | Generate one or more embeddings |
| `GET` | `/metrics` | Bearer key | Read Prometheus-format process metrics |

Use `/health/ready` for readiness checks before sending inference traffic.
`/health/live` only confirms that the process is responding; it does not mean
the models are ready.

### Discover models and limits

Call `/v1/models` with authentication instead of hard-coding limits:

```bash
curl "https://johnalbarkaibrahim-sentimentscope.hf.space/v1/models" \
  -H "Authorization: Bearer $EMBEDDING_API_KEY"
```

The response contains a `data` array. Each model entry includes:

- `alias`: the API model name to use in inference requests
- `model_id`, `revision`, `dimensions`, and `license`
- `max_input_texts`: maximum number of texts in one request
- `max_tokens_per_text` and `max_chars_per_text`

Current defaults are 8 texts per MiniLM request and 4 per BGE request. Read
`max_input_texts` at runtime so a service configuration change does not require
an application release.

## 4. Models and input behavior

| API alias | Output dimensions | `input_type` | Input handling |
|---|---:|---|---|
| `minilm-384-v1` | 384 | Omit it or use `"text"` | Text is passed to the model unchanged |
| `bge-base-768-v1` | 768 | Required: `"query"` or `"document"` | For queries, the service adds the BGE retrieval instruction; documents are passed without that instruction |

For BGE, do not add the query instruction yourself. Select `"query"` for
retrieval queries and `"document"` for passages being indexed. The API rejects
BGE requests without a supported `input_type`.

Both models return one vector per input text in input order. The selected model
must be the same model used by the downstream application when comparing or
retrieving embeddings.

## 5. Embeddings request

Send JSON to `POST /v1/embeddings` with `Content-Type: application/json`.

```json
{
  "model": "bge-base-768-v1",
  "input": [
    "What courses does the institute offer?",
    "When does registration open?"
  ],
  "input_type": "query",
  "normalize_embeddings": true
}
```

### Fields

| Field | Required | Type | Description |
|---|---|---|---|
| `model` | Yes | String | `minilm-384-v1` or `bge-base-768-v1` |
| `input` | Yes | String or array of strings | One text or a non-empty batch of texts |
| `input_type` | BGE only | String | BGE: `"query"` or `"document"`; MiniLM: omit or use `"text"` |
| `normalize_embeddings` | No | Boolean | L2-normalize output vectors; defaults to `false` |

Unknown fields are rejected. Each input must contain non-whitespace text.

### Input limits

The service applies all of these limits:

- Per-call text count: reported as `max_input_texts` by `/v1/models` (defaults:
  MiniLM 8, BGE 4).
- Characters per text: 2,048 by default.
- Tokens per text: 512 by default, counted with the selected model's tokenizer.
  Special tokens and the BGE query instruction are included in the count.
- HTTP request body: 262,144 bytes (256 KiB) by default.

Oversized text is rejected; it is not silently truncated. Split longer
collections into calls that stay within the discovered per-call limit. The
limit is the number of texts in one API call, not the number of requests that
may be queued. Multiple small requests can be queued and may be combined by the
inference scheduler, subject to its capacity and available processing time.
By default, each model has a queue capacity of 32 waiting requests, and a
scheduler batch is limited to 8 texts overall. Queue capacity counts requests,
not texts. The service uses a bounded CPU inference worker, so concurrent calls
may wait in the queue rather than running model inference in parallel.

### cURL examples

MiniLM, one text:

```bash
curl --fail-with-body \
  "https://johnalbarkaibrahim-sentimentscope.hf.space/v1/embeddings" \
  -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minilm-384-v1",
    "input": "A sample passage to embed.",
    "normalize_embeddings": true
  }'
```

BGE, a batch of query texts:

```bash
curl --fail-with-body \
  "https://johnalbarkaibrahim-sentimentscope.hf.space/v1/embeddings" \
  -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-base-768-v1",
    "input": ["First search question", "Another search question"],
    "input_type": "query",
    "normalize_embeddings": true
  }'
```

For document embeddings, use `"input_type": "document"`. The maximum BGE input
array length is available from `/v1/models`.

## 6. Response format

A successful request returns an object with:

- `object`: `"list"`
- `model`: the model alias used
- `dimensions`: vector length for that model
- `data`: ordered embedding items with a zero-based `index`
- `usage.input_texts`: number of input texts

Illustrative response shape (the actual values are 768-number arrays):

```jsonc
{
  "object": "list",
  "model": "bge-base-768-v1",
  "dimensions": 768,
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456 /* ... 766 more floats ... */]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0789, 0.0123 /* ... 766 more floats ... */]
    }
  ],
  "usage": {"input_texts": 2}
}
```

Match results to inputs by array order or the returned `index`. Each vector
must be stored and compared with its model alias and dimension. If you request
normalized vectors, their dot product is equivalent to cosine similarity;
otherwise normalize them in your application before using dot product as
cosine similarity.

## 7. Python integration example

This example discovers the active batch limit, splits a larger collection
into valid requests, and preserves input order. Install `requests` in the
consumer application if it is not already available.

```python
import os

import requests

BASE_URL = os.environ.get(
    "EMBEDDING_API_BASE_URL",
    "https://johnalbarkaibrahim-sentimentscope.hf.space",
).rstrip("/")
API_KEY = os.environ["EMBEDDING_API_KEY"]

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
)


def embed_texts(
    texts: list[str],
    model: str,
    input_type: str | None = None,
) -> list[list[float]]:
    if not texts:
        raise ValueError("texts must not be empty")

    response = session.get(f"{BASE_URL}/v1/models", timeout=(5, 15))
    response.raise_for_status()
    model_info = next(
        item for item in response.json()["data"] if item["alias"] == model
    )
    batch_limit = model_info["max_input_texts"]

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_limit):
        batch = texts[start : start + batch_limit]
        body = {
            "model": model,
            "input": batch,
            "normalize_embeddings": True,
        }
        if input_type is not None:
            body["input_type"] = input_type

        result = session.post(
            f"{BASE_URL}/v1/embeddings",
            json=body,
            timeout=(5, 100),
        )
        result.raise_for_status()
        items = result.json()["data"]
        if len(items) != len(batch):
            raise RuntimeError("Embedding response count did not match request")
        embeddings.extend(item["embedding"] for item in items)

    return embeddings


query_vectors = embed_texts(
    ["How do I apply?", "When is the next intake?"],
    model="bge-base-768-v1",
    input_type="query",
)
```

For MiniLM, call `embed_texts(texts, model="minilm-384-v1")` and omit
`input_type`. Do not log or print `API_KEY`. If a later chunk fails, handle
checkpointing or retrying at the application level if redoing earlier chunks
would be expensive.

## 8. Status codes, errors, and retries

| Status | Meaning | Client action |
|---:|---|---|
| `200` | Request succeeded | Validate model, dimensions, and item count before storing results |
| `400` | Invalid JSON, model, input type, text, or configured input limit | Correct the request; do not retry unchanged |
| `401` | Missing or invalid API key | Check the secret and Bearer header |
| `413` | Request body exceeds the byte limit | Reduce the number or size of texts per call |
| `415` | Request is not sent as `application/json` | Set the JSON content type |
| `429` | Client request rate limit reached | Wait for `Retry-After`, then retry with bounded backoff |
| `503` | Service not ready, inference unavailable, or queue full | Check readiness; when present, honor `Retry-After` and retry with bounded backoff |
| `504` | Request or inference timeout | Retry only a bounded number of times with backoff |

Embedding validation and overload errors use a JSON `error` field, for example
`{"error":"too_many_tokens"}`. Authentication failures use FastAPI's
`detail` field, for example `{"detail":"invalid_auth"}`. Use the response
status and error code rather than matching human-readable text.

The default per-client request rate limit is 60 HTTP requests per minute. A
batch containing several texts still counts as one HTTP request. Rate limiting
and queue capacity are separate: a request can pass the rate limit but still
receive `503` if the inference queue is full. `Retry-After` is supplied for
rate limiting and some readiness/queue errors; otherwise use a short,
bounded exponential backoff with jitter. Avoid unlimited retries.

## 9. Operational integration guidance

- Check `/health/ready` during startup and after transient service errors.
- Reuse an HTTP connection/session; set a client timeout longer than the
  service's 90-second total request timeout.
- Use the largest batch that fits the active model's advertised limit to
  reduce HTTP overhead, but keep batches bounded.
- Limit caller-side concurrency. A bounded service queue applies backpressure;
do not respond to `503` by immediately increasing parallel requests.
- On `429` or `503`, honor `Retry-After` when supplied. Apply bounded retries
  and jitter so multiple clients do not retry simultaneously.
- For BGE retrieval, embed corpus passages with `input_type: "document"` and
  search text with `input_type: "query"`.
- Store the model alias and vector dimension alongside each vector. Do not
  compare vectors from different models or dimensions.
- Treat `/metrics` as an authenticated operational endpoint, not an
  application data API. It returns process-local Prometheus metrics.
- Use the response `X-Request-ID` header when contacting the service operator
  about a failed request. Do not include API keys in support messages.

## 10. Troubleshooting

### `401 Unauthorized`

Send `Authorization: Bearer <key>` to protected endpoints. Check that the key
is current and configured as a Space Secret. Health routes do not need a key;
`/v1/models`, `/v1/embeddings`, and `/metrics` do.

### `400 too_many_texts`

The array exceeds the model's `max_input_texts`. Read the current limit from
`/v1/models` and split the input into smaller calls.

### `400 too_many_tokens` or `400 text_too_long`

The text exceeds the token or character limit. Shorten or split it in the
calling application; the service does not truncate input.

### `400 input_type_required` or `400 invalid_input_type`

Supply `"query"` or `"document"` for BGE. MiniLM accepts an omitted type or
`"text"`.

### `429 Too Many Requests`

The client has exceeded its request rate. Wait for `Retry-After` and retry
with bounded backoff. Batching texts within the advertised limit can reduce
the number of HTTP requests.

### `503 Service Unavailable`

The model worker may not be ready or the bounded inference queue may be full.
Check `/health/ready`; retry queue pressure after the indicated delay rather
than immediately resending at higher concurrency.

### `504 Gateway Timeout`

The request exceeded a service timeout. Check client connectivity and retry
only with bounded backoff. If this persists, provide the request ID and
timestamp to the service operator, without sharing the API key.
