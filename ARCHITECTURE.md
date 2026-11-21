# Architecture — Embedding Inference Service

## Deployment topology

```text
Existing RAG application (Render)
        |
        | POST /v1/embeddings
        | model=minilm-384-v1
        | RAG_API_KEY
        v
Hugging Face Docker Space
        |
        +-- FastAPI API
        |
        +-- Authentication / validation / rate limits
        |
        +-- Bounded MiniLM queue
        |       |
        |       +-- all-MiniLM-L6-v2 (384 dimensions)
        |
        +-- Bounded BGE queue
                |
                +-- BAAI/bge-base-en-v1.5 (768 dimensions)

ZIT assistant backend
        |
        | POST /v1/embeddings
        | model=bge-base-768-v1
        | ZIT_API_KEY
        v
Same Hugging Face API
```

## Process model

One application process, one Uvicorn worker, two model instances, one controlled CPU inference worker. Do not start multiple Uvicorn workers because that duplicates model memory and in-process queue state.

Load models in FastAPI lifespan startup. Readiness remains false until both models and the inference worker are ready.

## Scheduling

Use two bounded queues and one inference worker. BGE gets preferential scheduling because it serves the public-facing ZIT assistant, while MiniLM traffic is expected to be low. Use a fairness rule so MiniLM cannot starve.

Batch multiple texts from a request. Micro-batch across waiting requests for a short, configurable collection window. Set strict queue capacity, per-request batch limits and timeouts. The total request deadline includes queue wait and inference; model inference has its own deadline. When capacity is exhausted, reject quickly with HTTP 503 and a retry hint. Do not create an unbounded backlog.

The single inference thread runs model encoding outside the async event loop. Timed-out Python thread work cannot be forcibly cancelled, so the worker waits for it to finish before starting another batch; this preserves the one-inference-at-a-time guarantee. Limit PyTorch intra-op and inter-op CPU threads to configured values. Benchmark these values on the actual Space after deployment; do not assume local performance.

## Model registry

| Alias | Repository | Output dimension |
|---|---|---:|
| `minilm-384-v1` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `bge-base-768-v1` | `BAAI/bge-base-en-v1.5` | 768 |

Both model revisions must be immutable commit hashes. The service rejects unknown aliases. The API never accepts arbitrary model repository names.

## Request semantics

- MiniLM input is passed unchanged. Normalization is controlled by the request to preserve existing RAG behaviour.
- BGE query input receives the model's documented retrieval query instruction.
- BGE document input receives no query instruction.
- BGE document and query vectors must use the same model revision and compatible normalization.
- Input order is preserved in the response.
- Validate the model-tokenizer count on the final formatted input (including special tokens and the BGE query prefix), without truncation. The configured token ceiling cannot exceed the models' 512-token sequence limit.
- Every response reports the model alias and dimensions.

## Reliability

- Liveness reports whether the process is alive.
- Readiness reports whether both models and the worker are ready.
- A model load failure prevents readiness and is logged with a safe error message.
- Requests are rejected while not ready.
- A single model inference error must not silently return a fabricated vector.
- Do not silently switch models or dimensions.
- Structured logs include request ID, client label, model alias, input count, queue wait, inference duration, total duration and status.
- Never log raw input, API keys, authorization headers or embeddings.

## Resource expectations

The existing Space is the fixed deployment target. Its currently allocated CPU and memory must be verified in its settings before deployment. The design uses CPU inference and one worker to avoid duplicate model memory and uncontrolled contention.

The architecture improves throughput through batching and admission control; it cannot create additional CPU capacity. If sustained demand exceeds the Space's inference capacity, requests must be rejected or the Space must be upgraded. Do not disguise overload by allowing queues to grow without bound.
