# Implementation Plan — Shared Embedding Inference API

## 1. Goal

Replace the copied RAG application workspace with a dedicated embedding API hosted in the existing Hugging Face Docker Space. The original RAG application remains separate and will run on Render.

The API serves two models to two consumers:

| Consumer | Model alias | Model | Dimensions |
|---|---|---|---:|
| Existing RAG application on Render | `minilm-384-v1` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| ZIT website assistant | `bge-base-768-v1` | `BAAI/bge-base-en-v1.5` | 768 |

## 2. Fixed deployment decisions

- Keep the existing Space: `JohnAlbarkaIbrahim/SentimentScope`.
- Do not rename or transfer the Space.
- New GitHub repository: `JohnJodinho/embedding-inference-service`.
- GitHub `main` is the source of truth.
- GitHub Actions synchronises `main` to the existing Space.
- Hugging Face Docker build downloads the models and runs the tests.
- No local model downloads, local inference or local test execution.
- Python 3.11, FastAPI, Sentence Transformers, CPU inference.
- Port 7860; one Uvicorn worker.
- Run as non-root UID 1000.

## 3. Model behaviour

### MiniLM

- 384 dimensions.
- No added prompt/instruction.
- Request controls normalization explicitly.
- Preserve existing RAG call-site behaviour:
  - Supabase methodology query: `normalize_embeddings=false`.
  - Tool registry descriptions: `true`.
  - Tool ranking query: `true`.
  - RAG question retrieval: `true`.

### BGE

- 768 dimensions.
- Query prefix: `Represent this sentence for searching relevant passages: `
- Document inputs receive no query prefix.
- Query and document vectors use the same model revision.
- Normalization is explicit and consistent between document and query ingestion.
- Pin immutable model commit revisions.

## 4. Runtime design

- One FastAPI process and one Uvicorn worker.
- Load both models once during FastAPI lifespan startup.
- One central CPU inference worker.
- Two bounded queues, one per model.
- BGE receives higher scheduling weight; MiniLM must not starve.
- Batch inputs and use a short configurable micro-batching window.
- Reject overload rather than accepting unlimited work.
- Do not introduce Redis, Celery, Kafka, a database or multiple model-serving processes.

The scheduler must be small, deterministic and covered by tests. If implementation complexity becomes disproportionate, use a bounded single-worker FIFO queue with per-model batch limits rather than inventing a complicated scheduler.

## 5. HTTP API

| Method | Route | Authentication | Purpose |
|---|---|---|---|
| POST | `/v1/embeddings` | Required | Embed one or many texts |
| GET | `/v1/models` | Required | Return supported aliases and dimensions |
| GET | `/health/live` | No | Process liveness |
| GET | `/health/ready` | No | Model and worker readiness |
| GET | `/metrics` | Required | Lightweight service metrics |

Use an OpenAI-style request shape with the documented supported fields:
- `model`
- `input` (string or list of strings)
- `input_type` (`query` or `document` for BGE; `text` or omitted for MiniLM)
- `normalize_embeddings` (boolean)

Preserve input order in the response. Include model alias, dimensions, vector index, embeddings and input-text count.

## 6. Security

- Two distinct client secrets: one for RAG, one for ZIT.
- Configure them as Hugging Face Space Secrets.
- Use constant-time credential comparison.
- Never log keys, authorization headers, raw input text or embeddings.
- Apply request-size, text-count, character-count and token-count limits.
- Apply per-client rate limits.
- Use bounded queues and timeouts.
- Return clear HTTP status codes for invalid input, authentication failure and overload.

## 7. Testing strategy

All automated tests run during every Hugging Face Docker image build. The Docker build fails if any test fails.

Tests must cover:
- model registry and dimensions,
- model loading/readiness,
- API request and response contract,
- authentication and client separation,
- BGE query prefix and document formatting,
- normalization behaviour,
- batch ordering,
- invalid/oversized input,
- queue capacity and overload responses,
- scheduling fairness,
- health/readiness,
- real inference smoke tests for both pinned models.

Use the actual downloaded models for a small deterministic smoke test. Use fakes/mocks for scheduler, validation and API unit tests where appropriate. No tests may call external paid APIs.

A separate opt-in load-test script is provided for later manual use. Do not run a long load test during every Docker build.

## 8. Build and deploy sequence

1. Cline inspects the already-created `embedding-inference-service` project directory and its existing Git configuration.
2. Cline preserves the existing `.git`, `venv`, GitHub remote, branch and initial commit.
3. Cline builds the complete embedding service in this clean project directory; it does not copy RAG application source code into it.
3. Cline creates all configuration, source, test and documentation files.
4. Cline creates a Dockerfile whose build stage installs dependencies, downloads pinned models, and runs the complete test suite.
5. Cline creates the Hugging Face Docker README metadata (`sdk: docker`, `app_port: 7860`).
6. Cline creates GitHub Actions workflow to sync GitHub `main` to the existing Space.
7. Cline reviews the complete diff and checks for secrets, accidental local model artifacts and unintended changes to existing Git configuration.
8. User adds any missing secrets, commits and pushes the completed project to the already-created GitHub repository.
9. Hugging Face builds the image. Tests run during that build. Failed tests fail deployment.
10. User checks Space build logs and health endpoints.

## 9. Completion criteria

The service is complete only when:
- Both pinned models load and produce the expected dimensions.
- All automated tests pass during the HF build.
- API authentication and overload protection work.
- The application starts on port 7860 as UID 1000.
- The workflow deploys to the existing Space without renaming it.
- The original RAG application has not been modified.
