# Cline One-Shot Build Instructions

## Objective

Turn the currently open copied RAG-application workspace into a dedicated, production-minded embedding API. This workspace is a copy; do not modify the original RAG application directory.

The finished service will run in the existing Hugging Face Docker Space:

`JohnAlbarkaIbrahim/SentimentScope`

Do not rename or transfer that Space. Preserve its current identity and existing free-tier entitlement.

The GitHub repository already exists and has already received an initial push:

`JohnJodinho/embedding-inference-service`

The user's active project directory is:

`C:\\Users\\user\\Desktop\\python_projects\\AI_Engineering_And_Automation\\embedding-inference-service`

Git has already been initialised, a `venv` virtual environment has been created and activated, and the initial `.gitignore`, `Dockerfile`, `.dockerignore` and empty `.env` have been committed and pushed.

GitHub `main` is the only source of truth. On every push to `main`, GitHub Actions synchronises the repository to the existing Hugging Face Space. Hugging Face builds the Docker image and runs the complete automated test suite during the build. A failed test must fail the image build and prevent deployment.

## User constraints

- The user will not write or edit code. Cline must implement everything.
- The user can run terminal commands and provide secrets through GitHub/Hugging Face settings.
- Do not run local tests, install large ML dependencies, download model weights, or perform local model inference.
- Do not download model weights to the user's PC. Model downloads and real-model tests happen only during the Hugging Face Docker build.
- Do not create a preliminary scaffold and stop. Inspect the workspace, make the configuration decisions, implement the complete service, tests, documentation and deployment workflow in one pass.
- Do not modify the original RAG application. Work only in the currently open copied workspace.
- Keep the implementation straightforward. Avoid unnecessary services, databases, Redis, Celery, Kafka, distributed queues, multiple Uvicorn workers, or complex abstractions.
- Do not use deprecated APIs, guessed APIs, or unpinned dependencies.
- Do not claim tests passed locally. They are deliberately not run locally.

## Required model registry

Implement exactly these two public model aliases:

| Alias | Model ID | Dimensions |
|---|---|---:|
| `minilm-384-v1` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `bge-base-768-v1` | `BAAI/bge-base-en-v1.5` | 768 |

Pin both models to explicit immutable Hugging Face commit revisions. Resolve the current commit hashes from the official Hugging Face model repositories during implementation and record them in configuration. Do not use `main` as a production revision.

Use CPU inference only. Load each model once at application startup. Verify output dimensions at startup and fail readiness if a model is unavailable or produces the wrong dimension.

BGE behaviour:
- `input_type` must be either `query` or `document`.
- For queries, prepend exactly: `Represent this sentence for searching relevant passages: `
- For documents, do not prepend that instruction.
- Normalize embeddings when requested.
- Do not silently truncate inputs beyond the configured token limit.

MiniLM behaviour:
- Do not add a query instruction or alter input text.
- Accept an explicit `normalize_embeddings` boolean.
- Preserve the current RAG application's per-call normalization behaviour. Its old call sites differ: Supabase methodology search used SentenceTransformer's default (false); tool embeddings and tool queries used true; the RAG route used true.

## Service design

Implement:
- FastAPI application, listening on `0.0.0.0:7860`.
- One Uvicorn worker only.
- FastAPI lifespan for model loading and clean worker startup/shutdown. Do not use deprecated `@app.on_event` startup/shutdown handlers.
- One central CPU inference worker so the two models do not run uncontrolled concurrent CPU inference.
- Separate bounded request queues for BGE and MiniLM, with BGE receiving higher scheduling priority but MiniLM never starving.
- Batch requests and a short configurable micro-batching window. Keep this scheduler small, documented and thoroughly tested. Do not add a distributed queue.
- Return 429 for per-client rate-limit violations, 503 when a bounded queue is full or the service is not ready, 400 for invalid input, and 401 for invalid/missing API keys.
- Do not log API keys, authorization headers, or raw user text.
- Separate API credentials for the RAG application and ZIT assistant, supplied as Hugging Face Space Secrets.
- Validate request count, body size, per-text character count and token count.
- Validate returned vector count, vector dimension and finite numeric values.
- Provide structured logs with request ID, model alias, client label, batch size, queue wait, inference duration, total duration, status and error code. Never log embeddings or raw input text.
- Provide `/health/live`, `/health/ready`, `/v1/models`, `/v1/embeddings`, and `/metrics`.
- Health endpoints must not reveal secrets or internal configuration.
- Keep metrics lightweight and process-local; one worker is intentional.
- Add graceful shutdown and bounded request timeouts.

## API contract

Use a simple OpenAI-style embeddings contract, but document the exact supported subset.

Request:
```json
{
  "model": "bge-base-768-v1",
  "input": ["What courses does ZIT offer?"],
  "input_type": "query",
  "normalize_embeddings": true
}
```

`input` accepts either one string or a list of strings. Preserve input ordering in the response.

Response:
```json
{
  "object": "list",
  "model": "bge-base-768-v1",
  "dimensions": 768,
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.01, -0.02]
    }
  ],
  "usage": {
    "input_texts": 1
  }
}
```

The displayed vector above is illustrative; actual responses must contain the full vector.

For MiniLM, `input_type` may be omitted or set to `text`; reject unsupported values. For BGE, require `query` or `document`.

## Deployment and tests

Use a Docker Space README YAML block with:
- `sdk: docker`
- `app_port: 7860`

Use a Dockerfile with a build/test stage and a runtime stage:
1. Install pinned dependencies in the build stage.
2. Download both pinned model revisions during the Hugging Face build, not locally.
3. Run real-model smoke checks and the entire pytest suite in the Hugging Face build.
4. If any test fails, fail the Docker build.
5. Copy the tested application, installed runtime dependencies and model cache into the final runtime image.
6. Run as non-root UID 1000.
7. Start one Uvicorn worker in the foreground on port 7860.

Place test files under `tests/`. The Docker build must run the complete suite, not a selected subset. Tests must cover API validation, authentication, model registry, dimensions, BGE query/document formatting, normalization, batch ordering, queue limits, scheduling/fairness, errors, health/readiness and real model smoke inference.

Do not run load tests that consume excessive build time on every deployment. Include a small deterministic concurrency/queue smoke test in the regular suite. Provide a separate opt-in load-test script for manual use later.

The model cache and application must be downloaded/built on Hugging Face, not on the user's machine. Arrange Dockerfile layers so dependency/model layers can be cached when only application source changes.

## Documentation and current-code requirements

Before implementation:
1. Read `IMPLEMENTATION_PLAN.md`, `ARCHITECTURE.md` and `DEPLOYMENT_RUNBOOK.md`.
2. Read the project `.clinerules`.
3. Inspect the current workspace and its existing Git status, remote, branch, `.gitignore`, `Dockerfile`, `.dockerignore` and `.env`.
4. This is already a new, clean project directory, not the copied RAG application's working directory. Do not delete, reinitialise, or replace `.git`; do not recreate or delete `venv`; do not overwrite the existing Git remote or discard the user's initial commit.
5. Do not copy the RAG application's source code into this project. The RAG call-site changes will be made separately in the original RAG project later.
5. Check current official documentation for FastAPI, Hugging Face Docker Spaces, Sentence Transformers, PyTorch and GitHub Actions. Use current stable APIs and cite/record the exact documentation URLs in `TECHNICAL_REFERENCES.md`.
6. Use the official FastAPI AI-agent skill if available. FastAPI's current documentation points AI coding agents to `uvx library-skills`; install/use the FastAPI skill in the project if supported by the environment. If unavailable, use the official documentation directly and record that limitation.
7. Verify current model IDs, model cards, licences, model revisions, output dimensions and BGE query instruction from the official Hugging Face model pages before pinning.

Do not invent a skill, package, API, version, model revision, or Hugging Face setting. If a required fact cannot be verified, stop and report the exact unresolved item instead of guessing.

## Git and delivery

Prepare the project for the new repository `JohnJodinho/embedding-inference-service`.
- Remove the copied repository's old `.git` metadata from this copy only.
- Do not remove or alter the original RAG repository.
- Create `.gitignore`, `.dockerignore`, `.env.example`, pinned dependency files, Dockerfile, Hugging Face README, tests, docs and GitHub Actions workflow.
- GitHub Actions must trigger on push to `main` and manual dispatch.
- Use `actions/checkout` pinned to a current stable major version and least-privilege workflow permissions.
- Sync the checked-out GitHub source to `https://huggingface.co/spaces/JohnAlbarkaIbrahim/SentimentScope`.
- Do not use `git push -f` unless the workflow explicitly documents why the HF Space is a deployment mirror and safely handles the initial unrelated history. Prefer a normal push after the HF repository has been aligned once.
- Do not put tokens in URLs in logs. Use a masked credential mechanism.
- Keep `HF_TOKEN` in GitHub Actions secrets. It must never appear in repository files.
- The workflow should not run local ML tests. The Hugging Face Docker build runs them.

Cline must not push anything or invent credentials. Complete implementation and inspect the final diff first. Then provide the user with only the commands needed to:
1. verify the already-configured GitHub remote and branch,
2. review, commit and push the completed implementation to the existing `main` branch,
3. add `HF_TOKEN` as a GitHub Actions secret if not already configured,
4. add the two client API keys as Hugging Face Space Secrets,
5. trigger/monitor the first Space build.

## Final acceptance checklist

Do not declare completion unless:
- All requested files exist.
- No old RAG routes, database integrations, prompts, chat logic or unrelated assets remain in the embedding-service copy.
- Both model revisions are immutable and documented.
- Both models load once and pass real inference smoke checks during the HF Docker build.
- All tests run during every HF image build.
- One Uvicorn worker is configured.
- The scheduler is bounded and tested.
- Authentication is enforced on embedding and metrics endpoints.
- Health endpoints expose no sensitive information.
- Structured logs are useful but contain no secrets, raw text or vectors.
- The deployment workflow targets the existing Space and does not rename it.
- No local model downloads or local tests were performed.
- The final response clearly distinguishes implemented files from deployment steps the user must perform.
