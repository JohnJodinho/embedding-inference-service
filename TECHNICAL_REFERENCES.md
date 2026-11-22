# Technical References and Documentation Gate

Cline must consult the current official documentation before implementation. Do not rely on old snippets from memory or third-party tutorials when an official source exists.

| Area | Official reference | Required use |
|---|---|---|
| uv projects and lockfiles | https://docs.astral.sh/uv/concepts/projects/layout/ | Define direct runtime/test dependencies and commit the resolved lockfile |
| uv project sync | https://docs.astral.sh/uv/concepts/projects/sync/ | Install the frozen runtime environment and export isolated test dependencies |
| uv dependency sources | https://docs.astral.sh/uv/concepts/indexes/ | Pin PyTorch to its explicit CPU wheel index |
| Hugging Face Docker Spaces | https://huggingface.co/docs/hub/spaces-sdks-docker | Space YAML metadata, port 7860, Docker build and runtime secrets |
| Hugging Face Space configuration | https://huggingface.co/docs/hub/spaces-config-reference | Validate supported Space configuration fields |
| FastAPI lifespan | https://fastapi.tiangolo.com/advanced/events/ | Use lifespan context; avoid deprecated startup/shutdown event handlers |
| FastAPI async testing | https://fastapi.tiangolo.com/advanced/async-tests/ | Correct async test client and lifespan handling |
| FastAPI testing events | https://fastapi.tiangolo.com/advanced/testing-events/ | Ensure tests explicitly execute application lifespan |
| Sentence Transformers model API | https://www.sbert.net/docs/package_reference/sentence_transformer/model.html | Current encode API, batching, normalization, device and token handling |
| Hugging Face tokenizer API | https://huggingface.co/docs/transformers/en/main_classes/tokenizer | Count tokens with special tokens and `truncation=False` |
| Sentence Transformers migration guide | https://sbert.net/docs/migration_guide.html | Avoid outdated API patterns |
| MiniLM model card | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Verify model ID, dimensions and model behaviour |
| BGE model card | https://huggingface.co/BAAI/bge-base-en-v1.5 | Verify model ID, query instruction, dimensions, sequence length and licence |
| GitHub Actions workflow syntax | https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions | Current workflow syntax and least-privilege permissions |
| actions/checkout | https://github.com/actions/checkout | Current supported checkout action and inputs |

## Required verification record

During implementation, Cline must add a short section to this file recording:
- Date checked.
- Exact Python and package versions pinned.
- Exact immutable commit hash for each model.
- Model licence verified.
- Exact Hugging Face Space hardware shown in Space settings, if visible to the agent.
- Any documentation/API uncertainty that remains unresolved.

Do not claim the Space entitlement or hardware is guaranteed by a rename. The instruction is to retain the existing Space and not rename it.

## Required verification record

### Date checked
2026-11-22

### Pinned runtime Python and package versions (Python 3.11)
| Package | Pinned version | Source |
|---------|---------------|--------|
| fastapi | 0.165.0 | https://pypi.org/pypi/fastapi/json |
| uvicorn | 0.54.0 | https://pypi.org/pypi/uvicorn/json |
| sentence-transformers | 6.3.0 | https://pypi.org/pypi/sentence-transformers/json |
| torch | 2.15.0 CPU build | https://download.pytorch.org/whl/cpu/torch/ |

Direct dependencies are declared in `pyproject.toml`; `uv.lock` records the
complete resolved dependency graph and artifact hashes for the Python 3.11
deployment target. The Docker build uses the frozen lock and exports only the
test dependency group into a separate target directory, keeping test tooling
out of the runtime image.

The lock generator is pinned in the Docker build:

| Tool | Pinned version | Source |
|------|---------------|--------|
| uv | 0.13.9 | https://pypi.org/pypi/uv/json |

Test-only direct dependencies:

| Package | Pinned version | Source |
|---------|---------------|--------|
| pytest | 9.2.0 | https://pypi.org/pypi/pytest/json |
| httpx | 0.28.1 | https://pypi.org/pypi/httpx/json |

The GitHub workflow uses `actions/checkout@v7.0.1`, verified against the
official releases API on 2026-11-21. The workflow's `contents: read`
permission is sufficient because deployment authenticates directly to the HF
Space with the repository's `HF_TOKEN` secret.

`uvx` was unavailable, so the FastAPI AI-agent skill could not be installed via
`uvx library-skills`; the official FastAPI documentation was used directly.
Pinned uv 0.13.9 was installed in the ignored project environment only to
generate and verify the lockfile. No ML packages or model weights were
installed locally. A Linux-targeted uv dry run selected `torch==2.15.0+cpu`
without CUDA or Triton packages.

### Immutable model commit hashes (resolved from official HF model APIs)
| Alias | Model ID | Revision (commit hash) | Dimensions | Licence |
|-------|----------|------------------------|------------|---------|
| `minilm-384-v1` | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | 384 | Apache-2.0 |
| `bge-base-768-v1` | `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | 768 | MIT |

Hashes resolved via `https://huggingface.co/api/models/<model-id>` (field `sha`).
Dimensions confirmed from `config.json` (`hidden_size`): MiniLM 384, BGE 768.
Max sequence length: 512 tokens for both models.

### BGE query instruction (verified from BGE model card)
`Represent this sentence for searching relevant passages: ` (trailing space
significant). Prepended to query inputs only; document inputs receive no prefix.

### Official documentation consulted
- Hugging Face Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
- Hugging Face Space configuration: https://huggingface.co/docs/hub/spaces-config-reference
- FastAPI lifespan: https://fastapi.tiangolo.com/advanced/events/
- FastAPI testing: https://fastapi.tiangolo.com/advanced/async-tests/
- FastAPI response models: https://fastapi.tiangolo.com/tutorial/response-model/
- uv project layout and lockfiles: https://docs.astral.sh/uv/concepts/projects/layout/
- uv project sync and dependency groups: https://docs.astral.sh/uv/concepts/projects/sync/
- uv package indexes: https://docs.astral.sh/uv/concepts/indexes/
- PyTorch CPU installation selector: https://pytorch.org/get-started/locally/
- Sentence Transformers model API: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
- Hugging Face tokenizer API: https://huggingface.co/docs/transformers/en/main_classes/tokenizer
- Sentence Transformers migration guide: https://sbert.net/docs/migration_guide.html
- MiniLM model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- BGE model card: https://huggingface.co/BAAI/bge-base-en-v1.5
- GitHub Actions workflow syntax: https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions
- actions/checkout: https://github.com/actions/checkout

### Unresolved items
- Hugging Face Space hardware is not visible from this repository; verify it in
  the Space settings before tuning CPU thread or queue limits.
- No model IDs, revisions, dimensions, licences, or API parameter uncertainties
  remain.
- Full local tests, ML dependency installation and model downloads remain
  intentionally prohibited by the project rules; the HF Docker build is the
  authoritative verification environment.

