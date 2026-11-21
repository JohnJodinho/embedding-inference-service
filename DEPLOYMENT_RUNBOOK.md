# Deployment Runbook

## Fixed targets

- GitHub: `JohnJodinho/embedding-inference-service`
- Hugging Face Space: `JohnAlbarkaIbrahim/SentimentScope`
- Branch: `main`
- Hugging Face port: `7860`
- Do not rename or transfer the existing Space.

## One-time user actions

1. Create an empty GitHub repository named `embedding-inference-service` under `JohnJodinho`.
2. In the new GitHub repository, add an Actions secret named `HF_TOKEN`. Use a Hugging Face token with write access to the existing Space.
3. In Hugging Face Space Settings → Variables and secrets, add:
   - `RAG_API_KEY`: long random secret for the Render RAG application.
   - `ZIT_API_KEY`: a different long random secret for the ZIT assistant.
4. Do not put these values in `.env.example`, Git, workflow YAML or logs.

## Commit and push

The GitHub repository and local `main` branch already exist. Preserve the current
Git metadata and remote; do not run `git init` or replace the remote. Review the
diff, commit the completed implementation on `main`, and push to `origin`.

```powershell
git status --short --branch
git add .
git commit -m "Align service with implementation plan"
git push origin main
```

Never delete or overwrite the original RAG application's repository.

## Deployment flow

Every push to GitHub `main` triggers the workflow:
1. Check out the GitHub source.
2. Prepare the Hugging Face Space README metadata.
3. Force-replace HF `main` with GitHub `main`; the Space is an intentional
   deployment mirror, not a branch to merge into.
4. Hugging Face starts a Docker build.
5. The Docker build installs pinned runtime dependencies, downloads both pinned
   models, and runs the complete test suite with real-model smoke checks. No test
   is skipped.
6. If any test fails, the Docker build fails and the new image is not deployed.
7. If the build succeeds, the Space starts the API.

The Hugging Face build log is the authoritative record of build-time tests. GitHub Actions success means the source sync succeeded; it does not by itself prove the Space image became healthy.

## First deployment verification

Check:
- GitHub Actions workflow completed successfully.
- Hugging Face Space build completed successfully.
- Build log shows both model smoke checks and the full test suite passing.
- Space runtime log shows both models loaded and the inference worker ready.
- `GET /health/live` returns HTTP 200.
- `GET /health/ready` returns HTTP 200 only after both models are loaded.
- Authenticated `GET /v1/models` lists both aliases and expected dimensions.
- An authenticated test request returns 384 dimensions for MiniLM and 768 for BGE.

Never paste API keys into a command that will be saved in shell history or into a public issue.

## Updating the service

Make changes in the local Git working tree, review them, commit and push to `main`. Do not edit code directly in the Hugging Face Space. GitHub remains the source of truth.

## Failure diagnosis

- **GitHub Action failed:** inspect the Actions log; check `HF_TOKEN`, repository target and Git push permissions.
- **HF build failed before tests:** inspect dependency installation, model download, model revision and Docker build logs.
- **Tests failed:** use the first failing test and its traceback; do not bypass the test command.
- **Space starts but readiness is 503:** inspect model loading and worker startup logs.
- **Embedding endpoint returns 401:** verify the correct consumer key is configured in the caller and Space secret.
- **Embedding endpoint returns 429:** client rate limit reached; wait and retry according to `Retry-After`.
- **Embedding endpoint returns 503:** queue full, model not ready or worker unavailable; inspect queue and worker logs.
- **Slow requests:** inspect queue wait separately from inference time. Tune batch size and CPU thread settings based on measured Space performance.

## Recovery

If a deployment breaks, revert the offending commit in GitHub and push the revert. The workflow synchronises the reverted source and Hugging Face rebuilds it. Do not make an untracked hotfix directly in the Space.
