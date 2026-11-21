"""Optional load-test script for manual use.

NOT run during the Hugging Face Docker build. Run locally or from CI only when
you explicitly want to measure throughput/latency against a deployed Space.

Example:
    python scripts/load_test.py --url https://JohnAlbarkaIbrahim-SentimentScope.hf.space \\
        --key $RAG_API_KEY --model minilm-384-v1 --texts 50 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import time
from statistics import mean

import httpx


async def _worker(client, url, key, model, texts, results, semaphore):
    async with semaphore:
        headers = {"Authorization": f"Bearer {key}"}
        payload = {"model": model, "input": texts, "normalize_embeddings": True}
        start = time.monotonic()
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=60)
            elapsed = time.monotonic() - start
            results.append((resp.status_code, elapsed, len(texts)))
        except Exception as exc:  # noqa: BLE001
            results.append((0, time.monotonic() - start, 0))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in embedding load test")
    parser.add_argument("--url", required=True, help="Full /v1/embeddings URL")
    parser.add_argument("--key", required=True, help="Client API key")
    parser.add_argument("--model", default="minilm-384-v1")
    parser.add_argument("--texts", type=int, default=10, help="Total texts to send")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[tuple[int, float, int]] = []
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        per = max(1, args.texts // args.concurrency)
        for i in range(args.concurrency):
            count = per if i < args.concurrency - 1 else args.texts - per * (args.concurrency - 1)
            batch = [f"load test text {j}" for j in range(count)]
            tasks.append(_worker(client, args.url, args.key, args.model, batch, results, semaphore))
        await asyncio.gather(*tasks)

    ok = [r for r in results if r[0] == 200]
    print(f"total requests: {len(results)}")
    print(f"ok: {len(ok)}")
    if ok:
        print(f"avg latency: {mean(r[1] for r in ok):.3f}s")
        print(f"p95 latency: {sorted(r[1] for r in ok)[max(0, int(len(ok)*0.95)-1)]:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
