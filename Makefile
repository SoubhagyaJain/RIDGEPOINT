.PHONY: setup test lint gpu-check bench-smoke memory-budget download-model serve-v0 serve-null trace-smoke replay-smoke analyze-smoke

setup:
	uv sync --frozen --group dev

test:
	uv run --frozen pytest -q

lint:
	uv run --frozen ruff check src bench tests

gpu-check:
	uv run --frozen python -m ridgepoint.hardware

bench-smoke:
	uv run --frozen python -m bench.micro.run --smoke

memory-budget:
	uv run --frozen python -m ridgepoint.memory_budget configs/models/qwen2.5-1.5b-instruct.config.json

download-model:
	uv run --frozen python -m ridgepoint.baseline.download

serve-v0:
	uv run --frozen uvicorn ridgepoint.server.app:app --host 127.0.0.1 --port 8000

serve-null:
	uv run --frozen uvicorn bench.backends.null:app --host 127.0.0.1 --port 8001

trace-smoke:
	uv run --frozen python -m bench.workloads.generate --workload configs/workloads/chat.yaml --output results/raw/phase2-chat-smoke.jsonl --duration-s 20 --rate-rps 0.5 --warmup-s 0 --max-requests 8 --prompt-tokens 64 --output-tokens 16

replay-smoke:
	uv run --frozen python -m bench.client.replay results/raw/phase2-chat-smoke.jsonl --output results/raw/v0-chat-smoke.jsonl

analyze-smoke:
	uv run --frozen python -m bench.analyze.metrics results/raw/v0-chat-smoke.jsonl --duration-s 20 --warmup-s 0
