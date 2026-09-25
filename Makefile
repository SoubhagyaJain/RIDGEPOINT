.PHONY: setup test lint gpu-check bench-smoke memory-budget

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
