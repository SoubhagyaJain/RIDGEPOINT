"""Print one real streamed completion to verify the V0 HTTP path."""

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--prompt", default="Explain a GPU cache in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    body = {"prompt": args.prompt, "max_tokens": args.max_tokens,
            "temperature": 0, "ignore_eos": True, "stream": True}
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", args.url, json=body) as response:
            response.raise_for_status()
            print(f"request_id={response.headers['x-request-id']}")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "token":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "done":
                    print(f"\nusage={event['usage']} timings_ms={event['timings_ms']}")
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])


if __name__ == "__main__":
    main()
