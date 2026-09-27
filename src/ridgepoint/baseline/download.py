"""Download pinned model files outside Git, including on Windows without symlinks."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ridgepoint.baseline.hf_backend import DEFAULT_MODEL_PATH, MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    patterns = ["*.json", "merges.txt", "vocab.json"]
    if not args.tokenizer_only:
        patterns.append("*.safetensors")
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=patterns,
                             local_dir=args.local_dir, max_workers=1)
    print(f"Pinned revision: {MODEL_REVISION}\nLocal path: {path}")
    print("The default path is discovered automatically. For a custom path, set RIDGEPOINT_MODEL_PATH.")


if __name__ == "__main__":
    main()
