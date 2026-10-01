"""Run V1: uvicorn ridgepoint.server.v1:app --port 8002."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ridgepoint.runtime.backend import V1Backend
from ridgepoint.server.app import create_app


def make_app():
    config_path = Path(os.environ.get("RIDGEPOINT_V1_CONFIG", "configs/engine/v1.yaml"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return create_app(V1Backend(max_batch_size=config["max_batch_size"],
                                batch_window_ms=config["batch_window_ms"],
                                kv_cap_mib=config["kv_cap_mib"],
                                free_reserve_mib=config["free_reserve_mib"]))


app = make_app()
