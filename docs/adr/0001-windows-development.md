# ADR 0001: Windows development, native Linux final measurements

The available host is Windows with an RTX 4050 Laptop GPU in WDDM mode and an Ubuntu WSL distribution. Phase 1 development and exploratory E1 run here. The notebook calls for native Linux for final GPU benchmarks because WDDM/WSL scheduling and profiling differ. We will preserve hardware manifests and repeat headline measurements on native Linux if one becomes available; until then, label Windows results exploratory. The repository stays cross-platform through Python entry points and a portable Makefile for Unix hosts.
