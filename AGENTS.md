# Ridgepoint agent instructions

- Follow `docs/spec.md`, `docs/plan.md`, and `docs/status.md` before changing scope.
- Complete one phase at a time. Stop after its code, verification, teaching notes, and commit; wait for the user to say "continue".
- Never present an estimate as a measurement. Keep raw runs and model weights out of Git.
- Use the same workload trace, SLOs, and metric definitions for every backend comparison.
- Preserve allocator invariants: every block is in exactly one state, and refcounts never become negative.
- Update `docs/status.md` and `docs/learning/phase-N.md` as work progresses.
