# Phase 06A server execution notes

- 2026-08-25T06:49:06Z: Preflight passed on NVIDIA GeForce RTX 3090 with the pinned model revision loaded offline in BF16.
- The supplied workspace has no Git metadata. Existing files are preserved and source SHA-256 hashes are used for provenance.
- The provided Python environment lacked `nbformat`, `nbclient`, and `nbconvert`; these notebook-runtime dependencies were installed into `/root/venvs/roadbuddy-rtx3090-py310`. No scientific parameter was changed.
- No implementation fix has been made at this point.

- 2026-08-25: Stage 00 smoke completed, then the full temporary copy changed only `RUN_SCOPE`. Full data audit passed all hard gates, including 549/549 video decodes and zero exact-duplicate cross-split sets.
- 2026-08-25: Stage 01 is legitimately awaiting two independent annotations; blind artifacts were generated. Stage 02 completed a 951/241-row, 351/88-group inner split with zero group overlap.
- 2026-08-25: Stage 03 full completed all three 298-row arms. Z-F1 retained one parse failure as incorrect with raw response; Z-F3/Z-F8 had no parse failures.
- 2026-08-25: Stage 04 full completed. Stage A early-stopped at step 32 and locked step 8 from inner-dev only; Stage B reset and ran 8 steps on all 1,192 training rows. All 192 LoRA tensors changed and adapter reload/hash passed. PyTorch peak was 17.83 GiB. The log contained recoverable allocator-pressure OOM warnings without a raised CUDA exception; execution exited 0 and all integrity gates passed.
- 2026-08-25: Stage 05 full attempt 1 failed before inference because the server-side temporary-copy regex removed same-line config assignments after `RUN_SCOPE`. Source notebook/artifacts were unchanged. The copy generator was corrected to replace only the exact assignment; retry count set to 1.
- 2026-08-25: Stage 05 retry 1 completed all three full r16 evaluation arms with 298 unique frozen IDs and no parse failures.
- 2026-08-25: Stage 06 full completed r8/r16/r32/r64 with 298 IDs each, alpha/r=2, parse 100%, and peak PyTorch VRAM 17.80–18.09 GiB. Recoverable allocator-pressure warnings occurred without CUDA exception; notebook exited 0 and all integrity gates passed.
- Packaging retry 1: the first archive verification found that `execution_state.json` had been mutated after its manifest hash was computed. State/notes were frozen first, then manifest/filelist/archive were regenerated; scientific artifacts were unchanged.
