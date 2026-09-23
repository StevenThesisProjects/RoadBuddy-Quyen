# Phase 06B server implementation notes

Updated: `2026-09-12T14:52:05.132937+00:00`

## Preflight and preservation

- Baseline gate passed for locked `L32-F1`, 298 IDs / 110 groups, canonical validation checksum and winner prediction hash.
- Python 3.10.12, CUDA/BF16 and RTX 3090 are available; the pinned Vintern revision exists in the local Hugging Face cache; disk has 190.52 GiB free.
- This uploaded repository package contains no `.git` directory. `git status` and `git diff --check` therefore cannot audit the working tree. No reset, checkout, deletion of user artifacts, public-test access, or submission action was performed. Manual trailing-whitespace scan passed.
- The host sandbox cannot create a user namespace, so built-in `apply_patch` was unavailable. Small unified diffs were attempted first; later anchored edits asserted unique anchors and were followed by compile/tests.

## Protocol hardening implemented

- Taxonomy manifest schema v2 with human/input hashes, canonical membership hash, agreement metrics, annotator/adjudicator provenance and timestamp; legacy schema is explicitly rejected.
- Cohen kappa, raw agreement and prevalence for all axes/primary label. Kappa below 0.70 yields `awaiting_reannotation` unless signed exception provenance is valid.
- Canonical labels and novelty gates: 30 rows, 15 groups, 0.30 error share, 0.10 share gap; recommendation remains non-binding evidence.
- Exact 16-arm temporal registry roles, k/candidate/tile budgets, hashes, membership and eligibility; Uniform, Random and Oracle cannot contaminate winner candidates.
- Train-only temporal support, traffic extractor, feature-bank and full prediction validators; validation banks reject relevance targets.

## Notebook work

- Updated Phase 06A.01, Phase 06B.00, Phase 06B.03 preflight and Phase 06B.04.
- Added Phase 06B.03A–03E for train annotation audit, deterministic frozen-encoder feature-bank build, two-stage selector training, isolated grounding diagnostics and full temporal VQA integrity.
- Executed Phase 06A.01, Phase 06B.00 and Phase 06B.03A sequentially. All stopped at expected gates and wrote status/templates; no labels were synthesized.

## Verification

- 18 Phase 06A/06B notebooks: valid JSON and all code cells parse.
- All Python source compiles; unit tests: 19/19 pass.
- Public-test/submission creator scan and validation checkpoint/support leakage scan: no hits.
- Fixed allocations: k=1 `[8]`, k=3 `[3,2,3]`, k=8 eight `[1]` values.

## Current gates

Taxonomy annotation and train temporal support are missing. Phase 06B is not complete. Selector training, frozen-validation inference, grounding diagnostics and statistics were not run; no smoke output is treated as scientific evidence.
