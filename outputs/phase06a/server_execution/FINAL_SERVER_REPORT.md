# RoadBuddy Phase 06A — Final server report

Generated: 2026-08-25T15:44:05Z

## Outcome

- Status: **complete**
- Baseline winner: **L32-F1**
- Primary accuracy: **0.530201** (95% group-cluster CI **0.464830–0.593333**)
- Secondary macro-F1: **0.514057** (95% group-cluster CI **0.441516–0.579008**)
- Parse rate: **1.000000** (95% CI **1.000000–1.000000**)
- Validation contract: 298 frozen samples / 110 groups; ID-file SHA-256 `dbfa2d337f56bd681df70206fa8cee83d743c6743a3493619083d03b804145c5`.
- Exact McNemar vs Z-F1: raw p=0.011598; Holm-adjusted p=0.092786. McNemar is secondary; clustered bootstrap is primary.
- Non-inferiority margin: 0.02. Only L32-F1 passed the preregistered non-inferiority gate relative to empirical best.

## Environment and provenance

- Python 3.10.12 at `/root/venvs/roadbuddy-rtx3090-py310/bin/python`.
- PyTorch 2.13.0+cu126, CUDA 12.6, BF16 supported.
- GPU: NVIDIA GeForce RTX 3090 (24576 MiB).
- Model: `5CD-AI/Vintern-1B-v3_5` revision `b98f263eab246eb5269ade64edbdca8a887dc44d`, native template `Hermes-2`.
- Workspace has no Git metadata; `git diff --check` was attempted and is unavailable. Source and all eight source-notebook SHA-256 values remained unchanged from preflight.
- Notebook runtime dependencies `nbformat`, `nbclient`, and `nbconvert` were installed into the existing RoadBuddy venv; no scientific setting changed.

## Stage status

| Stage | Status | Verification |
|---|---|---|
| 00 | smoke_complete / PASS | Smoke then full; 549/549 videos decoded, zero exact duplicates/leakage. |
| 01 | awaiting_annotation | Blind sheet/guideline created; no taxonomy labels fabricated. |
| 02 | complete | 951-row train_fit / 241-row inner_dev; 351/88 groups; zero overlap. |
| 03 | smoke_complete / complete | Z-F1/Z-F3/Z-F8, 298 rows each in full scope. |
| 04 | smoke_complete / complete | r16 locked step 8 via inner-dev; reset retrain on all 1,192 training rows. |
| 05 | smoke_complete / complete | L16-F1/F3/F8, 298 rows each; one pre-inference retry for temp-copy bug. |
| 06 | smoke_complete / complete | r8/r16/r32/r64, 298 rows each; alpha/r=2. |
| 07 | complete | 9-arm clustered bootstrap/statistics and winner lock. |

## Full experiment matrix

| Experiment | Accuracy | Accuracy 95% CI | Macro-F1 | Macro-F1 95% CI | Parse rate |
|---|---:|---:|---:|---:|---:|
| L32-F1 | 0.530201 | 0.464830–0.593333 | 0.514057 | 0.441516–0.579008 | 1.000000 |
| L64-F1 | 0.500000 | 0.435252–0.563333 | 0.496023 | 0.427341–0.558437 | 1.000000 |
| Z-F8 | 0.486577 | 0.430034–0.542662 | 0.448848 | 0.377759–0.513256 | 1.000000 |
| L16-F8 | 0.483221 | 0.420661–0.546392 | 0.467457 | 0.393876–0.532509 | 1.000000 |
| Z-F3 | 0.483221 | 0.426117–0.539571 | 0.444526 | 0.377480–0.505244 | 1.000000 |
| L16-F3 | 0.473154 | 0.405694–0.540543 | 0.447494 | 0.370881–0.516909 | 1.000000 |
| L16-F1 | 0.463087 | 0.399350–0.527586 | 0.438381 | 0.367019–0.506965 | 1.000000 |
| L8-F1 | 0.456376 | 0.388689–0.523490 | 0.427714 | 0.350651–0.499614 | 1.000000 |
| Z-F1 | 0.439597 | 0.379658–0.503268 | 0.394899 | 0.324149–0.464187 | 0.996644 |

## Rank training/resource summary

| Rank | Alpha | Locked step | Trainable params | Peak VRAM GiB | Train seconds | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 16 | 1081344 | 17.804 | 319.7 | 0.456376 | 0.427714 |
| 16 | 32 | 8 | 2162688 | 17.826 | 158.7 | 0.463087 | 0.438381 |
| 32 | 64 | 64 | 4325376 | 18.003 | 1329.7 | 0.530201 | 0.514057 |
| 64 | 128 | 80 | 8650752 | 18.089 | 1678.4 | 0.500000 | 0.496023 |

## Integrity and warnings

- All nine full prediction artifacts contain exactly 298 unique frozen IDs with identical membership and retained raw responses.
- Z-F1 has one parse failure; it retained its raw response and was counted incorrect. The other eight arms have no parse failures.
- Full data audit decoded 549/549 unique videos and found zero sample/group overlap and zero exact-duplicate cross-split sets. Near-duplicate audit remains a documented manual follow-up (`not_implemented_manual_followup`).
- Stage 04/06 logs contain CUDA caching-allocator allocation-failure warnings under high pressure, but no raised CUDA exception/Traceback; notebooks exited 0 and all training/reload/artifact gates passed. Peak PyTorch VRAM was recorded per run.
- Stage 05 full attempt 1 failed before inference because the temporary execution-copy regex removed same-line config assignments. The source notebook was unchanged; retry 1 replaced only the exact `RUN_SCOPE` assignment and completed.
- Taxonomy status is `awaiting_annotation`; two independent human annotation files are missing, and no labels were fabricated.
- Final unit tests: 8 passed, 0 failed, 0 skipped.
- Repository-wide scan found no Phase 06A public-test prediction/submission workflow or output. A legacy Phase 01 notebook still contains a public-test sample-submission path/reference; it was not executed by Phase 06A.
- No public-test predictions or submission were created.

## Primary artifacts to download

- `outputs/phase06a/final_analysis/final_leaderboard.csv`
- `outputs/phase06a/final_analysis/baseline_winner_manifest.json`
- `outputs/phase06a/final_analysis/PHASE06A_REPRODUCIBILITY_REPORT.md`
- `outputs/phase06a/final_analysis/paired_statistical_tests.csv`
- `outputs/phase06a/final_analysis/noninferiority_analysis.csv`
- `outputs/phase06a/final_analysis/per_class_metrics.csv`
- `outputs/phase06a/server_execution/export_manifest.json`
- `outputs/phase06a/roadbuddy_phase06a_results.tar.gz`
- Final adapter paths and hashes are listed in `export_manifest.json`.
