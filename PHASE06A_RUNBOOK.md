# RoadBuddy Phase 06A runbook

## Objective and boundary

Phase 06A validates the baseline matrix on exactly 298 frozen validation
samples and locks one baseline winner. It does not create held-out-test
predictions or submissions. Debug20 and the legacy 319-row split are excluded
from all full-scope decisions.

Traffic-Aware Temporal Grounding remains gated until
`outputs/phase06a/final_analysis/baseline_winner_manifest.json` exists with
`status=complete`.

## Required server inputs

- `/workspace/RoadBuddy/data/splits/phase01/train.csv`
- `/workspace/RoadBuddy/data/splits/phase01/validation.csv`
- `/workspace/RoadBuddy/data/splits/phase01/validation_sample_ids.json`
- all train/validation video files referenced by those tables
- the pinned Vintern revision in the Hugging Face cache or network access
- kernel `roadbuddy-rtx3090-py310`

Canonical full contract:

- 1,192 train rows / 439 groups
- 298 validation rows / 110 groups
- validation ID file SHA-256
  `dbfa2d337f56bd681df70206fa8cee83d743c6743a3493619083d03b804145c5`
- zero sample and group overlap

## Preflight commands

```bash
cd /workspace/RoadBuddy
source /root/venvs/roadbuddy-rtx3090-py310/bin/activate
python -m unittest discover -s tests -v
```

Run Phase 01 environment preflight again if the GPU, kernel, CUDA, model cache,
or package environment has changed.

## Notebook order and gates

### 00 — Frozen data audit and protocol lock

Start with `RUN_SCOPE="smoke"`. Inspect decode and duplicate reports, then set
`RUN_SCOPE="full"`. Full PASS requires the exact canonical counts/checksum and
decode success for every referenced video.

Outputs under `outputs/phase06a/data_audit/<scope>/`:

- `dataset_audit.json`
- `video_decode_report.csv`
- `duplicate_video_report.csv`
- `phase06a_protocol.json`
- `PHASE06A_00_STATUS.json`

Do not continue full runs if the status is not `PASS`.

### 01 — Blind question taxonomy

Run once to generate the guideline and blank sheet. Two annotators independently
complete copies named `annotator_1.csv` and `annotator_2.csv` without viewing
model results. Rerun to generate agreement and adjudication files. Place the
final adjudicated table at `taxonomy_adjudicated.csv`, then rerun to freeze it.

The expected intermediate statuses are `awaiting_annotation` and
`awaiting_adjudication`; neither is an error or permission to fabricate labels.

### 02 — Inner development split

Creates `data/splits/phase06a_inner/` from the 1,192 training rows only. Review
class-distribution deviations and membership hashes. Frozen validation is not
loaded by this notebook.

### 03 — Zero-shot frame grid

Run smoke first, inspect raw responses, parse status, exact indices, tile counts,
latency, and VRAM, then use full scope. The independent arms are Z-F1, Z-F3,
and Z-F8. They share an eight-tile maximum budget with allocations `[8]`,
`[3,2,3]`, and `[1,1,1,1,1,1,1,1]`; thumbnails are disabled in this clean
compute-matched ablation.

### 04 — LoRA r16 two-stage training

Stage A trains on `train_fit` and selects the optimizer step on `inner_dev`.
Stage B resets base weights, adapter, optimizer, scheduler, and RNG, then trains
on all 1,192 training rows to the locked step. This notebook never evaluates
frozen validation.

Review before continuing:

- accumulated sample-ID provenance
- partial accumulation window sizes
- LR/loss/gradient logs
- changed LoRA tensor count
- adapter reload gate
- adapter and manifest hashes

### 05 — Locked r16 evaluation

Evaluates the same F1-trained adapter as L16-F1/L16-F3/L16-F8. F3/F8 are
inference-frame transfer configurations, not multi-frame fine-tuning.

### 06 — Rank ablation

Full scope runs r8/r16/r32/r64 with `alpha/r=2`. A protocol-compatible r16
adapter is reused from notebook 04. Each other rank is reset, selected on inner
development, retrained on all training rows, and evaluated on the same 298 IDs.

### 07 — Statistics and winner lock

This notebook is full-only. It requires nine explicit 298-row artifacts and
performs 10,000 paired video-group cluster bootstrap resamples, secondary exact
McNemar tests, Holm correction, and the preregistered 0.02 non-inferiority rule.
It writes the only authoritative baseline-winner manifest.

## Resume after interruption

- Never rename smoke artifacts to `full`.
- Each experiment arm has its own output directory and manifest.
- Re-run the interrupted notebook from its configuration/import cells; completed
  arms can be inspected independently, but the final statistics notebook still
  requires the complete explicit registry.
- Training checkpoints include adapter plus optimizer/scheduler/RNG state for
  audit. The current notebook intentionally performs clean stage resets; do not
  resume Stage B from Stage A state.
- If source, protocol, split, or checkpoint hashes change, rerun all downstream
  dependent arms rather than mixing provenance.

## Artifacts to download from the server

Download small research artifacts:

- all Phase 06A configs and status JSON files
- data/split manifests and ID lists
- training histories and best-step decision
- predictions, metrics, runtimes, and run manifests
- statistical CSV/JSON outputs
- taxonomy guideline, agreement, adjudication, and frozen taxonomy
- `baseline_winner_manifest.json`
- `PHASE06A_REPRODUCIBILITY_REPORT.md`

Keep large base-model weights and videos on the server. Download final LoRA
adapters only when they are needed for archival or another compute environment.

## Hard-stop conditions

Stop and investigate if any of these occur:

- validation checksum or count differs from the canonical contract
- any train/validation group overlap
- a required video cannot be decoded
- exact duplicate video leakage across splits
- full prediction membership is not exactly 298 unique IDs
- raw response is missing for a parse failure
- a parse failure is counted as correct
- checkpoint selection reads frozen-validation metrics
- two compared runs use different model revision, prompt, parser, frame policy,
  attention implementation, or tile-budget protocol

