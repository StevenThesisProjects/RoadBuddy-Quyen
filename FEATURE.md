# RoadBuddy — notebook-first experiment pipeline

Project root on the server is fixed at `/workspace/RoadBuddy`.

## Install this package

Copy the contents of this folder into `/workspace/RoadBuddy` so that `notebooks/` and `src/` sit directly under the project root. Activate the existing environment and register it once as a Jupyter kernel:

```bash
source /root/venvs/roadbuddy-rtx3090-py310/bin/activate
python -m ipykernel install --user --name roadbuddy-rtx3090-py310 --display-name "RoadBuddy RTX 3090 (Python 3.10)"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
cd /workspace/RoadBuddy
jupyter lab
```

Select **RoadBuddy RTX 3090 (Python 3.10)** for every notebook.

## Phase 01 run order

1. `Phase01_00_Environment_Preflight.ipynb`
2. `Phase01_01_Dataset_Contract_Split_Freeze.ipynb`
3. `Phase01_02_ZeroShot_Baseline.ipynb`
4. `Phase01_03_LoRA_R16_Training.ipynb`
5. `Phase01_04_LoRA_R16_Evaluation.ipynb`
6. `Phase01_05_Comparison.ipynb`

Run each notebook top-to-bottom. Do not run training until preflight, dataset validation, and a small zero-shot debug run pass.

## Frozen corrections

- Model `5CD-AI/Vintern-1B-v3_5`, revision `b98f263eab246eb5269ade64edbdca8a887dc44d`.
- Visual expansion uses `model.num_image_token`; no hard-coded token count.
- Native `Hermes-2` conversation template.
- Labels are masked by tokenized assistant-prefix length, never character length.
- `image_flags` contains one entry for each dynamic visual tile.
- Seed 42 and leakage-safe group/video splitting.
- FlashAttention is disabled (`use_flash_attn=False`, eager attention) for correctness-first runs.
- Validation membership is frozen in `data/splits/phase01/validation_sample_ids.json`.

## Canonical dataset contract

The normalized tables contain: `sample_id`, `group_id`, `video_path`, `question`, `option_a`, `option_b`, `option_c`, `option_d`, `answer`, `frame_time_sec`, and `question_type`. Edit only the schema/path configuration cell in notebook 01 when raw column names differ.

## Naming convention

Use `PhaseNN_SS_Descriptive_Name.ipynb`, where `NN` is the experimental phase and `SS` is the order inside it. Outputs use `outputs/phaseNN/<experiment_slug>/`; immutable split artifacts use `data/splits/phaseNN/`. Never overwrite another experiment's config or predictions.

Skeleton notebooks reserve the next phases:

- Phase 02: multi-frame F1/F3/F8 experiments.
- Phase 03: LoRA rank ablation r={8,16,32,64}.
- Phase 04: question-type analysis.
- Phase 05: final comparison and report.

## Novelty track: Traffic-Aware Temporal Grounding

The instructor-proposed temporal-grounding track is specified in
`TRAFFIC_TEMPORAL_GROUNDING.md` and implemented at the model-independent level
in `src/traffic_temporal_grounding.py`. It is gated behind full baseline
validation: do not interpret selector smoke tests as VQA improvements, and do
not use support-frame annotations outside the training side.

## Phase 06A run order — full baseline validation

Run these notebooks top-to-bottom after the Phase 01 canonical split has been
restored on the RTX 3090 server:

1. `Phase06A_00_Frozen_Data_Audit_Protocol_Lock.ipynb`
2. `Phase06A_01_Question_Taxonomy_Freeze.ipynb`
3. `Phase06A_02_Inner_Dev_Checkpoint_Protocol.ipynb`
4. `Phase06A_03_ZeroShot_Full_Validation.ipynb`
5. `Phase06A_04_LoRA_R16_Full_Training.ipynb`
6. `Phase06A_05_LoRA_R16_Full_Evaluation.ipynb`
7. `Phase06A_06_LoRA_Rank_Ablation.ipynb`
8. `Phase06A_07_Statistical_Analysis_Winner_Lock.ipynb`

Every GPU notebook defaults to an explicit `smoke` scope. Full artifacts are
written only when `RUN_SCOPE="full"` and the canonical 298-ID checksum and
upstream status gates pass. Phase 06A never creates a held-out-test submission.
See `PHASE06A_RUNBOOK.md` for server execution and resume instructions.

The skeletons intentionally create no training results; they validate dependencies and reserve consistent artifact paths for the next implementation round.
