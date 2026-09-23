# RoadBuddy Phase 1 Report

## 1. Objective

Phase 1 repaired and validated the Vintern Traffic Video QA pipeline so that
training, validation, public-test inference and submission generation are
reproducible from one notebook.

## 2. Critical fixes implemented

- Validated `<IMG_CONTEXT>` as one non-unknown tokenizer token.
- Removed fixed image-token assumptions from the notebook pipeline.
- Used the Vintern multimodal chat route for inference.
- Applied token-level answer masking.
- Replaced full fine-tuning with LoRA.
- Kept the frozen base model in FP16 and trainable LoRA weights in FP32.
- Disabled unsupported outer-model gradient checkpointing.
- Added grouped train/validation split by `video_id`.
- Fixed duplicated train/public-test video path prefixes.
- Added path, batch, label, wrapper and dtype guards.
- Verified that LoRA tensors actually changed during training.

## 3. Runtime evidence

- Training samples: **1171**
- Training videos: **439**
- Validation samples: **319**
- Validation videos: **110**
- Public-test samples: **405**
- Missing training video paths: **0**
- Missing public-test video paths: **0**
- Changed LoRA tensors: **192/192**
- Submission rows: **405**

## 4. Baseline validation metrics

- Raw exact-match accuracy:
  **9.40%**
- Normalized A/B/C/D accuracy:
  **33.86%**
- Empty-output rate:
  **0.00%**
- Parse-failure rate:
  **0.00%**

## 5. Submission

- File: `outputs/roadbuddy_phase1_final/artifacts/submission_phase1.csv`
- Rows: **405**
- Columns: `id, answer`
- SHA-256: `7dbd435711fa23fdf45072203e3b12274ffbeee2fe62a1fbd28a8df83ff73e60`

## 6. Error-analysis sample

A deterministic sample of **20** incorrect validation
predictions was exported for manual review.

- `visual_recognition`: 11
- `temporal_information_missing`: 7
- `traffic_rule_knowledge`: 1
- `insufficient_frame`: 1

The initial categories are heuristic suggestions. Inspect each source video and
revise `error_category`, `review_status` and `review_note` in
`phase1_error_analysis_20.csv`.

## 7. Remaining limitations

- Phase 1 uses only **1 frame per video**.
- Temporal information is not explicitly modeled.
- The current accuracy is a baseline result, not the final research result.
- Error categories require manual video-level verification.
- Multi-frame experiments, hyperparameter ablations, traffic knowledge and
  ensemble methods belong to Phase 2.

## 8. Phase 1 conclusion

**PASSED**

The critical inference failure was removed: validation and public-test
predictions are non-empty, LoRA updates are verified, and the submission is
generated reproducibly from the same notebook.

## 9. Phase 2 direction

1. Build 3-frame and 5-frame temporal baselines.
2. Measure accuracy by question/error category.
3. Run LoRA rank and learning-rate ablations.
4. Investigate traffic-domain knowledge augmentation after error analysis.
