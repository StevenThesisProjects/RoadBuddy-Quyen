# RoadBuddy — Traffic-Aware Temporal Grounding

## Research boundary

This is a novelty track proposed after the instructor review. It does not
replace full baseline validation. No temporal-grounding result may be described
as an improvement until the baseline winner has been locked on the 298 frozen
validation samples.

Support-frame annotations are weak labels, not inference inputs. They may be
used only on the training side. Validation and public-test frame selection must
use the learned scorer without support timestamps or answer labels.

## System definition

For each video/question pair:

1. Decode a deterministic pool of 32 uniform temporal-bin midpoint frames.
2. Extract a frozen visual embedding for every candidate frame.
3. Extract one frozen question embedding.
4. Optionally extract traffic features for every frame, such as traffic-light,
   sign, vehicle, pedestrian, lane, and motion evidence.
5. Score every candidate with `TrafficAwareTemporalGrounder`.
6. Select top-k candidates with an optional minimum temporal gap and restore
   chronological order before passing frames to the pinned VLM.

The selector is lightweight. The base visual/question encoders and the baseline
VLM remain frozen during selector training, so an observed change can be
attributed to frame grounding rather than a different VLM checkpoint.

## Training targets

Primary supervision is a soft Gaussian distribution around train-only
`support_frames`. If multiple support timestamps exist, their probability mass
is combined. Samples without verified temporal labels must either be excluded
from supervised selector training or handled by a separately preregistered
self-supervised objective; they must not receive fabricated positive frames.

The implementation rejects support-frame target construction when the split is
not named `train`.

## Required feature-bank schema

Each record must contain:

- `sample_id`, `group_id`, and `split`;
- candidate frame indices and timestamps;
- `frame_features[T,Dv]`;
- `question_features[Dq]`;
- optional `traffic_features[T,Dt]` with a documented feature vocabulary;
- `valid_mask[T]`;
- train-only relevance targets and their annotation provenance.

Every feature bank requires hashes for source video, question text, encoder
revision, preprocessing configuration, and split membership.

## Minimum experiment matrix

| ID | Frame selector | Traffic features | Selected k | Purpose |
|---|---|---:|---:|---|
| U-k | Uniform temporal bins | No | 1/3/8 | Locked baseline control |
| QTG-k | Learned question-guided | No | 1/3/8 | Grounding contribution |
| TATG-k | Learned question-guided | Yes | 1/3/8 | Traffic-feature contribution |
| RND-k | Seeded random candidates | No | 1/3/8 | Negative control |
| ORACLE-k | Human support frames | N/A | 1/3/8 | Diagnostic upper bound only |

All non-oracle systems use the same candidate pool, selected-frame budget,
Vintern revision, prompt, generation parameters, and frozen validation IDs.
Oracle results must never participate in winner selection or public-test
inference.

## Evaluation

Frame-grounding evaluation and answer evaluation are separate:

- Grounding: Recall@k against held-out support intervals, mAP or nDCG, and mean
  temporal distance to the nearest relevant interval.
- VQA: accuracy, macro-F1, parse rate, per-class F1, and paired group-cluster
  bootstrap confidence intervals against uniform sampling.
- Efficiency: feature extraction time, selector latency, VLM latency, peak VRAM,
  and selected visual-token count.

Primary claims require all 298 frozen validation predictions, exact membership
matching, a locked selector checkpoint selected on a train-side development
split, and correction for multiple comparisons.

## Artifacts still required before a scientific run

- The canonical 1,192/298 split files and checksums.
- A baseline-winner manifest from full validation.
- Train-only support-frame annotations with an interval/timestamp definition.
- A frozen visual/question encoder specification and extracted feature bank.
- A traffic-feature extractor, label vocabulary, checkpoint, and provenance.
- A group-safe inner train/development split for selector checkpointing.
- Full grounding labels or an annotation protocol for held-out evaluation.
- Run configs, selector checkpoints, raw scores for all candidates, selected
  indices/timestamps, predictions, metrics, and paired statistical tests.

## Implementation entry point

`src/traffic_temporal_grounding.py` provides:

- deterministic candidate generation;
- train-only weak-target construction;
- the traffic-aware question-conditioned scorer;
- masked grounding loss;
- diverse top-k selection returned in chronological order.

`src/roadbuddy_common.py` additionally provides
`read_video_frames_at_indices` and `load_visual_tiles_at_indices`, which decode
the exact selected indices and connect them to the existing Vintern
`num_patches_list` inference path without falling back to uniform sampling.

The module intentionally does not select an encoder or detector before their
datasets, revisions, and licensing/provenance are fixed.
