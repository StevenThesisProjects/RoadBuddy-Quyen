# RoadBuddy experiment report — debug20

## Result summary

The best configuration in scope **debug20** is **zero_shot_F1**, with accuracy **0.7000** and macro-F1 **0.6637** on 20 validation samples.

## Leaderboard

| position | phase | experiment | frames | rank | accuracy | macro_f1 | parse_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 02 | zero_shot_F1 | 1 | 0 | 0.7000 | 0.6637 | 1.0000 |
| 2 | 01 | zero_shot_f1 | 1 | 0 | 0.6500 | 0.5985 | 1.0000 |
| 3 | 03 | lora_r8_F1 | 1 | 8 | 0.6500 | 0.5985 | 1.0000 |
| 4 | 02 | zero_shot_F3 | 3 | 0 | 0.6500 | 0.5967 | 1.0000 |
| 5 | 02 | lora_r16_F1 | 1 | 16 | 0.6000 | 0.6162 | 1.0000 |
| 6 | 02 | zero_shot_F8 | 8 | 0 | 0.6000 | 0.6035 | 1.0000 |
| 7 | 02 | lora_r16_F3 | 3 | 16 | 0.6000 | 0.6035 | 1.0000 |
| 8 | 02 | lora_r16_F8 | 8 | 16 | 0.6000 | 0.6035 | 1.0000 |
| 9 | 01 | lora_r16_f1 | 1 | 16 | 0.6000 | 0.5694 | 1.0000 |

## Reproducibility contract

- Run scope: `debug20`
- Model: `5CD-AI/Vintern-1B-v3_5`
- Revision: `b98f263eab246eb5269ade64edbdca8a887dc44d`
- Seed: `42`
- Native template: `Hermes-2`
- FlashAttention2: disabled
- Phase02 inner language attention: PyTorch SDPA
- Frozen validation ID SHA-256: `dbfa2d337f56bd681df70206fa8cee83d743c6743a3493619083d03b804145c5`
- Dynamic visual token expansion: `model.num_image_token`
- Image flags: one per visual tile
- LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`

## Interpretation constraints

- This report uses `debug20` artifacts; it is not a full-phase claim when the scope is debug.
- Question-type slices with fewer than 20 examples are exploratory.
- All current rows have `question_type=unknown`, so type-specific conclusions are unavailable.
- Frame-count and rank conclusions apply only to this frozen prediction subset and model revision.
