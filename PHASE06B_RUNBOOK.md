# RoadBuddy Phase 06B Runbook

## Mục tiêu

Phase 06B dùng baseline đã khóa `L32-F1` và taxonomy do con người gán để chọn đúng một novelty track. Phase này không truy cập public-test labels, không tạo submission và không dùng frozen validation để chọn checkpoint hoặc tinh chỉnh protocol.

## Thứ tự notebook

1. `notebooks/Phase06B_00_Taxonomy_Slice_Analysis.ipynb`
2. `notebooks/Phase06B_01_Novelty_Track_Decision_Lock.ipynb`
3. Chạy notebook preflight đúng với track đã khóa:
   - `notebooks/Phase06B_02_Knowledge_Augmented_Preflight.ipynb`; hoặc
   - `notebooks/Phase06B_03_Temporal_Grounding_Preflight.ipynb`.
4. Sau khi train-side development và full novelty inference hoàn tất, chạy `notebooks/Phase06B_04_Novelty_Statistical_Analysis_Winner_Lock.ipynb`.

Notebook preflight của track không được chọn sẽ ghi `status="not_selected"`; đây là trạng thái hợp lệ.

## Gate 00 — taxonomy

Các file human-authored bắt buộc:

- `outputs/phase06a/question_taxonomy/annotator_1.csv`
- `outputs/phase06a/question_taxonomy/annotator_2.csv`
- `outputs/phase06a/question_taxonomy/taxonomy_adjudicated.csv`
- `outputs/phase06a/question_taxonomy/taxonomy_frozen.csv`
- `outputs/phase06a/question_taxonomy/taxonomy_manifest.json`

Nếu thiếu, Stage 00 dừng với `awaiting_annotation`. Không tự sinh nhãn.

## Gate 01 — human novelty decision

Stage 01 tạo `novelty_track_decision.template.json`. Người nghiên cứu phải tạo `novelty_track_decision.json`, khóa một trong hai giá trị:

- `knowledge_augmented`
- `traffic_temporal_grounding`

`evidence_sha256` phải khớp evidence hiện tại. Recommendation của Stage 00 chỉ là diagnostic, không tự động thay thế quyết định.

## Gate 02 — Knowledge-Augmented VQA

Corpus manifest phải lưu local path, SHA-256, URL, issuing authority, ngày hiệu lực và access/license note cho từng nguồn. Notebook chỉ khóa corpus và matrix `L32-F1`, `Static-K`, `RAG-k`, `Oracle-K`; nó không tải dữ liệu mạng. `Oracle-K` chỉ là subset diagnostic và không được vào winner selection.

## Gate 03 — Traffic-Aware Temporal Grounding

Input manifest phải khóa visual/question encoders, revisions, feature-bank schema, 32 candidates và train-only support annotations. Matrix gồm `U/QTG/TATG/RND/ORACLE` với `k={1,3,8}`. `ORACLE-k` không được vào winner selection hoặc public-test inference.

## Gate 04 — novelty statistics

Tạo `outputs/phase06b/experiments/novelty_experiment_registry.json` từ template, chỉ đăng ký các arm đã preregister và winner-eligible. Mỗi prediction CSV phải:

- có đúng 298 frozen IDs;
- giữ raw response, parse status, selected frames/passages, latency và provenance theo protocol;
- có SHA-256 khớp registry.

Improvement gate đã khóa:

- lower bound của 95% paired group-cluster bootstrap CI cho accuracy delta phải lớn hơn 0;
- exact McNemar Holm-adjusted `p < 0.05`.

Nếu không arm nào qua cả hai gate, `L32-F1` vẫn là winner. Mỗi comparison dùng 10.000 bootstrap resamples.

## Kiểm tra notebook trước execution

```bash
source /root/venvs/roadbuddy-rtx3090-py310/bin/activate
python -m unittest discover -s tests -v
python - <<'PY'
import ast, pathlib, nbformat
for path in sorted(pathlib.Path('notebooks').glob('Phase06B_*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
