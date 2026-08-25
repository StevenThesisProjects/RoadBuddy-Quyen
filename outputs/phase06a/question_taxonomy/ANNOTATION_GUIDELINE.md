# RoadBuddy question taxonomy guideline

Annotate without viewing model predictions or correctness. Binary axes use 0/1.

- visual_required: a static visual observation is necessary.
- temporal_required: motion, order, direction, or change over time is necessary.
- traffic_knowledge_required: an external traffic rule/convention is necessary.
- mixed_or_ambiguous: evidence requirements cannot be assigned cleanly.
- primary_label: visual_static, temporal, traffic_knowledge, mixed, or ambiguous.

Inspect the video when the question alone is insufficient. Do not consult prediction artifacts.
