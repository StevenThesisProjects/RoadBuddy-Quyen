from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms.functional import InterpolationMode


MODEL_ID = "5CD-AI/Vintern-1B-v3_5"
MODEL_REVISION = "b98f263eab246eb5269ade64edbdca8a887dc44d"
PROJECT_ROOT = Path("/workspace/RoadBuddy")
SEED = 42
IMAGE_SIZE = 448
MAX_DYNAMIC_TILES = 6
USE_THUMBNAIL = True
IGNORE_INDEX = -100
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT

    @property
    def raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def phase1_split(self) -> Path:
        return self.project_root / "data" / "splits" / "phase01"

    @property
    def phase1_output(self) -> Path:
        return self.project_root / "outputs" / "phase01"

    @property
    def zero_shot(self) -> Path:
        return self.phase1_output / "zero_shot_f1"

    @property
    def lora(self) -> Path:
        return self.phase1_output / "lora_r16_f1"


PATHS = Paths()


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    for path in (PATHS.raw, PATHS.phase1_split, PATHS.zero_shot, PATHS.lora):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            records = payload["data"]
            declared_count = payload.get("__count__")
            if declared_count is not None and int(declared_count) != len(records):
                raise ValueError(f"Declared __count__={declared_count} does not match {len(records)} records in {path}")
            return pd.DataFrame.from_records(records)
        return pd.DataFrame.from_records(payload)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


ALIASES = {
    "sample_id": ("sample_id", "id", "question_id", "qid"),
    "group_id": ("group_id", "video_id", "clip_id", "drive_id", "route_id"),
    "video_path": ("video_path", "video", "path", "file_path", "filepath"),
    "question": ("question", "query", "prompt"),
    "answer": ("answer", "label", "correct_answer", "target"),
    "frame_time_sec": ("frame_time_sec", "timestamp", "time_sec", "time", "support_frames"),
    "question_type": ("question_type", "type", "category"),
    "choices": ("choices", "options", "answers"),
    "option_a": ("option_a", "a", "choice_a", "answer_a"),
    "option_b": ("option_b", "b", "choice_b", "answer_b"),
    "option_c": ("option_c", "c", "choice_c", "answer_c"),
    "option_d": ("option_d", "d", "choice_d", "answer_d"),
}


def infer_schema(columns: Iterable[str], overrides: Optional[dict[str, str]] = None) -> dict[str, str]:
    overrides = overrides or {}
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    schema: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        if canonical in overrides:
            schema[canonical] = overrides[canonical]
            continue
        for alias in aliases:
            if alias in lookup:
                schema[canonical] = lookup[alias]
                break
    return schema


def _answer_letter(raw: Any, options: dict[str, str]) -> str:
    text = str(raw).strip()
    match = re.search(r"(?:^|\b)([ABCD])(?:\b|[.)])", text.upper())
    if match:
        return match.group(1)
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    matches = [letter for letter, value in options.items() if re.sub(r"\s+", " ", value).strip().casefold() == normalized]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Cannot map answer {raw!r} to A/B/C/D")


def _choice_options(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"choices must be a list, got {raw!r}") from exc
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(f"choices must be a list, got {type(raw).__name__}")

    unique_choices = []
    for raw_choice in raw:
        choice = str(raw_choice).strip()
        if choice not in unique_choices:
            unique_choices.append(choice)
    if not 2 <= len(unique_choices) <= len(CHOICES):
        raise ValueError(f"Expected 2-4 distinct choices, got {len(unique_choices)}: {raw!r}")

    options = {letter: "" for letter in CHOICES}
    for letter, choice in zip(CHOICES, unique_choices):
        value = re.sub(r"^[A-D]\s*[.)]\s*", "", choice, flags=re.IGNORECASE).strip()
        if not value:
            raise ValueError(f"Choice text may not be empty: {raw!r}")
        options[letter] = value
    return options

def normalize_dataset(
    frame: pd.DataFrame,
    *,
    video_root: Path,
    schema_overrides: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    schema = infer_schema(frame.columns, schema_overrides)
    required = {"video_path", "question", "answer"}
    missing = sorted(required - schema.keys())
    has_choice_list = "choices" in schema
    missing_option_columns = sorted({f"option_{letter.lower()}" for letter in CHOICES} - schema.keys())
    if not has_choice_list and missing_option_columns:
        missing.extend(missing_option_columns)
    if missing:
        raise ValueError(f"Missing required canonical fields {missing}. Columns: {list(frame.columns)}; inferred: {schema}")

    rows = []
    for row_number, (_, source) in enumerate(frame.iterrows()):
        if has_choice_list:
            options = _choice_options(source[schema["choices"]])
        else:
            options = {letter: str(source[schema[f"option_{letter.lower()}"]]).strip() for letter in CHOICES}
        relative = Path(str(source[schema["video_path"]]).strip())
        video_path = relative if relative.is_absolute() else (video_root / relative)
        group_id = str(source[schema["group_id"]]).strip() if "group_id" in schema else relative.stem
        question = str(source[schema["question"]]).strip()
        raw_id = str(source[schema["sample_id"]]).strip() if "sample_id" in schema else ""
        if not raw_id or raw_id.lower() == "nan":
            digest = hashlib.sha1(f"{group_id}|{question}|{row_number}".encode("utf-8")).hexdigest()[:16]
            raw_id = f"rb_{digest}"
        timestamp = source[schema["frame_time_sec"]] if "frame_time_sec" in schema else np.nan
        if isinstance(timestamp, (list, tuple, np.ndarray)):
            timestamp = timestamp[0] if len(timestamp) else np.nan
        qtype = str(source[schema["question_type"]]).strip() if "question_type" in schema else "unknown"
        rows.append({
            "sample_id": raw_id,
            "group_id": group_id,
            "video_path": str(video_path.resolve()),
            "question": question,
            **{f"option_{letter.lower()}": value for letter, value in options.items()},
            "answer": _answer_letter(source[schema["answer"]], options),
            "frame_time_sec": pd.to_numeric(timestamp, errors="coerce"),
            "question_type": qtype if qtype and qtype.lower() != "nan" else "unknown",
        })
    result = pd.DataFrame(rows)
    if result["sample_id"].duplicated().any():
        duplicates = result.loc[result["sample_id"].duplicated(False), "sample_id"].tolist()[:10]
        raise ValueError(f"sample_id must be unique; duplicates include {duplicates}")
    if result[[f"option_{x.lower()}" for x in CHOICES]].isna().any().any():
        raise ValueError("Options may not be null")
    return result


def validate_video_paths(frame: pd.DataFrame, check_decode: bool = False) -> pd.DataFrame:
    checks = []
    for video_path in sorted(frame["video_path"].unique()):
        path = Path(video_path)
        exists = path.is_file()
        decodes = None
        if exists and check_decode:
            capture = cv2.VideoCapture(str(path))
            decodes = bool(capture.isOpened() and capture.read()[0])
            capture.release()
        checks.append({"video_path": video_path, "exists": exists, "decodes": decodes})
    return pd.DataFrame(checks)


def freeze_group_split(frame: pd.DataFrame, validation_fraction: float = 0.2, seed: int = SEED):
    groups = sorted(frame["group_id"].astype(str).unique())
    if len(groups) < 2:
        raise ValueError("Need at least two video/group IDs for a leakage-safe split")
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    val_groups = set(groups[:n_val])
    val = frame[frame["group_id"].astype(str).isin(val_groups)].copy()
    train = frame[~frame["group_id"].astype(str).isin(val_groups)].copy()
    assert set(train["group_id"]).isdisjoint(set(val["group_id"]))
    assert set(train["sample_id"]).isdisjoint(set(val["sample_id"]))
    return train.reset_index(drop=True), val.reset_index(drop=True)


def split_manifest(train: pd.DataFrame, val: pd.DataFrame, source_files: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": SEED,
        "strategy": "provided_validation" if len(source_files) > 1 else "seeded_group_video_split",
        "source_files": source_files,
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_groups": int(train["group_id"].nunique()),
        "validation_groups": int(val["group_id"].nunique()),
        "validation_sample_ids_file": str(PATHS.phase1_split / "validation_sample_ids.json"),
    }


def build_transform(input_size: int = IMAGE_SIZE):
    return T.Compose([
        T.Lambda(lambda image: image.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _closest_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best, best_diff = (1, 1), float("inf")
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]):
            best, best_diff = ratio, diff
    return best


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=MAX_DYNAMIC_TILES, image_size=IMAGE_SIZE, use_thumbnail=USE_THUMBNAIL):
    width, height = image.size
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num}, key=lambda pair: pair[0] * pair[1])
    ratio = _closest_ratio(width / height, target_ratios, width, height, image_size)
    resized = image.resize((image_size * ratio[0], image_size * ratio[1]))
    tiles = []
    for index in range(ratio[0] * ratio[1]):
        x = index % ratio[0]
        y = index // ratio[0]
        tiles.append(resized.crop((x * image_size, y * image_size, (x + 1) * image_size, (y + 1) * image_size)))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def read_video_frame(video_path: str, frame_time_sec: float | None = None) -> Image.Image:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    target = duration / 2 if frame_time_sec is None or pd.isna(frame_time_sec) else float(frame_time_sec)
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target) * 1000.0)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame at {target:.3f}s: {video_path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_visual_tiles(row: pd.Series, dtype=torch.bfloat16) -> torch.Tensor:
    image = read_video_frame(row["video_path"], row.get("frame_time_sec"))
    transform = build_transform()
    return torch.stack([transform(tile) for tile in dynamic_preprocess(image)]).to(dtype=dtype)


def format_mcq(row: pd.Series) -> str:
    available = [letter for letter in CHOICES if str(row[f"option_{letter.lower()}"]).strip()]
    options = "\n".join(f"{letter}. {row[f'option_{letter.lower()}']}" for letter in available)
    return "<image>\nAnswer the multiple-choice question using only one letter: A, B, C, or D.\n" + f"Question: {row['question']}\n{options}"


def parse_choice(text: str) -> Optional[str]:
    upper = str(text).strip().upper()
    patterns = (r"^\s*([ABCD])(?:\b|[.)])", r"(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b", r"\b([ABCD])\b")
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return None


def load_model_and_tokenizer(training: bool = False, attn_implementation: str = "eager"):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        attn_implementation=attn_implementation,
    ).cuda()
    if training:
        model.train()
    else:
        model.eval()
    if not hasattr(model, "num_image_token") or int(model.num_image_token) <= 0:
        raise RuntimeError("Pinned Vintern model does not expose a valid model.num_image_token")
    if str(model.template) != "Hermes-2":
        raise RuntimeError(f"Expected native Hermes-2 template, got {model.template!r}")
    language_context = int(getattr(model.language_model.config, "max_position_embeddings", 4096))
    tokenizer.model_max_length = min(4096, language_context)
    return model, tokenizer


def native_chat(model, tokenizer, pixel_values: torch.Tensor, question: str, max_new_tokens: int = 16) -> str:
    tiles = int(pixel_values.shape[0])
    if tiles <= 0:
        raise ValueError("pixel_values must contain at least one visual tile")
    native_model = _unwrap_vintern(model)
    required_parameters = {"tokenizer", "pixel_values", "question", "generation_config", "num_patches_list"}
    missing_parameters = sorted(required_parameters - set(inspect.signature(native_model.chat).parameters))
    if missing_parameters:
        raise RuntimeError(f"Pinned native chat API is missing required parameters: {missing_parameters}")
    with torch.inference_mode():
        return native_model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config={"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1},
            num_patches_list=[tiles],
            history=None,
            return_history=False,
        )


def read_video_frames(video_path: str, frame_count: int) -> list[Image.Image]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Video reports no frames: {video_path}")

    indices = [min(total_frames - 1, max(0, int((index + 0.5) * total_frames / frame_count))) for index in range(frame_count)]
    frames = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_index} from {video_path}")
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return frames


def load_visual_tiles_multiframe(row: pd.Series, frame_count: int, dtype=torch.bfloat16) -> tuple[torch.Tensor, list[int]]:
    transform = build_transform()
    all_tiles = []
    num_patches_list = []
    for image in read_video_frames(str(row["video_path"]), frame_count):
        frame_tiles = dynamic_preprocess(image)
        num_patches_list.append(len(frame_tiles))
        all_tiles.extend(transform(tile) for tile in frame_tiles)
    if len(num_patches_list) != frame_count or not all_tiles:
        raise RuntimeError(f"Invalid multiframe tile expansion: frames={frame_count}, patches={num_patches_list}")
    return torch.stack(all_tiles).to(dtype=dtype), num_patches_list


def native_chat_multiframe(
    model,
    tokenizer,
    pixel_values: torch.Tensor,
    question: str,
    num_patches_list: list[int],
    max_new_tokens: int = 16,
) -> str:
    if sum(num_patches_list) != int(pixel_values.shape[0]):
        raise ValueError("sum(num_patches_list) must match the visual tile batch")
    if question.count("<image>") != 1:
        raise ValueError("The canonical MCQ prompt must contain exactly one <image> placeholder")
    multiframe_question = question.replace("<image>", "\n".join(["<image>"] * len(num_patches_list)), 1)
    native_model = _unwrap_vintern(model)
    required_parameters = {"tokenizer", "pixel_values", "question", "generation_config", "num_patches_list"}
    missing_parameters = sorted(required_parameters - set(inspect.signature(native_model.chat).parameters))
    if missing_parameters:
        raise RuntimeError(f"Pinned native chat API is missing required parameters: {missing_parameters}")
    original_max_length = tokenizer.model_max_length
    language_context = int(getattr(native_model.language_model.config, "max_position_embeddings", original_max_length))
    tokenizer.model_max_length = max(original_max_length, language_context)
    try:
        with torch.inference_mode():
            return native_model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=multiframe_question,
                generation_config={"max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1},
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
    finally:
        tokenizer.model_max_length = original_max_length


def evaluate_rows_multiframe(
    model,
    tokenizer,
    frame: pd.DataFrame,
    frame_count: int,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    selected = frame.head(limit) if limit else frame
    for _, row in selected.iterrows():
        pixels, num_patches_list = load_visual_tiles_multiframe(row, frame_count)
        pixels = pixels.cuda(non_blocking=True)
        response = native_chat_multiframe(model, tokenizer, pixels, format_mcq(row), num_patches_list)
        prediction = parse_choice(response)
        records.append({
            "sample_id": row["sample_id"], "group_id": row["group_id"], "question_type": row["question_type"],
            "answer": row["answer"], "prediction": prediction, "raw_response": response,
            "correct": prediction == row["answer"], "frame_count": frame_count,
            "num_tiles": int(pixels.shape[0]), "num_patches_list": json.dumps(num_patches_list),
        })
    predictions = pd.DataFrame(records)
    valid_predictions = predictions["prediction"].fillna("INVALID")
    metrics = {
        "frame_count": frame_count,
        "rows": len(predictions),
        "accuracy": float(accuracy_score(predictions["answer"], valid_predictions)),
        "macro_f1": float(f1_score(predictions["answer"], valid_predictions, labels=list(CHOICES), average="macro", zero_division=0)),
        "parse_rate": float(predictions["prediction"].notna().mean()),
        "mean_tiles": float(predictions["num_tiles"].mean()),
    }
    return predictions, metrics

def _unwrap_vintern(model):
    candidates = [model]
    direct_model = getattr(model, "model", None)
    base_model = getattr(model, "base_model", None)
    if direct_model is not None:
        candidates.append(direct_model)
    if base_model is not None:
        candidates.append(base_model)
        nested_model = getattr(base_model, "model", None)
        if nested_model is not None:
            candidates.append(nested_model)
    for candidate in candidates:
        if hasattr(candidate, "num_image_token") and "internvl" in candidate.__class__.__module__.lower():
            return candidate
    return model


def _conversation_factory(model):
    native_model = _unwrap_vintern(model)
    module = importlib.import_module(native_model.__class__.__module__)
    factory = getattr(module, "get_conv_template", None)
    if factory is None:
        raise RuntimeError("Pinned remote-code module does not export get_conv_template")
    return factory


def _replace_image_placeholder(prompt: str, num_tiles: int, num_image_token: int) -> str:
    visual_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_image_token * num_tiles) + IMG_END_TOKEN
    if "<image>" not in prompt:
        raise ValueError("Prompt must contain exactly one <image> placeholder")
    return prompt.replace("<image>", visual_tokens, 1)


def build_training_tokens(model, tokenizer, question: str, answer: str, num_tiles: int, max_length: int = 4096):
    native_model = _unwrap_vintern(model)
    factory = _conversation_factory(model)
    prefix_conv = factory(native_model.template)
    prefix_conv.system_message = native_model.system_message
    prefix_conv.append_message(prefix_conv.roles[0], question)
    prefix_conv.append_message(prefix_conv.roles[1], None)
    prefix = _replace_image_placeholder(prefix_conv.get_prompt(), num_tiles, int(native_model.num_image_token))

    full_conv = factory(native_model.template)
    full_conv.system_message = native_model.system_message
    full_conv.append_message(full_conv.roles[0], question)
    full_conv.append_message(full_conv.roles[1], answer)
    full = _replace_image_placeholder(full_conv.get_prompt(), num_tiles, int(native_model.num_image_token))

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    common = 0
    for left, right in zip(prefix_ids, full_ids):
        if left != right:
            break
        common += 1
    if common < len(prefix_ids) - 2:
        raise RuntimeError(f"Native-template token prefix diverged too early: {common}/{len(prefix_ids)}")
    if len(full_ids) > max_length:
        raise ValueError(f"Tokenized sample length {len(full_ids)} exceeds max_length={max_length}; do not truncate visual tokens")
    # Mask the common token prefix. This remains correct if the tokenizer merges
    # one trailing assistant-prefix token with the first answer token.
    labels = [IGNORE_INDEX] * common + full_ids[common:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise RuntimeError("No supervised assistant tokens after token-prefix masking")
    expected_visual = int(native_model.num_image_token) * num_tiles
    actual_visual = sum(token_id == native_model.img_context_token_id for token_id in full_ids)
    if actual_visual != expected_visual:
        raise RuntimeError(f"Visual-token alignment failed: expected {expected_visual}, found {actual_visual}")
    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class RoadBuddySFTDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, model, tokenizer, limit: Optional[int] = None):
        self.frame = frame.head(limit).reset_index(drop=True) if limit else frame.reset_index(drop=True)
        self.model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        pixels = load_visual_tiles(row)
        input_ids, labels = build_training_tokens(self.model, self.tokenizer, format_mcq(row), row["answer"], len(pixels))
        return {"sample_id": row["sample_id"], "input_ids": input_ids, "labels": labels, "pixel_values": pixels, "image_flags": torch.ones(len(pixels), dtype=torch.long)}


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate(items):
        max_len = max(len(item["input_ids"]) for item in items)
        input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(items), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, item in enumerate(items):
            length = len(item["input_ids"])
            input_ids[i, :length] = item["input_ids"]
            labels[i, :length] = item["labels"]
            attention_mask[i, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": torch.cat([item["pixel_values"] for item in items]),
            "image_flags": torch.cat([item["image_flags"] for item in items]),
            "sample_ids": [item["sample_id"] for item in items],
        }
    return collate


def evaluate_rows(model, tokenizer, frame: pd.DataFrame, limit: Optional[int] = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    selected = frame.head(limit) if limit else frame
    for _, row in selected.iterrows():
        pixels = load_visual_tiles(row).cuda(non_blocking=True)
        response = native_chat(model, tokenizer, pixels, format_mcq(row))
        prediction = parse_choice(response)
        records.append({
            "sample_id": row["sample_id"], "group_id": row["group_id"], "question_type": row["question_type"],
            "answer": row["answer"], "prediction": prediction, "raw_response": response,
            "correct": prediction == row["answer"], "num_tiles": len(pixels),
        })
    predictions = pd.DataFrame(records)
    valid_predictions = predictions["prediction"].fillna("INVALID")
    metrics = {
        "rows": len(predictions),
        "accuracy": float(accuracy_score(predictions["answer"], valid_predictions)),
        "macro_f1": float(f1_score(predictions["answer"], valid_predictions, labels=list(CHOICES), average="macro", zero_division=0)),
        "parse_rate": float(predictions["prediction"].notna().mean()),
    }
    return predictions, metrics
