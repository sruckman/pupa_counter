"""Optional vision-model fallback for difficult cluster crops.

Supported providers:
- ``gemini`` via the Gemini REST API
- ``openai`` via the OpenAI Responses API

The fallback is only active when:
- ``vision_fallback.enabled`` is true
- a matching API key exists in the environment
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig


def _resize_for_model(image: np.ndarray, max_side_px: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side_px) / max(height, width))
    if scale >= 1.0:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _encode_png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Failed to encode cluster crop as PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _cluster_crop(image: np.ndarray, row: pd.Series, padding_px: int) -> np.ndarray:
    y0 = max(0, int(row["bbox_y0"]) - padding_px)
    x0 = max(0, int(row["bbox_x0"]) - padding_px)
    y1 = min(image.shape[0], int(row["bbox_y1"]) + padding_px)
    x1 = min(image.shape[1], int(row["bbox_x1"]) + padding_px)
    return image[y0:y1, x0:x1].copy()


def _extract_json_candidate(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    candidates: List[str] = []
    if isinstance(payload.get("output_text"), str):
        candidates.append(payload["output_text"])

    for item in payload.get("output", []) if isinstance(payload.get("output"), list) else []:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    candidates.append(text)
                if isinstance(content.get("json"), dict):
                    return content["json"]

    for candidate in payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts", []) if isinstance(content, dict) else []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                candidates.append(part["text"])

    for text in candidates:
        text = text.strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def _gemini_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _request_openai_cluster_count(
    image_b64: str,
    cfg: AppConfig,
    prompt: str,
    schema: Dict[str, object],
) -> Optional[Dict[str, object]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": cfg.vision_fallback.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": "data:image/png;base64,%s" % image_b64},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cluster_count",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": 200,
    }
    request = urllib.request.Request(
        cfg.vision_fallback.api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.vision_fallback.timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    return _extract_json_candidate(response_payload)


def _request_gemini_cluster_count(
    image_b64: str,
    cfg: AppConfig,
    prompt: str,
    schema: Dict[str, object],
) -> Optional[Dict[str, object]]:
    api_key = _gemini_api_key()
    if not api_key:
        return None

    model_name = cfg.vision_fallback.model
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model_name
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=cfg.vision_fallback.timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    return _extract_json_candidate(response_payload)


def estimate_cluster_counts_with_openai(
    image: np.ndarray,
    candidate_df: pd.DataFrame,
    cfg: AppConfig,
) -> pd.DataFrame:
    if candidate_df.empty or not cfg.vision_fallback.enabled:
        return pd.DataFrame()

    provider = str(cfg.vision_fallback.provider or "openai").strip().lower()
    unresolved = candidate_df.loc[
        candidate_df["is_active"].astype(bool)
        & candidate_df["cluster_unresolved"].astype(bool)
        & (
            candidate_df["cluster_fallback_eligible"].astype(bool)
            if "cluster_fallback_eligible" in candidate_df.columns
            else (candidate_df["label"].astype(str) == "cluster")
        )
    ].sort_values("area_px", ascending=False)
    unresolved = unresolved.head(cfg.vision_fallback.max_clusters_per_image)
    if unresolved.empty:
        return pd.DataFrame()

    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 12},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "notes": {"type": "string"},
        },
        "required": ["count", "confidence", "notes"],
        "additionalProperties": False,
    }
    prompt = (
        "Count the number of distinct brown or tan pupa-like bodies in this crop. "
        "Blue marks are never pupa. Ignore background paper, stains, or scan artifacts. "
        "If bodies overlap, estimate how many separate pupa are present and return JSON only."
    )

    rows = []
    for _, row in unresolved.iterrows():
        crop = _cluster_crop(image, row, cfg.vision_fallback.padding_px)
        crop = _resize_for_model(crop, cfg.vision_fallback.max_side_px)
        image_b64 = _encode_png_base64(crop)

        parsed = None
        if provider == "gemini":
            parsed = _request_gemini_cluster_count(image_b64, cfg, prompt, schema)
        else:
            parsed = _request_openai_cluster_count(image_b64, cfg, prompt, schema)

        if not isinstance(parsed, dict):
            continue
        count_value = int(parsed.get("count", 0))
        confidence_value = float(parsed.get("confidence", 0.0))
        if count_value < 1:
            continue
        rows.append(
            {
                "component_id": row["component_id"],
                "vision_cluster_count": count_value,
                "vision_cluster_confidence": confidence_value,
                "vision_cluster_notes": str(parsed.get("notes", "")),
                "vision_cluster_model": cfg.vision_fallback.model,
                "vision_cluster_provider": provider,
            }
        )

    return pd.DataFrame(rows)
