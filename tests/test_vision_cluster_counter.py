from __future__ import annotations

import json

from pupa_counter.vision.openai_cluster_counter import _extract_json_candidate


def test_extract_json_candidate_from_gemini_style_payload():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {"count": 3, "confidence": 0.78, "notes": "three overlapping pupa"}
                            )
                        }
                    ]
                }
            }
        ]
    }
    parsed = _extract_json_candidate(payload)
    assert parsed is not None
    assert parsed["count"] == 3
    assert parsed["confidence"] == 0.78
