from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pupa_counter.config import AppConfig
from pupa_counter.detect.components import extract_components
from pupa_counter.detect.features import featurize_components
from pupa_counter.detect.rule_filter import rule_classify_components
from pupa_counter.detect.split_clusters import split_cluster_candidates


def test_cluster_split_creates_child_components():
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (48, 60), (14, 8), 25, 0, 360, (175, 125, 80), -1)
    cv2.ellipse(image, (70, 60), (14, 8), -20, 0, 360, (175, 125, 80), -1)
    mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = ((255 - mask) > 20).astype(np.uint8) * 255

    components = extract_components(binary, AppConfig())
    blue_mask = np.zeros(binary.shape, dtype=np.uint8)
    featured = featurize_components(image, blue_mask, components)
    labeled = rule_classify_components(featured, AppConfig())
    labeled.loc[:, "label"] = "cluster"

    split = split_cluster_candidates(image, labeled, blue_mask=blue_mask, cfg=AppConfig())
    assert any(str(component_id).endswith("_child_01") for component_id in split["component_id"].tolist())
