"""YAML-backed config loader for the pupa counting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProjectConfig:
    name: str = "pupa-counter"
    config_version: str = "baseline_v1"
    random_seed: int = 42


@dataclass
class InputConfig:
    raster_dpi: int = 300
    accepted_suffixes: List[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".pdf", ".pptx"]
    )


@dataclass
class PreprocessConfig:
    auto_crop_black_border: bool = True
    crop_dark_threshold: int = 25
    crop_min_run_ratio: float = 0.50
    max_border_crop_fraction: float = 0.08
    paper_white_min_value: int = 180
    paper_white_max_saturation: int = 80
    paper_support_ratio_threshold: float = 0.18
    paper_dynamic_col_quantile: float = 0.90
    paper_dynamic_col_scale: float = 0.68
    paper_smooth_window_px: int = 41
    paper_bbox_padding_px: int = 6
    paper_min_bbox_inside_fraction: float = 0.78
    background_percentile_low: float = 2.0
    background_percentile_high: float = 98.0
    clip_limit: float = 2.0


@dataclass
class BlueMaskConfig:
    enabled: bool = True
    hsv_lower_1: List[int] = field(default_factory=lambda: [85, 40, 40])
    hsv_upper_1: List[int] = field(default_factory=lambda: [140, 255, 255])
    lab_b_max: int = 115
    morphology_open_kernel: int = 3
    morphology_close_kernel: int = 5
    remove_mode: str = "exclude"
    inpaint_radius: int = 3


@dataclass
class BrownDetectionConfig:
    enabled: bool = True
    min_saturation: int = 18
    max_value: int = 250
    lab_a_min: int = 126
    brown_score_threshold: float = 0.18
    adaptive_block_size: int = 31
    adaptive_c: int = -3
    morphology_open_kernel: int = 3
    morphology_close_kernel: int = 3
    min_component_area_px: int = 20
    # Grayscale-mode auto-detection: when an image is essentially desaturated
    # (e.g. a clean PDF where pupae appear nearly black on white), the
    # brown-color path miss-fires because pupae have ~zero saturation.
    # In grayscale mode the detector skips the saturation gate and relies on
    # darkness + adaptive threshold instead.
    auto_grayscale_mode: bool = True
    grayscale_sat_median_threshold: float = 12.0
    grayscale_max_value: int = 180
    # Yellow imprint rejection. Brown pupa/egg pixels have lab b* ~140-155;
    # light yellow imprints (faded marks where an egg used to be) score
    # higher b* (~160-180). Cap b* to keep brown in and yellow out.
    max_lab_b: int = 158


@dataclass
class ComponentsConfig:
    min_area_px: int = 20
    max_area_px: int = 2500
    max_border_touch_ratio: float = 0.60


@dataclass
class RuleFilterConfig:
    min_solidity: float = 0.55
    min_eccentricity: float = 0.55
    min_aspect_ratio: float = 1.20
    min_color_score: float = 0.20
    min_local_contrast: float = 6.0
    max_blue_overlap_ratio: float = 0.15
    cluster_area_multiplier: float = 2.20
    # Cluster sanity caps. A "cluster" wider than ~30x the median pupa area
    # is almost certainly a paper stain, scan artifact, or edge band rather
    # than a pile of pupae — reject it as artifact instead of feeding a
    # bogus 8-pupa estimate to the cluster_fallback path.
    max_cluster_area_multiplier: float = 30.0
    # Same idea for image-relative size: anything taking more than 0.8% of
    # the page is far bigger than any plausible pupa cluster on a 300 DPI
    # scan and is almost always a stain or border noise.
    max_cluster_image_fraction: float = 0.008
    # A "cluster" hugging the image border is the top/bottom/left/right
    # noise band, not a real pile of pupae.
    cluster_max_border_touch_ratio: float = 0.40
    # A "cluster" with very high mean V (bright color) is a paper
    # stain — pupae are dark/brown, never near-white.
    cluster_max_mean_v: float = 215.0


@dataclass
class SplitClustersConfig:
    enabled: bool = True
    distance_peak_min_distance: int = 4
    peak_abs_threshold: float = 1.0
    watershed_min_child_area_px: int = 18
    max_children_per_cluster: int = 6


@dataclass
class ClusterFallbackConfig:
    enabled: bool = True
    use_for_counts: bool = True
    reference_confidence_min: float = 0.55
    min_area_ratio: float = 1.80
    min_major_axis_ratio: float = 1.30
    exclude_split_children: bool = True
    min_estimated_instances: int = 2
    max_estimated_instances: int = 8
    area_weight: float = 0.65
    major_axis_weight: float = 0.35


@dataclass
class VisionFallbackConfig:
    enabled: bool = False
    provider: str = "gemini"
    model: str = "gemini-3.1-pro-preview"
    api_base: str = "https://api.openai.com/v1/responses"
    timeout_s: int = 45
    max_side_px: int = 768
    padding_px: int = 16
    max_clusters_per_image: int = 4
    confidence_threshold: float = 0.60
    use_for_counts: bool = True


@dataclass
class CountingConfig:
    anchor_mode: str = "centroid"
    min_final_instances: int = 2
    min_span_px: float = 25.0
    min_instance_confidence: float = 0.65


@dataclass
class ReviewConfig:
    flag_low_anchor_confidence: bool = True
    low_anchor_confidence_threshold: float = 0.55
    flag_border_anchor: bool = True
    border_anchor_threshold: float = 0.20
    flag_unresolved_cluster: bool = True
    unresolved_cluster_min_count: int = 10
    unresolved_cluster_ratio_threshold: float = 0.10
    flag_high_blue_ratio_threshold: float = 0.02
    flag_blue_trust_disagreement_threshold: int = 5
    suspicious_color_low: float = 0.18
    flag_large_run_diff_threshold: int = 3
    previous_counts_csv: Optional[str] = None


@dataclass
class ClassifierConfig:
    enabled: bool = False
    model_path: Optional[str] = None
    model_type: str = "random_forest"
    probability_threshold: float = 0.60
    uncertain_low: float = 0.40
    uncertain_high: float = 0.60


@dataclass
class OutputConfig:
    save_intermediate_masks: bool = True
    save_candidate_table: bool = True
    save_overlays: bool = True
    save_review_queue: bool = True
    save_reports: bool = True


@dataclass
class DetectorConfig:
    """Selects which detection backend produces the candidate components.

    - ``classical`` (default): brown_mask + watershed split, the legacy path
    - ``cellpose``: learned instance segmentation via cellpose cyto3 model
    """
    backend: str = "classical"
    cellpose_diameter: Optional[float] = None  # None → cellpose auto-estimates
    cellpose_max_side_px: int = 1400
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    cellpose_overlap_split_enabled: bool = True
    cellpose_overlap_split_area_ratio: float = 1.70
    cellpose_overlap_split_max_aspect_ratio: float = 1.60
    cellpose_overlap_split_max_eccentricity: float = 0.80
    cellpose_overlap_split_annotated_ignore_shape: bool = True
    cellpose_overlap_split_peak_min_distance: int = 3
    cellpose_overlap_split_peak_abs_threshold: float = 1.0
    cellpose_overlap_split_min_child_area_ratio: float = 0.35
    cellpose_overlap_split_max_children: int = 4
    cellpose_overlap_split_combo_enabled: bool = True
    cellpose_overlap_split_combo_area_ratio: float = 1.25
    cellpose_overlap_split_combo_peak_min_distance: int = 3
    cellpose_overlap_split_combo_peak_abs_threshold: float = 0.35
    cellpose_overlap_split_combo_min_child_area_ratio: float = 0.22
    cellpose_overlap_split_combo_max_child_area_ratio: float = 1.45
    cellpose_overlap_split_combo_brown_weight: float = 0.55
    cellpose_overlap_split_combo_max_border_touch_ratio: float = 0.12
    cellpose_dense_patch_refine_enabled: bool = True
    cellpose_dense_patch_min_instances: int = 8
    cellpose_dense_patch_dbscan_eps_px: float = 48.0
    cellpose_dense_patch_dbscan_eps_scale: float = 1.15
    cellpose_dense_patch_padding_scale: float = 1.1
    cellpose_dense_patch_min_fill_ratio: float = 0.08
    cellpose_dense_patch_diameter_scale: float = 0.82
    cellpose_dense_patch_min_diameter_px: float = 18.0
    cellpose_dense_patch_flow_threshold: float = 0.35
    cellpose_dense_patch_cellprob_threshold: float = -0.1
    cellpose_dense_patch_min_area_ratio: float = 0.38
    cellpose_dense_patch_max_area_ratio: float = 1.90
    cellpose_dense_patch_min_extra_instances: int = 1
    cellpose_dense_patch_max_gain_ratio: float = 1.35
    cellpose_annotated_dual_path_enabled: bool = True
    cellpose_annotated_dual_path_min_instances: int = 6
    cellpose_annotated_dual_path_padding_scale: float = 1.0
    cellpose_annotated_dual_path_min_extra_instances: int = 1
    cellpose_annotated_dual_path_max_gain_ratio: float = 1.45
    cellpose_annotated_dual_path_max_blue_overlap_ratio: float = 0.18
    cellpose_annotated_dual_path_min_confidence: float = 0.52
    cellpose_annotated_dual_path_match_iou: float = 0.22
    cellpose_annotated_dual_path_match_distance_scale: float = 0.45
    cellpose_annotated_dual_path_diameter_scale: float = 0.82
    cellpose_annotated_dual_path_min_diameter_px: float = 18.0
    cellpose_annotated_dual_path_flow_threshold: float = 0.35
    cellpose_annotated_dual_path_cellprob_threshold: float = -0.10
    cellpose_annotated_pair_rescue_enabled: bool = True
    cellpose_annotated_pair_rescue_min_children: int = 2
    cellpose_annotated_pair_rescue_max_children: int = 3
    cellpose_annotated_pair_rescue_padding_scale: float = 0.55
    cellpose_annotated_pair_rescue_min_confidence: float = 0.45
    cellpose_annotated_pair_rescue_max_blue_overlap_ratio: float = 0.15
    cellpose_annotated_pair_rescue_min_area_ratio: float = 0.35
    cellpose_annotated_pair_rescue_max_area_ratio: float = 1.85
    cellpose_annotated_pair_rescue_max_gain_ratio: float = 1.80
    cellpose_annotated_png_supplement_enabled: bool = True
    cellpose_annotated_png_supplement_max_unmatched_ratio: float = 0.22
    cellpose_annotated_png_supplement_max_added_count: int = 14
    cellpose_annotated_png_supplement_min_area_px: float = 40.0
    cellpose_annotated_png_supplement_max_area_px: float = 260.0
    cellpose_annotated_png_supplement_max_mean_v: float = 185.0
    cellpose_annotated_png_supplement_min_color_score: float = 0.14
    cellpose_annotated_png_supplement_min_local_contrast: float = 3.5
    cellpose_annotated_png_supplement_max_blue_overlap_ratio: float = 0.10
    cellpose_annotated_png_supplement_max_border_touch_ratio: float = 0.35
    clean_filter_max_mean_v: float = 190.0
    clean_filter_max_color_score: float = 0.32
    clean_png_supplement_enabled: bool = True
    clean_png_supplement_max_cellpose_count: int = 60
    clean_png_supplement_max_unmatched_ratio: float = 0.16
    clean_png_supplement_min_area_px: float = 50.0
    clean_png_supplement_max_area_px: float = 130.0
    clean_png_supplement_max_mean_v: float = 130.0
    clean_png_supplement_min_color_score: float = 0.30
    clean_png_supplement_min_local_contrast: float = 10.0


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    input: InputConfig = field(default_factory=InputConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    blue_mask: BlueMaskConfig = field(default_factory=BlueMaskConfig)
    brown_detection: BrownDetectionConfig = field(default_factory=BrownDetectionConfig)
    components: ComponentsConfig = field(default_factory=ComponentsConfig)
    rule_filter: RuleFilterConfig = field(default_factory=RuleFilterConfig)
    split_clusters: SplitClustersConfig = field(default_factory=SplitClustersConfig)
    cluster_fallback: ClusterFallbackConfig = field(default_factory=ClusterFallbackConfig)
    vision_fallback: VisionFallbackConfig = field(default_factory=VisionFallbackConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _from_section(dataclass_type, payload: Optional[Dict[str, Any]]):
    payload = payload or {}
    return dataclass_type(**payload)


def load_config(path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if overrides:
        raw = _merge_dicts(raw, overrides)
    return AppConfig(
        project=_from_section(ProjectConfig, raw.get("project")),
        input=_from_section(InputConfig, raw.get("input")),
        preprocess=_from_section(PreprocessConfig, raw.get("preprocess")),
        blue_mask=_from_section(BlueMaskConfig, raw.get("blue_mask")),
        brown_detection=_from_section(BrownDetectionConfig, raw.get("brown_detection")),
        components=_from_section(ComponentsConfig, raw.get("components")),
        rule_filter=_from_section(RuleFilterConfig, raw.get("rule_filter")),
        split_clusters=_from_section(SplitClustersConfig, raw.get("split_clusters")),
        cluster_fallback=_from_section(ClusterFallbackConfig, raw.get("cluster_fallback")),
        vision_fallback=_from_section(VisionFallbackConfig, raw.get("vision_fallback")),
        counting=_from_section(CountingConfig, raw.get("counting")),
        review=_from_section(ReviewConfig, raw.get("review")),
        classifier=_from_section(ClassifierConfig, raw.get("classifier")),
        output=_from_section(OutputConfig, raw.get("output")),
        detector=_from_section(DetectorConfig, raw.get("detector")),
    )
