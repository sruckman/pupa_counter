"""End-to-end pipeline orchestration."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pupa_counter.annotate.blue_supervision import extract_blue_components, summarize_blue_supervision
from pupa_counter.config import AppConfig
from pupa_counter.count.anchors import compute_band_geometry
from pupa_counter.count.assign import assign_bands
from pupa_counter.count.summarize import combine_instances_for_counting, select_final_instances, summarize_counts
from pupa_counter.detect.brown_mask import detect_brown_candidates
from pupa_counter.detect.classifier import apply_optional_classifier, load_classifier
from pupa_counter.detect.cv_peak_deblend import (
    compute_fast_brown_score,
    refine_component_candidates as refine_cv_components,
    refine_labeled_candidates as refine_cv_labeled,
)
from pupa_counter.detect.cellpose_dual_path import merge_annotated_detection_paths, merge_annotated_pair_rescue
from pupa_counter.detect.cellpose_postprocess import (
    build_annotated_png_supplement,
    build_clean_png_supplement,
    calibrate_cellpose_detections,
    prune_annotated_false_positives,
)
from pupa_counter.detect.cellpose_dense_patch import refine_dense_cellpose_patches
from pupa_counter.detect.cellpose_split import split_large_cellpose_instances
from pupa_counter.detect.cluster_fallback import (
    apply_vision_cluster_counts,
    attach_cluster_count_estimates,
    synthesize_cluster_instances,
)
from pupa_counter.detect.components import extract_components
from pupa_counter.detect.features import featurize_components
from pupa_counter.detect.rule_filter import rule_classify_components
from pupa_counter.detect.split_clusters import split_cluster_candidates
from pupa_counter.eval.compare import compare_runs
from pupa_counter.eval.error_gallery import build_error_gallery
from pupa_counter.eval.metrics import evaluate_counts
from pupa_counter.io.discover import discover_inputs, manifest_dataframe
from pupa_counter.io.rasterize import rasterize_record
from pupa_counter.io.save import ensure_run_dirs, save_dataframe, save_image, save_json, save_mask, serializable_candidates
from pupa_counter.preprocess.blue_mask import detect_blue_annotations
from pupa_counter.preprocess.crop import crop_scanner_border
from pupa_counter.preprocess.inpaint import remove_or_ignore_blue
from pupa_counter.preprocess.normalize import build_reference_view, normalize_background
from pupa_counter.preprocess.paper_region import estimate_paper_bounds
from pupa_counter.report.html_report import build_run_report
from pupa_counter.report.overlay import build_overlay
from pupa_counter.report.review_queue import build_review_flags, build_review_queue_frame
from pupa_counter.report.worksheet import export_running_totals_workbook
from pupa_counter.vision.openai_cluster_counter import estimate_cluster_counts_with_openai


def _select_anchor_pool(final_instances: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    if final_instances.empty:
        return final_instances
    robust = final_instances.loc[
        (~final_instances["touches_image_border"].astype(bool))
        & (final_instances["anchor_confidence"].astype(float) >= cfg.review.low_anchor_confidence_threshold)
    ]
    if len(robust) >= cfg.counting.min_final_instances:
        return robust
    non_border = final_instances.loc[~final_instances["touches_image_border"].astype(bool)]
    if len(non_border) >= cfg.counting.min_final_instances:
        return non_border
    return final_instances


def _load_previous_lookup(path: Optional[str]) -> Dict[str, pd.Series]:
    if not path:
        return {}
    previous_path = Path(path)
    if not previous_path.exists():
        return {}
    frame = pd.read_csv(previous_path)
    if "image_id" not in frame.columns:
        return {}
    return {row["image_id"]: row for _, row in frame.iterrows()}


def _write_partial_outputs(
    run_root: Path,
    summaries,
    flags_by_image,
    all_candidates,
    all_blue_components,
    blue_supervision_rows,
    all_vision_cluster_rows,
    run_dirs,
):
    counts_df = pd.DataFrame([summary.to_row() for summary in summaries])
    save_dataframe(run_root / "counts.partial.csv", counts_df)

    candidate_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    save_dataframe(run_root / "candidate_table.partial.csv", candidate_df)

    blue_component_df = pd.concat(all_blue_components, ignore_index=True) if all_blue_components else pd.DataFrame()
    save_dataframe(run_root / "blue_component_table.partial.csv", blue_component_df)

    blue_supervision_df = pd.DataFrame(blue_supervision_rows)
    save_dataframe(run_root / "blue_supervision.partial.csv", blue_supervision_df)

    vision_cluster_df = pd.concat(all_vision_cluster_rows, ignore_index=True) if all_vision_cluster_rows else pd.DataFrame()
    save_dataframe(run_root / "vision_cluster_counts.partial.csv", vision_cluster_df)

    review_df = build_review_queue_frame(summaries, flags_by_image, str(run_dirs["overlays"]))
    save_dataframe(run_root / "review_queue.partial.csv", review_df)

    payload = {
        "images_completed": len(counts_df),
        "review_rows": len(review_df),
        "candidate_rows": len(candidate_df),
        "vision_rows": len(vision_cluster_df),
        "latest_image_id": None if counts_df.empty else str(counts_df.iloc[-1]["image_id"]),
    }
    save_json(run_root / "progress.json", payload)


def run_pipeline(
    input_root: Path,
    cfg: AppConfig,
    output_root: Path,
    gold_csv: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    random.seed(cfg.project.random_seed)
    np.random.seed(cfg.project.random_seed)

    run_root = output_root / cfg.project.config_version
    run_dirs = ensure_run_dirs(run_root)

    records = discover_inputs(input_root, cfg)
    manifest_df = manifest_dataframe(records)
    save_dataframe(run_root / "manifest.csv", manifest_df)

    processable = manifest_df.loc[manifest_df["source_type"].isin(["annotated_png", "clean_pdf", "clean_png"])]
    if limit is not None:
        processable = processable.head(limit)

    classifier = load_classifier(cfg.classifier.model_path) if cfg.classifier.enabled else None
    previous_lookup = _load_previous_lookup(cfg.review.previous_counts_csv)

    all_candidates = []
    all_blue_components = []
    blue_supervision_rows = []
    all_vision_cluster_rows = []
    summaries = []
    flags_by_image = {}

    for _, record_row in processable.iterrows():
        record = next(item for item in records if item.image_id == record_row["image_id"])
        start = time.perf_counter()

        raster = rasterize_record(record, dpi=cfg.input.raster_dpi)
        cropped = crop_scanner_border(raster, cfg)
        reference_view = build_reference_view(cropped, cfg)
        normalized = normalize_background(cropped, cfg)
        blue_mask = detect_blue_annotations(normalized, cfg)
        blue_components_df = extract_blue_components(blue_mask, normalized.shape)
        blue_supervision = summarize_blue_supervision(blue_components_df, blue_mask, normalized.shape)
        paper_bounds = estimate_paper_bounds(cropped, blue_mask=blue_mask, cfg=cfg)
        cleaned = remove_or_ignore_blue(normalized, blue_mask, cfg)
        reference_cleaned = remove_or_ignore_blue(reference_view, blue_mask, cfg)

        if cfg.detector.backend == "cellpose":
            from pupa_counter.detect.cellpose_backend import detect_instances as cellpose_detect

            def _run_annotated_alt_path(alt_image, *, feature_image, component_prefix: str):
                alt_components_df = cellpose_detect(
                    alt_image,
                    cfg,
                    diameter=dual_path_diameter,
                    flow_threshold=cfg.detector.cellpose_annotated_dual_path_flow_threshold,
                    cellprob_threshold=cfg.detector.cellpose_annotated_dual_path_cellprob_threshold,
                    component_prefix=component_prefix,
                )
                if alt_components_df.empty:
                    return alt_components_df.copy()
                alt_components_df = split_large_cellpose_instances(
                    alt_components_df,
                    alt_image.shape[:2],
                    source_type=record.source_type,
                    cfg=cfg,
                )
                alt_features_df = featurize_components(feature_image, blue_mask, alt_components_df)
                return calibrate_cellpose_detections(
                    alt_features_df,
                    source_type=record.source_type,
                    cfg=cfg,
                )

            brown_mask = np.zeros(cleaned.shape[:2], dtype=np.uint8)
            if record.source_type == "annotated_png":
                primary_detection_image = reference_cleaned
                primary_feature_image = reference_cleaned
            else:
                primary_detection_image = cleaned
                primary_feature_image = cleaned
            dense_refine_image = reference_cleaned if record.source_type == "annotated_png" else cleaned
            components_df = cellpose_detect(primary_detection_image, cfg)
            if not components_df.empty:
                components_df = split_large_cellpose_instances(
                    components_df,
                    primary_detection_image.shape[:2],
                    source_type=record.source_type,
                    cfg=cfg,
                    guide_image=primary_feature_image,
                )
                components_df = refine_dense_cellpose_patches(
                    dense_refine_image,
                    components_df,
                    source_type=record.source_type,
                    cfg=cfg,
                )
                components_df = split_large_cellpose_instances(
                    components_df,
                    primary_detection_image.shape[:2],
                    source_type=record.source_type,
                    cfg=cfg,
                    guide_image=dense_refine_image,
                    restrict_to_dense_patch=True,
                )
            features_df = (
                featurize_components(primary_feature_image, blue_mask, components_df)
                if not components_df.empty
                else components_df.copy()
            )
            labeled_df = calibrate_cellpose_detections(features_df, source_type=record.source_type, cfg=cfg) if not features_df.empty else features_df.copy()

            if record.source_type == "annotated_png" and cfg.detector.cellpose_annotated_dual_path_enabled:
                dual_path_diameter = None
                if not components_df.empty:
                    dual_path_diameter = max(
                        cfg.detector.cellpose_annotated_dual_path_min_diameter_px,
                        float(np.median(components_df["major_axis_px"].astype(float)))
                        * cfg.detector.cellpose_annotated_dual_path_diameter_scale,
                    )
                normalized_labeled_df = _run_annotated_alt_path(
                    cleaned,
                    feature_image=cleaned,
                    component_prefix="npn",
                )
                if not normalized_labeled_df.empty:
                    labeled_df = merge_annotated_detection_paths(
                        labeled_df,
                        normalized_labeled_df,
                        image_shape=cleaned.shape[:2],
                        cfg=cfg,
                    )
            if record.source_type == "annotated_png" and cfg.detector.cellpose_annotated_pair_rescue_enabled:
                brown_mask = detect_brown_candidates(reference_view, blue_mask=blue_mask, cfg=cfg)
                classical_components_df = extract_components(brown_mask, cfg)
                classical_features_df = (
                    featurize_components(reference_view, blue_mask, classical_components_df)
                    if not classical_components_df.empty
                    else classical_components_df.copy()
                )
                classical_labeled_df = (
                    rule_classify_components(classical_features_df, cfg)
                    if not classical_features_df.empty
                    else classical_features_df.copy()
                )
                classical_split_df = (
                    split_cluster_candidates(reference_view, classical_labeled_df, blue_mask=blue_mask, cfg=cfg)
                    if not classical_labeled_df.empty
                    else classical_labeled_df.copy()
                )
                labeled_df = merge_annotated_pair_rescue(
                    labeled_df,
                    classical_split_df,
                    image_shape=reference_view.shape[:2],
                    cfg=cfg,
                    paper_bounds=paper_bounds,
                )
                supplement_df = build_annotated_png_supplement(
                    labeled_df,
                    classical_split_df,
                    source_type=record.source_type,
                    cfg=cfg,
                    paper_bounds=paper_bounds,
                )
                if not supplement_df.empty:
                    labeled_df = pd.concat([labeled_df, supplement_df], ignore_index=True)
                labeled_df = prune_annotated_false_positives(
                    labeled_df,
                    source_type=record.source_type,
                    cfg=cfg,
                    paper_bounds=paper_bounds,
                )

            if record.source_type == "clean_png" and cfg.detector.clean_png_supplement_enabled:
                brown_mask = detect_brown_candidates(cleaned, blue_mask=blue_mask, cfg=cfg)
                classical_components_df = extract_components(brown_mask, cfg)
                classical_features_df = (
                    featurize_components(cleaned, blue_mask, classical_components_df)
                    if not classical_components_df.empty
                    else classical_components_df.copy()
                )
                classical_labeled_df = (
                    rule_classify_components(classical_features_df, cfg)
                    if not classical_features_df.empty
                    else classical_features_df.copy()
                )
                supplement_df = build_clean_png_supplement(
                    labeled_df,
                    classical_labeled_df,
                    source_type=record.source_type,
                    cfg=cfg,
                )
                if not supplement_df.empty:
                    labeled_df = pd.concat([labeled_df, supplement_df], ignore_index=True)
            split_df = labeled_df
        elif cfg.detector.backend == "cv_peak_deblend":
            detection_image = reference_view if record.source_type == "annotated_png" else cleaned
            feature_image = cleaned
            brown_mask = detect_brown_candidates(feature_image, blue_mask=blue_mask, cfg=cfg)
            base_components_df = extract_components(brown_mask, cfg)
            if not base_components_df.empty:
                cv_components_df = refine_cv_components(
                    base_components_df,
                    compute_fast_brown_score(detection_image),
                    brown_mask > 0,
                    cfg,
                    blue_mask=blue_mask,
                    paper_bounds=paper_bounds,
                    component_prefix="cv",
                )
                cv_features_df = featurize_components(feature_image, blue_mask, cv_components_df) if not cv_components_df.empty else cv_components_df.copy()
                cv_labeled_df = rule_classify_components(cv_features_df, cfg) if not cv_features_df.empty else cv_features_df.copy()
                cv_labeled_df = refine_cv_labeled(
                    cv_labeled_df,
                    score_image=compute_fast_brown_score(detection_image),
                    foreground_mask=brown_mask > 0,
                    feature_image=feature_image,
                    blue_mask=blue_mask,
                    paper_bounds=paper_bounds,
                    cfg=cfg,
                ) if not cv_labeled_df.empty else cv_labeled_df.copy()
                split_df = (
                    split_cluster_candidates(feature_image, cv_labeled_df, blue_mask=blue_mask, cfg=cfg)
                    if not cv_labeled_df.empty
                    else cv_labeled_df.copy()
                )
            else:
                split_df = base_components_df.copy()
        elif cfg.detector.backend == "classical":
            brown_mask = detect_brown_candidates(cleaned, blue_mask=blue_mask, cfg=cfg)
            components_df = extract_components(brown_mask, cfg)
            features_df = featurize_components(cleaned, blue_mask, components_df) if not components_df.empty else components_df.copy()
            labeled_df = rule_classify_components(features_df, cfg) if not features_df.empty else features_df.copy()
            split_df = split_cluster_candidates(cleaned, labeled_df, blue_mask=blue_mask, cfg=cfg) if not labeled_df.empty else labeled_df.copy()
        else:
            raise ValueError(f"Unknown detector backend: {cfg.detector.backend}")
        classified_df = apply_optional_classifier(split_df, classifier=classifier, cfg=cfg) if not split_df.empty else split_df.copy()
        classified_df = attach_cluster_count_estimates(classified_df, cfg)
        vision_cluster_df = estimate_cluster_counts_with_openai(cleaned, classified_df, cfg)
        classified_df = apply_vision_cluster_counts(classified_df, vision_cluster_df, cfg)
        final_instances = select_final_instances(classified_df, cfg)
        synthetic_instances = synthesize_cluster_instances(classified_df, cfg)

        geometry = None
        final_instances = final_instances.copy()
        if not final_instances.empty:
            final_instances["anchor_role"] = ""
        if len(final_instances) >= 1:
            try:
                anchor_pool = _select_anchor_pool(final_instances, cfg)
                geometry = compute_band_geometry(anchor_pool, anchor_mode=cfg.counting.anchor_mode)
                if not anchor_pool.empty:
                    top_component_id = anchor_pool.loc[anchor_pool["centroid_y"].idxmin(), "component_id"]
                    bottom_component_id = anchor_pool.loc[anchor_pool["centroid_y"].idxmax(), "component_id"]
                    final_instances.loc[final_instances["component_id"] == top_component_id, "anchor_role"] = "top"
                    final_instances.loc[final_instances["component_id"] == bottom_component_id, "anchor_role"] = "bottom"
            except ValueError:
                geometry = None
        if geometry is not None:
            final_instances = assign_bands(final_instances, geometry)
        else:
            final_instances = final_instances.copy()
            if not final_instances.empty:
                final_instances["band"] = "middle"
        if not synthetic_instances.empty:
            synthetic_instances = synthetic_instances.copy()
            if geometry is not None:
                synthetic_instances = assign_bands(synthetic_instances, geometry)
            else:
                synthetic_instances["band"] = "middle"
        count_instances = combine_instances_for_counting(final_instances, synthetic_instances)

        runtime_ms = (time.perf_counter() - start) * 1000.0
        blue_pixel_ratio = float((blue_mask > 0).mean()) if blue_mask is not None and blue_mask.size else 0.0
        model_version = None if classifier is None else classifier.__class__.__name__
        summary = summarize_counts(
            record,
            count_instances,
            geometry,
            config_version=cfg.project.config_version,
            model_version=model_version,
            runtime_ms=runtime_ms,
            candidate_df=classified_df,
            blue_pixel_ratio=blue_pixel_ratio,
        )
        summary.extra["detected_pupa_instances"] = int(len(final_instances))
        summary.extra["synthetic_cluster_instances"] = int(len(synthetic_instances))
        summary.extra["vision_cluster_count_used"] = int(
            classified_df.loc[classified_df["cluster_count_source"] == "vision", "estimated_cluster_count"].sum()
        ) if not classified_df.empty and "cluster_count_source" in classified_df.columns else 0
        summary.extra.update(blue_supervision)
        if geometry is not None and blue_supervision.get("trusted_line_upper_y") is not None:
            summary.extra["trusted_upper_line_error_px"] = abs(
                geometry.upper_middle_y - float(blue_supervision["trusted_line_upper_y"])
            )
        else:
            summary.extra["trusted_upper_line_error_px"] = None
        if geometry is not None and blue_supervision.get("trusted_line_lower_y") is not None:
            summary.extra["trusted_lower_line_error_px"] = abs(
                geometry.lower_middle_y - float(blue_supervision["trusted_line_lower_y"])
            )
        else:
            summary.extra["trusted_lower_line_error_px"] = None
        if blue_supervision.get("trusted_dot_middle") is not None:
            summary.extra["trusted_middle_disagreement"] = abs(summary.n_middle - int(blue_supervision["trusted_dot_middle"]))
        else:
            summary.extra["trusted_middle_disagreement"] = None
        if blue_supervision.get("trusted_dot_total") is not None:
            summary.extra["trusted_total_disagreement"] = abs(summary.n_pupa_final - int(blue_supervision["trusted_dot_total"]))
        else:
            summary.extra["trusted_total_disagreement"] = None
        previous_row = previous_lookup.get(record.image_id)
        flags = build_review_flags(summary, count_instances, candidate_df=classified_df, previous_row=previous_row, cfg=cfg)
        flags_by_image[record.image_id] = flags
        summaries.append(summary)
        blue_supervision_rows.append({"image_id": record.image_id, "source_path": str(record.source_path), **blue_supervision})

        if not blue_components_df.empty:
            blue_export = blue_components_df.copy()
            blue_export.insert(0, "image_id", record.image_id)
            all_blue_components.append(blue_export)
        if vision_cluster_df is not None and not vision_cluster_df.empty:
            vision_export = vision_cluster_df.copy()
            vision_export.insert(0, "image_id", record.image_id)
            all_vision_cluster_rows.append(vision_export)

        if cfg.output.save_intermediate_masks:
            save_image(run_dirs["intermediate"] / ("%s_stage0.png" % record.image_id), normalized)
            save_image(run_dirs["intermediate"] / ("%s_original_stage0.png" % record.image_id), cropped)
            save_image(run_dirs["intermediate"] / ("%s_reference_stage0.png" % record.image_id), reference_view)
            save_image(run_dirs["intermediate"] / ("%s_normalized_stage0.png" % record.image_id), normalized)
            save_mask(run_dirs["intermediate"] / ("%s_blue_mask.png" % record.image_id), blue_mask)
            save_image(
                run_dirs["intermediate"] / ("%s_clean_stage1.png" % record.image_id),
                reference_cleaned if record.source_type == "annotated_png" else cleaned,
            )
            if record.source_type == "annotated_png":
                save_image(run_dirs["intermediate"] / ("%s_normalized_clean_stage1.png" % record.image_id), cleaned)
            save_mask(run_dirs["intermediate"] / ("%s_brown_mask.png" % record.image_id), brown_mask)

        if cfg.output.save_overlays:
            overlay_base = reference_view if record.source_type == "annotated_png" else normalized
            overlay = build_overlay(
                overlay_base,
                count_instances,
                geometry,
                flags=flags,
                candidate_df=classified_df,
                show_middle_labels=cfg.output.overlay_show_middle_labels,
                show_unresolved_clusters=cfg.output.overlay_show_unresolved_clusters,
            )
            save_image(run_dirs["overlays"] / ("%s.png" % record.image_id), overlay)

        if cfg.output.save_candidate_table and not classified_df.empty:
            candidate_export = serializable_candidates(classified_df)
            candidate_export.insert(0, "image_id", record.image_id)
            all_candidates.append(candidate_export)

        _write_partial_outputs(
            run_root=run_root,
            summaries=summaries,
            flags_by_image=flags_by_image,
            all_candidates=all_candidates,
            all_blue_components=all_blue_components,
            blue_supervision_rows=blue_supervision_rows,
            all_vision_cluster_rows=all_vision_cluster_rows,
            run_dirs=run_dirs,
        )

    counts_df = pd.DataFrame([summary.to_row() for summary in summaries])
    save_dataframe(run_root / "counts.csv", counts_df)

    candidate_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    if cfg.output.save_candidate_table:
        save_dataframe(run_root / "candidate_table.csv", candidate_df)

    blue_component_df = pd.concat(all_blue_components, ignore_index=True) if all_blue_components else pd.DataFrame()
    save_dataframe(run_root / "blue_component_table.csv", blue_component_df)
    blue_supervision_df = pd.DataFrame(blue_supervision_rows)
    save_dataframe(run_root / "blue_supervision.csv", blue_supervision_df)
    vision_cluster_df = pd.concat(all_vision_cluster_rows, ignore_index=True) if all_vision_cluster_rows else pd.DataFrame()
    save_dataframe(run_root / "vision_cluster_counts.csv", vision_cluster_df)

    review_df = build_review_queue_frame(summaries, flags_by_image, str(run_dirs["overlays"]))
    if cfg.output.save_review_queue:
        save_dataframe(run_root / "review_queue.csv", review_df)
    workbook_path = None
    if cfg.output.save_running_totals_workbook:
        workbook_path = export_running_totals_workbook(run_root, counts_df, review_df)

    metrics = {}
    if gold_csv is not None and Path(gold_csv).exists():
        gold_df = pd.read_csv(gold_csv)
        metrics = evaluate_counts(counts_df, gold_df)
        merged = counts_df.merge(gold_df, on="image_id", suffixes=("_pred", "_gold"))
        if not merged.empty:
            merged["abs_error_middle"] = (
                merged["n_middle_pred"] - merged["true_middle"]
                if "true_middle" in merged.columns
                else merged["n_middle_pred"] - merged["n_middle_gold"]
            ).abs()
            build_error_gallery(merged, run_dirs["reports"] / "error_gallery.md")

    if cfg.output.save_reports:
        build_run_report(counts_df, review_df, run_dirs["reports"], metrics=metrics)

    comparison_df = pd.DataFrame()
    if cfg.review.previous_counts_csv and Path(cfg.review.previous_counts_csv).exists():
        comparison_df = compare_runs(counts_df, pd.read_csv(cfg.review.previous_counts_csv))
        if not comparison_df.empty:
            save_dataframe(run_root / "compare_previous.csv", comparison_df)

    return {
        "run_root": run_root,
        "manifest_df": manifest_df,
        "counts_df": counts_df,
        "candidate_df": candidate_df,
        "blue_component_df": blue_component_df,
        "blue_supervision_df": blue_supervision_df,
        "vision_cluster_df": vision_cluster_df,
        "review_df": review_df,
        "metrics": metrics,
        "comparison_df": comparison_df,
        "running_totals_workbook": workbook_path,
    }
