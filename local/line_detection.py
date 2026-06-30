"""Line detection orchestration without S3."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LINE_DETECTION_DIR = REPO_ROOT / "cdk" / "lambda" / "line_detection"
SHARED_DIR = REPO_ROOT / "cdk" / "lambda" / "shared"

for path in (LINE_DETECTION_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from line_detection import (  # noqa: E402
    BoundingBox,
    LineSegment,
    apply_thinning,
    clear_bounding_boxes,
    detect_line_segments,
    to_binary,
    to_grayscale,
)
from line_postprocessing import LinePostProcessor  # noqa: E402
from symbol_aware_line_detection import SymbolAwareLinePostProcessor  # noqa: E402
from coordinate_transform import (  # noqa: E402
    get_transformation_metadata_from_notes_processor,
    transform_coordinates_to_original,
    transform_coordinates_to_processed,
    validate_transformation_result,
)
from image_processing import convert_bounding_boxes, convert_bda_text_elements  # noqa: E402


def _load_image_cv2(image_path: Path) -> np.ndarray:
    pil_image = Image.open(image_path)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _extract_line_config(line_config: dict[str, Any]) -> dict[str, Any]:
    postprocess = line_config.get("postprocess_params", {})
    return {
        "max_line_gap": line_config.get("max_line_gap", 120),
        "threshold": line_config.get("threshold", 100),
        "min_line_length": line_config.get("min_line_length", 30),
        "rho": line_config.get("rho", 1.0),
        "theta_param": line_config.get("theta_param", 180),
        "enable_thinning": line_config.get("enable_thinning", True),
        "merge_distance_threshold": postprocess.get("merge_distance_threshold", 0.05),
        "angular_tolerance": postprocess.get("angular_tolerance", 15.0),
        "min_processed_line_length": postprocess.get("min_line_length", 20.0),
        "extension_padding": postprocess.get("extension_padding", 0.02),
        "enable_symbol_intersection": postprocess.get("enable_symbol_intersection", True),
    }


def _transform_boxes_to_processed(
    symbol_boxes: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    notes_result: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not notes_result:
        return symbol_boxes, text_boxes
    metadata = get_transformation_metadata_from_notes_processor({"Payload": notes_result})
    if symbol_boxes:
        symbol_boxes = transform_coordinates_to_processed(
            symbol_boxes, metadata, coordinate_type="bounding_box"
        )
    if text_boxes:
        text_boxes = transform_coordinates_to_processed(
            text_boxes, metadata, coordinate_type="bounding_box"
        )
    return symbol_boxes, text_boxes


def _preprocess_image(
    image: np.ndarray,
    symbol_boxes: list[BoundingBox],
    text_boxes: list[BoundingBox],
    config_params: dict[str, Any],
) -> np.ndarray:
    working = image.copy()
    if symbol_boxes:
        working = clear_bounding_boxes(working, symbol_boxes)
    if text_boxes:
        working = clear_bounding_boxes(working, text_boxes)
    grayscale = to_grayscale(working)
    binary = to_binary(grayscale)
    if config_params["enable_thinning"]:
        return apply_thinning(binary)
    return binary


def _postprocess_lines(
    line_segments: list[LineSegment],
    symbol_boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
    config_params: dict[str, Any],
) -> tuple[list[LineSegment], list[dict[str, Any]]]:
    base = LinePostProcessor(
        merge_distance_threshold=config_params["merge_distance_threshold"],
        angular_tolerance=config_params["angular_tolerance"],
        min_line_length=config_params["min_processed_line_length"],
        extension_padding=config_params["extension_padding"],
    )
    symbol_processor = SymbolAwareLinePostProcessor(symbol_boxes=symbol_boxes, base_processor=base)

    extended_pixel = base._denormalize_lines(line_segments, image_width, image_height)
    extended_pixel = base._extend_lines(extended_pixel)
    merged_pixel = base._merge_connected_lines(extended_pixel)
    filtered_pixel = base._filter_short_lines(merged_pixel, image_width, image_height)
    base_processed = base._normalize_lines(filtered_pixel, image_width, image_height)

    if config_params["enable_symbol_intersection"] and symbol_boxes:
        return symbol_processor.process_lines(base_processed, image_width, image_height)
    return base_processed, []


def detect_lines(
    processed_image_path: Path,
    text_results: dict[str, Any],
    symbol_results: dict[str, Any],
    notes_result: dict[str, Any] | None,
    line_config: dict[str, Any],
    path_manager,
) -> dict[str, Any]:
    """Detect line segments on the processed image and write results locally."""
    image = _load_image_cv2(processed_image_path)
    image_height, image_width = image.shape[:2]
    config_params = _extract_line_config(line_config)

    symbol_boxes_raw = symbol_results.get("detections", [])
    text_boxes_raw = text_results.get("text_elements", [])
    symbol_boxes_raw, text_boxes_raw = _transform_boxes_to_processed(
        symbol_boxes_raw, text_boxes_raw, notes_result
    )

    symbol_boxes = convert_bounding_boxes(symbol_boxes_raw, image_width, image_height)
    text_boxes = convert_bda_text_elements(text_boxes_raw, image_width, image_height)
    processed = _preprocess_image(image, symbol_boxes, text_boxes, config_params)

    line_segments = detect_line_segments(
        processed,
        image_height=image_height,
        image_width=image_width,
        bounding_box_inclusive=None,
        max_line_gap=config_params["max_line_gap"],
        threshold=config_params["threshold"],
        min_line_length=config_params["min_line_length"],
        rho=config_params["rho"],
        theta_param=config_params["theta_param"],
    )
    processed_segments, intersections = _postprocess_lines(
        line_segments, symbol_boxes, image_width, image_height, config_params
    )

    lines_json = [
        {"startX": line.startX, "startY": line.startY, "endX": line.endX, "endY": line.endY}
        for line in processed_segments
    ]

    processing_metadata: dict[str, Any]
    if notes_result:
        try:
            metadata = get_transformation_metadata_from_notes_processor({"Payload": notes_result})
            original_lines = transform_coordinates_to_original(
                lines_json, metadata, coordinate_type="line"
            )
            validation = validate_transformation_result(
                original_coords=lines_json,
                transformed_coords=original_lines,
                transformation_metadata=metadata,
            )
            lines_json = original_lines
            processing_metadata = {
                "coordinate_transformation": {"applied": True, "validation": validation},
            }
        except Exception as exc:
            logger.warning("Line coordinate transform failed: %s", exc)
            processing_metadata = {"coordinate_transformation": {"applied": False, "error": str(exc)}}
    else:
        processing_metadata = {"coordinate_transformation": {"applied": False}}

    base = path_manager.get_line_detection_path()
    lines_key = f"{base}/detected_lines.json"
    path_manager.write_json(
        lines_key,
        {
            "execution_id": path_manager.execution_id,
            "detected_lines": lines_json,
            "line_count": len(lines_json),
        },
    )
    path_manager.write_json(
        f"{base}/symbol_intersections.json",
        {"symbol_intersections": intersections, "intersection_count": len(intersections)},
    )
    path_manager.write_json(
        f"{base}/processing_metadata.json",
        {"processing_metadata": processing_metadata},
    )

    _save_debug_image(
        image,
        processed_segments,
        symbol_boxes,
        text_boxes,
        path_manager.abs_path("line-detection", "debug_image.png"),
    )

    logger.info("Detected %d line segments", len(lines_json))
    return {
        "statusCode": 200,
        "line_count": len(lines_json),
        "s3_results": {
            "lines_s3_key": lines_key,
            "bucket": str(path_manager.output_dir),
            "execution_id": path_manager.execution_id,
        },
        "detected_lines": lines_json,
        "processing_metadata": processing_metadata,
    }


def _save_debug_image(
    image: np.ndarray,
    line_segments: list[LineSegment],
    symbol_boxes: list[BoundingBox],
    text_boxes: list[BoundingBox],
    output_path: Path,
) -> None:
    debug = image.copy()
    height, width = debug.shape[:2]
    for box in symbol_boxes:
        cv2.rectangle(
            debug,
            (int(box.topX), int(box.topY)),
            (int(box.bottomX), int(box.bottomY)),
            (0, 0, 255),
            2,
        )
    for box in text_boxes:
        cv2.rectangle(
            debug,
            (int(box.topX), int(box.topY)),
            (int(box.bottomX), int(box.bottomY)),
            (255, 0, 0),
            2,
        )
    for line in line_segments:
        x1 = int(line.startX * width)
        y1 = int(line.startY * height)
        x2 = int(line.endX * width)
        y2 = int(line.endY * height)
        cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug)
