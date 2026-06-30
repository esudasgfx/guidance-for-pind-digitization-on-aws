"""Notes and frame preprocessing without S3."""

from __future__ import annotations

import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTES_PROCESSOR_DIR = REPO_ROOT / "cdk" / "lambda" / "notes_processor"
if str(NOTES_PROCESSOR_DIR) not in sys.path:
    sys.path.insert(0, str(NOTES_PROCESSOR_DIR))

from frame_detector import FrameDetector  # noqa: E402
from notes_section_detector import NotesSectionDetector  # noqa: E402

_frame_detector = FrameDetector()
_notes_detector = NotesSectionDetector()


def _get_image_dimensions(image_data: bytes) -> dict[str, int]:
    with Image.open(BytesIO(image_data)) as img:
        width, height = img.size
    return {"width": width, "height": height}


def _validate_coordinates(coords: dict[str, Any], image_dims: dict[str, int]) -> dict[str, Any]:
    required = ["x", "y", "width", "height"]
    missing = [k for k in required if k not in coords]
    if missing:
        return {"valid": False, "error": f"Missing required coordinate fields: {missing}"}
    try:
        x, y, w, h = int(coords["x"]), int(coords["y"]), int(coords["width"]), int(coords["height"])
    except (ValueError, TypeError) as exc:
        return {"valid": False, "error": f"Invalid coordinate values: {exc}"}
    img_w, img_h = image_dims["width"], image_dims["height"]
    if x < 0 or y < 0:
        return {"valid": False, "error": f"Coordinates must be non-negative: x={x}, y={y}"}
    if w > 0 and h > 0:
        if x >= img_w or y >= img_h or x + w > img_w or y + h > img_h:
            return {
                "valid": False,
                "error": f"Crop region exceeds image bounds: ({x},{y},{w},{h}) vs {img_w}x{img_h}",
            }
    return {"valid": True}


def _apply_manual_crop(image_data: bytes, coordinates: dict[str, Any]) -> bytes:
    image = Image.open(BytesIO(image_data))
    x, y = int(coordinates["x"]), int(coordinates["y"])
    w, h = int(coordinates["width"]), int(coordinates["height"])
    if w > 0 and h > 0:
        cropped = image.crop((x, y, x + w, y + h))
    else:
        cropped = image
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _calculate_processed_dimensions(
    original_dimensions: dict[str, int],
    manual_coordinates: dict[str, Any] | None,
    manual_processing_applied: bool,
    notes_coordinates: dict[str, Any] | None,
    frame_info: dict[str, Any],
) -> dict[str, int]:
    width = original_dimensions["width"]
    height = original_dimensions["height"]
    if frame_info.get("frame_removed") and "frame_bounds" in frame_info:
        bounds = frame_info["frame_bounds"]
        width -= bounds.get("left", 0) + bounds.get("right", 0)
        height -= bounds.get("top", 0) + bounds.get("bottom", 0)
    if manual_processing_applied and manual_coordinates:
        crop_w = int(manual_coordinates.get("width", 0))
        crop_h = int(manual_coordinates.get("height", 0))
        if crop_w > 0 and crop_h > 0:
            width, height = crop_w, crop_h
    elif notes_coordinates and not manual_processing_applied:
        notes_x = notes_coordinates.get("x", 0)
        if notes_x > 0:
            width = notes_x
    return {"width": width, "height": height}


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_make_json_serializable(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def process_notes(image_path: Path, config: dict[str, Any], path_manager) -> dict[str, Any]:
    """Run notes/frame preprocessing and write outputs to the local execution directory."""
    image_data = image_path.read_bytes()
    original_dimensions = _get_image_dimensions(image_data)
    current_image_data = image_data

    manual_coordinates = config.get("manual_coordinates", {})
    has_manual_coords = (
        manual_coordinates.get("width", 0) >= 0 and manual_coordinates.get("height", 0) >= 0
    )

    if has_manual_coords:
        remove_notes_section = False
        remove_frame = False
        processing_mode = "manual"
    else:
        remove_notes_section = config.get("remove_notes_section", True)
        frame_config = config.get("frame_config", {})
        remove_frame = frame_config.get("remove_frame", True)
        processing_mode = "auto"
        manual_coordinates = None
    frame_config = config.get("frame_config", {})

    frame_info: dict[str, Any] = {"frame_detected": False, "frame_removed": False}
    if remove_frame:
        current_image_data, frame_info = _frame_detector.detect_and_remove_frame(
            current_image_data, frame_config
        )

    layout_analysis = _notes_detector.analyze_image_layout(current_image_data)
    notes_coordinates = None
    coordinates_used = None
    manual_processing_applied = False

    if remove_notes_section and layout_analysis.get("notes_section_detected", False):
        notes_section = layout_analysis.get("notes_section", {})
        notes_coordinates = {
            "x": notes_section.get("x"),
            "y": notes_section.get("y"),
            "width": notes_section.get("width"),
            "height": notes_section.get("height"),
            "confidence": notes_section.get("confidence"),
            "method_used": notes_section.get("method_used"),
        }
        current_image_data = _notes_detector.remove_notes_section(current_image_data)

    if manual_coordinates:
        current_dimensions = _get_image_dimensions(current_image_data)
        validation = _validate_coordinates(manual_coordinates, current_dimensions)
        if validation["valid"]:
            current_image_data = _apply_manual_crop(current_image_data, manual_coordinates)
            manual_processing_applied = True
            coordinates_used = {
                **manual_coordinates,
                "method_used": "manual_coordinates",
                "validation_status": "valid",
            }
        else:
            processing_mode = "manual_failed"
            coordinates_used = {
                **manual_coordinates,
                "method_used": "manual_coordinates",
                "validation_status": "invalid",
                "error": validation["error"],
            }

    processed_key = path_manager.get_processed_image_s3_key()
    path_manager.write_bytes(processed_key, current_image_data)
    path_manager.write_bytes(path_manager.rel("input", image_path.name), image_data)

    processed_dimensions = _calculate_processed_dimensions(
        original_dimensions,
        manual_coordinates,
        manual_processing_applied,
        notes_coordinates,
        frame_info,
    )

    notes_metadata = {
        "processing_mode": processing_mode,
        "original_image_dimensions": original_dimensions,
        "notes_coordinates": notes_coordinates,
        "coordinates_used": coordinates_used,
        "layout_analysis": layout_analysis,
        "frame_info": frame_info,
        "manual_processing_applied": manual_processing_applied,
        "execution_id": path_manager.execution_id,
    }
    path_manager.write_json(path_manager.get_notes_metadata_s3_key(), notes_metadata)

    frame_adjusted_notes_coordinates = None
    if notes_coordinates and frame_info.get("frame_removed", False):
        frame_bounds = frame_info.get("frame_bounds", {})
        if frame_bounds:
            frame_adjusted_notes_coordinates = {
                "x": notes_coordinates["x"] + frame_bounds.get("left", 0),
                "y": notes_coordinates["y"] + frame_bounds.get("top", 0),
                "width": notes_coordinates["width"],
                "height": notes_coordinates["height"],
                "confidence": notes_coordinates["confidence"],
                "method_used": notes_coordinates["method_used"],
                "coordinate_space": "original_image",
            }

    result = {
        "statusCode": 200,
        "success": True,
        "processing_mode": processing_mode,
        "source_key": image_path.name,
        "processed_key": processed_key,
        "original_image_dimensions": original_dimensions,
        "processed_image_dimensions": processed_dimensions,
        "notes_coordinates": notes_coordinates,
        "frame_adjusted_notes_coordinates": frame_adjusted_notes_coordinates,
        "coordinates_used": coordinates_used,
        "frame_info": frame_info,
        "manual_processing_applied": manual_processing_applied,
        "execution_id": path_manager.execution_id,
    }
    logger.info("Notes processing complete (%s)", json.dumps(processed_dimensions))
    return _make_json_serializable(result)
