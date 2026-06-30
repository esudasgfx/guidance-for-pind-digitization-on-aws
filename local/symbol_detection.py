"""Local symbol detection using the pre-trained Faster R-CNN + Siamese models."""

from __future__ import annotations

import base64
import json
import logging
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
SHARED_DIR = REPO_ROOT / "cdk" / "lambda" / "shared"

for path in (INFERENCE_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_models = None


def _load_models(model_dir: Path):
    global _models
    if _models is None:
        import inference as inference_module

        _models = inference_module.model_fn(str(model_dir))
        logger.info("Symbol detection models loaded from %s", model_dir)
    return _models


def detect_symbols(
    image_bytes: bytes,
    model_dir: Path,
    path_manager,
    config: dict[str, Any],
    notes_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run tiled symbol detection and optionally transform coordinates to original space."""
    import inference as inference_module
    from coordinate_transform import (
        get_transformation_metadata_from_notes_processor,
        transform_coordinates_to_original,
        validate_transformation_result,
    )

    models = _load_models(model_dir)
    score_threshold = config.get("confidence_threshold", config.get("score_threshold", 0.9))
    n_closest = config.get("n_closest", 3)

    payload = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "score_threshold": score_threshold,
        "n_closest": n_closest,
    }
    input_data = inference_module.input_fn(json.dumps(payload))
    prediction = inference_module.predict_fn(input_data, models)
    detection_result = json.loads(inference_module.output_fn(prediction))

    if "detections" in detection_result and "num_detections" not in detection_result:
        detection_result["num_detections"] = len(detection_result["detections"])

    filter_classes = config.get("filter_classes")
    if filter_classes:
        detection_result["detections"] = [
            d for d in detection_result["detections"] if d.get("class_id") in filter_classes
        ]
        detection_result["num_detections"] = len(detection_result["detections"])

    if notes_result:
        try:
            transformation_metadata = get_transformation_metadata_from_notes_processor(
                {"Payload": notes_result}
            )
            original_detections = transform_coordinates_to_original(
                coordinates=detection_result.get("detections", []),
                transformation_metadata=transformation_metadata,
                coordinate_type="bounding_box",
            )
            validation = validate_transformation_result(
                original_coords=detection_result.get("detections", []),
                transformed_coords=original_detections,
                transformation_metadata=transformation_metadata,
            )
            detection_result["detections"] = original_detections
            detection_result["coordinate_transformation"] = {
                "applied": True,
                "validation": validation,
            }
        except Exception as exc:
            logger.warning("Coordinate transformation failed: %s", exc)
            detection_result["coordinate_transformation"] = {"applied": False, "error": str(exc)}
    else:
        detection_result["coordinate_transformation"] = {"applied": False}

    detection_result["timestamp"] = datetime.now(timezone.utc).isoformat()
    rel_key = path_manager.get_symbol_results_s3_key()
    path_manager.write_json(rel_key, detection_result)
    logger.info("Detected %d symbols", detection_result.get("num_detections", 0))
    return detection_result
