"""PaddleOCR-based text detection replacing Amazon Bedrock Data Automation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ocr_engine = None


def _get_ocr(use_gpu: bool = True):
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )
        logger.info("PaddleOCR engine initialized (use_gpu=%s)", use_gpu)
    return _ocr_engine


def _filter_text_by_region(
    text_elements: list[dict[str, Any]],
    manual_coordinates: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    crop_x = int(manual_coordinates["x"])
    crop_y = int(manual_coordinates["y"])
    crop_w = int(manual_coordinates["width"])
    crop_h = int(manual_coordinates["height"])
    crop_right = crop_x + crop_w
    crop_bottom = crop_y + crop_h

    kept: list[dict[str, Any]] = []
    filtered = 0
    for element in text_elements:
        bbox = element["bounding_box"]
        text_x = bbox["x"]
        text_y = bbox["y"]
        text_right = text_x + bbox["width"]
        text_bottom = text_y + bbox["height"]
        overlaps = (
            text_x < crop_right
            and text_right > crop_x
            and text_y < crop_bottom
            and text_bottom > crop_y
        )
        if overlaps:
            kept.append(element)
        else:
            filtered += 1

    stats = {
        "original_count": len(text_elements),
        "kept_count": len(kept),
        "filtered_count": filtered,
        "crop_region": {
            "x": crop_x,
            "y": crop_y,
            "width": crop_w,
            "height": crop_h,
        },
        "coordinates_in_original_space": True,
    }
    return kept, stats


def detect_text(
    image_path: Path,
    path_manager,
    notes_config: dict[str, Any] | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    """
    Run PaddleOCR on the original image and return BDA-compatible text detection results.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        width, height = img.size

    ocr = _get_ocr(use_gpu=use_gpu)
    raw = ocr.ocr(str(image_path), cls=True)
    text_elements: list[dict[str, Any]] = []

    if raw:
        for page in raw:
            if not page:
                continue
            for line in page:
                if not line or len(line) < 2:
                    continue
                box, text_info = line[0], line[1]
                if not text_info:
                    continue
                text, confidence = text_info[0], float(text_info[1])
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                x, y = min(xs), min(ys)
                text_elements.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bounding_box": {
                            "x": float(x),
                            "y": float(y),
                            "width": float(max(xs) - x),
                            "height": float(max(ys) - y),
                        },
                    }
                )

    filtering_stats: dict[str, Any] = {}
    manual_coordinates = (notes_config or {}).get("manual_coordinates", {})
    if manual_coordinates.get("width", 0) > 0 and manual_coordinates.get("height", 0) > 0:
        text_elements, filtering_stats = _filter_text_by_region(text_elements, manual_coordinates)

    results = {
        "text_elements": text_elements,
        "image_dimensions": {"width": width, "height": height},
        "ocr_engine": "paddleocr",
        "execution_id": path_manager.execution_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate_processing": {
            "coordinates_in_original_space": True,
            "manual_coordinates": manual_coordinates if manual_coordinates.get("width", 0) > 0 else None,
            "filtering_applied": bool(filtering_stats),
            "filtering_stats": filtering_stats,
        },
    }

    rel_key = path_manager.get_text_detection_results_s3_key()
    path_manager.write_json(rel_key, results)

    debug_path = path_manager.abs_path("text-detection", "debug_image.png")
    _save_debug_image(image_path, text_elements, debug_path)

    logger.info("PaddleOCR found %d text elements", len(text_elements))
    return results


def _save_debug_image(image_path: Path, text_elements: list[dict[str, Any]], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for element in text_elements:
        bbox = element["bounding_box"]
        x1 = int(bbox["x"])
        y1 = int(bbox["y"])
        x2 = int(bbox["x"] + bbox["width"])
        y2 = int(bbox["y"] + bbox["height"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            element["text"][:20],
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
