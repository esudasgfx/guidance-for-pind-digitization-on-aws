"""Graph generation and DEXPI export without S3."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_GENERATOR_DIR = REPO_ROOT / "cdk" / "lambda" / "graph_generator"

if str(GRAPH_GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_GENERATOR_DIR))

from dexpi_converter import convert_to_dexpi  # noqa: E402
from document_processor import DocumentProcessor  # noqa: E402


def _process_text_elements(text_results: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text_elements: list[dict[str, Any]] = []
    original_text_elements: list[dict[str, Any]] = []
    for idx, element in enumerate(text_results.get("text_elements", [])):
        bbox = element.get("bounding_box", {})
        if not bbox or "text" not in element:
            continue
        text_elements.append(
            {
                "id": f"text-{idx}",
                "text": element["text"],
                "bbox": [
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("x", 0) + bbox.get("width", 0),
                    bbox.get("y", 0) + bbox.get("height", 0),
                ],
                "confidence": element.get("confidence", 0.0),
            }
        )
        original_text_elements.append(
            {
                "id": f"text-{idx}",
                "text": element["text"],
                "original_bbox": {
                    "x": bbox.get("x", 0),
                    "y": bbox.get("y", 0),
                    "width": bbox.get("width", 0),
                    "height": bbox.get("height", 0),
                },
                "confidence": element.get("confidence", 0.0),
            }
        )
    return text_elements, original_text_elements


def _process_symbols(symbol_results: dict[str, Any]) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for idx, detection in enumerate(symbol_results.get("detections", [])):
        bbox = detection.get("bbox")
        if not bbox:
            continue
        symbols.append(
            {
                "id": str(idx),
                "type": str(detection.get("class_id", "unknown")),
                "class_name": detection.get("class_name", "unknown"),
                "bbox": bbox,
                "score": detection.get("confidence", detection.get("score", 0.9)),
            }
        )
    return symbols


def _process_lines(
    line_results: dict[str, Any],
    image_dimensions: dict[str, int],
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    detected = line_results.get("detected_lines")
    if detected is None:
        rel_key = line_results.get("s3_results", {}).get("lines_s3_key")
        if rel_key:
            detected = json.loads(
                (Path(line_results["s3_results"]["bucket"]) / rel_key).read_text(encoding="utf-8")
            ).get("detected_lines", [])
        else:
            detected = []

    width = image_dimensions.get("width", 1280)
    height = image_dimensions.get("height", 1280)
    for idx, line in enumerate(detected):
        if all(k in line for k in ("startX", "startY", "endX", "endY")):
            lines.append(
                {
                    "id": str(idx),
                    "points": [
                        [line["startX"] * width, line["startY"] * height],
                        [line["endX"] * width, line["endY"] * height],
                    ],
                }
            )
    return lines


def _transform_config(config: dict[str, Any]) -> dict[str, Any]:
    component_filter = config.get("component_filter", {})
    return {
        "graph_distance_threshold_for_symbols": config.get("distance_threshold_symbols", 60),
        "graph_distance_threshold_for_text": config.get("distance_threshold_text", 30),
        "graph_distance_threshold_for_lines": config.get("distance_threshold_lines", 20),
        "graph_line_buffer": config.get("line_buffer", 5),
        "graph_symbol_to_symbol_distance_threshold": config.get("symbol_distance_threshold", 100),
        "graph_symbol_to_symbol_overlap_region_threshold": config.get("symbol_overlap_threshold", 0.3),
        "graph_symbol_text_association_threshold": config.get("symbol_text_association_threshold", 80),
        "graph_symbol_text_fallback_threshold": config.get("symbol_text_fallback_threshold", 120),
        "junction_detection_tolerance": config.get("junction_detection_tolerance", 10.0),
        "t_junction_endpoint_threshold": config.get("t_junction_endpoint_threshold", 15.0),
        "junction_clustering_radius": config.get("junction_clustering_radius", 20.0),
        "junction_angle_tolerance": config.get("junction_angle_tolerance", 15.0),
        "minimum_line_length": config.get("minimum_line_length", 5.0),
        "intersection_snap_distance": config.get("intersection_snap_distance", 5.0),
        "line_aberration_tolerance": config.get("line_aberration_tolerance", 5.0),
        "junction_proximity_threshold": config.get("junction_proximity_threshold", 10.0),
        "max_merge_iterations": config.get("max_merge_iterations", 5),
        "geometric_continuation_tolerance": config.get("geometric_continuation_tolerance", 20.0),
        "component_filter_enabled": component_filter.get("enabled", True),
        "min_component_size": component_filter.get("min_component_size", 3),
        "max_line_density": component_filter.get("max_line_density", 0.9),
        "min_symbol_density": component_filter.get("min_symbol_density", 0.1),
        "max_notes_component_size": component_filter.get("max_notes_component_size", 15),
        "frame_aspect_ratio_threshold": component_filter.get("frame_aspect_ratio_threshold", 0.1),
        "max_symbol_density_for_removal": component_filter.get("max_symbol_density_for_removal", 0.1),
        "extreme_symbol_density_threshold": component_filter.get(
            "extreme_symbol_density_threshold", 0.05
        ),
    }


def generate_graph(
    image_key: str,
    text_results: dict[str, Any],
    symbol_results: dict[str, Any],
    line_results: dict[str, Any],
    notes_result: dict[str, Any],
    graph_config: dict[str, Any],
    path_manager,
) -> dict[str, Any]:
    """Build the P&ID graph and write JSON + DEXPI outputs."""
    text_elements, original_text_elements = _process_text_elements(text_results)
    original_dims = notes_result.get("original_image_dimensions") or text_results.get(
        "image_dimensions", {"width": 0, "height": 0}
    )
    symbols = _process_symbols(symbol_results)
    lines = _process_lines(line_results, original_dims)

    transformed_config = _transform_config(graph_config)
    processor = DocumentProcessor(transformed_config)
    graph_data = processor.process_document(
        image_key, symbols, lines, text_elements, original_text_elements
    )
    dexpi_xml = convert_to_dexpi(graph_data, image_key, original_dims)

    graph_key = path_manager.get_graph_data_s3_key()
    dexpi_key = path_manager.get_dexpi_s3_key()
    path_manager.write_json(graph_key, graph_data)
    path_manager.write_bytes(dexpi_key, dexpi_xml.encode("utf-8"))

    summary = {
        "text_elements_count": len(text_elements),
        "symbols_count": len(symbols),
        "lines_count": len(lines),
        "total_nodes": graph_data.get("graph_stats", {}).get("num_nodes", 0),
        "total_edges": graph_data.get("graph_stats", {}).get("num_edges", 0),
    }
    logger.info("Graph generated: %s", summary)
    return {
        "statusCode": 200,
        "graph_data_key": graph_key,
        "dexpi_key": dexpi_key,
        "graph_summary": summary,
        "graph_data": graph_data,
    }
