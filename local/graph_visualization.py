"""Graph visualization without S3 (physical layout + graph representation PNGs)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_VIZ_DIR = REPO_ROOT / "cdk" / "lambda" / "graph_visualization"

# Headless backend must be set before matplotlib is imported by visualization modules.
os.environ.setdefault("MPLBACKEND", "Agg")

if str(GRAPH_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_VIZ_DIR))

from visualization import GraphVisualizer  # noqa: E402


def _notes_overlay_info(notes_result: dict[str, Any]) -> dict[str, Any] | None:
    """Prefer frame-adjusted notes coordinates to match the AWS visualization step."""
    frame_adjusted = notes_result.get("frame_adjusted_notes_coordinates")
    if frame_adjusted:
        return frame_adjusted
    notes_coordinates = notes_result.get("notes_coordinates")
    if notes_coordinates:
        return notes_coordinates
    return None


def generate_visualizations(
    graph_data: dict[str, Any],
    path_manager,
    notes_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Render physical layout and graph representation PNGs to the visualization output folder.

    Returns a summary dict with relative keys and absolute paths for each PNG.
    """
    notes_info = _notes_overlay_info(notes_result) if notes_result else None
    visualizer = GraphVisualizer()
    images = visualizer.create_all_visualizations(graph_data, notes_info=notes_info)

    if not images:
        logger.warning("No visualizations were generated")
        return {"statusCode": 200, "success": False, "visualizations": {}}

    output: dict[str, str] = {}
    abs_paths: dict[str, str] = {}
    for viz_type, viz_bytes in images.items():
        if not viz_bytes:
            continue
        rel_key = path_manager.rel("visualization", f"{viz_type}.png")
        abs_path = path_manager.write_bytes(rel_key, viz_bytes)
        output[viz_type] = rel_key
        abs_paths[viz_type] = str(abs_path)
        logger.info("Saved %s visualization to %s", viz_type, abs_path)

    return {
        "statusCode": 200,
        "success": True,
        "visualizations": output,
        "visualization_paths": abs_paths,
        "count": len(output),
    }
