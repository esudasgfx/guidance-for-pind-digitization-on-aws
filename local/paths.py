"""Filesystem-based execution path manager mirroring the AWS pipeline layout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LocalExecutionPathManager:
    """Organizes pipeline outputs under ``<output_dir>/<execution_id>/``."""

    def __init__(self, output_dir: Path, execution_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.execution_id = execution_id or datetime.now(timezone.utc).strftime(
            "local-%Y%m%d-%H%M%S"
        )
        self.clean_execution_id = self.execution_id
        self.output_bucket = str(self.output_dir)  # compatibility with AWS helpers

    @property
    def base_path(self) -> Path:
        return self.output_dir / self.clean_execution_id

    def ensure_dirs(self) -> None:
        for sub in (
            "config",
            "input",
            "notes-processing",
            "text-detection",
            "symbol-detection",
            "line-detection",
            "graph",
            "visualization",
        ):
            (self.base_path / sub).mkdir(parents=True, exist_ok=True)

    def rel(self, *parts: str) -> str:
        return str(Path(self.clean_execution_id, *parts))

    def abs_path(self, *parts: str) -> Path:
        return self.base_path.joinpath(*parts)

    def get_config_s3_key(self) -> str:
        return self.rel("config", "processing_config.json")

    def get_processed_image_s3_key(self) -> str:
        return self.rel("notes-processing", "processed_image.png")

    def get_notes_metadata_s3_key(self) -> str:
        return self.rel("notes-processing", "notes_metadata.json")

    def get_text_detection_results_s3_key(self) -> str:
        return self.rel("text-detection", "text_detection_results.json")

    def get_symbol_results_s3_key(self) -> str:
        return self.rel("symbol-detection", "detections.json")

    def get_line_detection_path(self) -> str:
        return self.rel("line-detection")

    def get_graph_data_s3_key(self) -> str:
        return self.rel("graph", "graph_data.json")

    def get_dexpi_s3_key(self) -> str:
        return self.rel("graph", "dexpi_output.xml")

    def write_json(self, rel_key: str, data: Any) -> Path:
        path = self.output_dir / rel_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def read_json(self, rel_key: str) -> Any:
        path = self.output_dir / rel_key
        return json.loads(path.read_text(encoding="utf-8"))

    def write_bytes(self, rel_key: str, data: bytes) -> Path:
        path = self.output_dir / rel_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
