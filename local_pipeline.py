#!/usr/bin/env python3
"""
Local P&ID digitization pipeline (no AWS).

Chains notes preprocessing, PaddleOCR text detection, local symbol detection,
line detection, and graph generation using code extracted from the AWS Lambda
functions in this repository.

Example:
    python local_pipeline.py diagram.png --model-dir ./models --output ./output
    python local_pipeline.py diagram.png --model-dir ./models --visualize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from local.graph_generation import generate_graph
from local.graph_visualization import generate_visualizations
from local.line_detection import detect_lines
from local.notes_processing import process_notes
from local.paths import LocalExecutionPathManager
from local.symbol_detection import detect_symbols
from local.text_detection import detect_text

DEFAULT_CONFIG_PATH = REPO_ROOT / "cdk" / "lambda" / "input_validator" / "default_config.json"


def _load_config(config_path: Path | None, overrides: dict[str, Any] | None) -> dict[str, Any]:
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif DEFAULT_CONFIG_PATH.exists():
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        config = {}

    if overrides:
        for section, values in overrides.items():
            config.setdefault(section, {}).update(values)
    return config


def _resolve_model_dir(model_dir: Path) -> Path:
    model_dir = model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Download model.tar.gz from the GitHub release and extract it:\n"
            "  wget https://github.com/aws-solutions-library-samples/"
            "guidance-for-piping-and-instrumentation-diagrams-digitization-on-aws/"
            "releases/download/v1.0.0/model.tar.gz\n"
            "  mkdir -p models && tar -xzf model.tar.gz -C models"
        )
    required = ["frcnn_checkpoint_50000.pth", "last-v9.ckpt"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Model directory is missing required files: {missing}\n"
            f"Checked: {model_dir}"
        )
    return model_dir


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    model_dir: Path,
    config: dict[str, Any],
    use_gpu: bool = True,
    skip_notes: bool = False,
    visualize: bool = False,
    execution_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full local pipeline and return a summary dict."""
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    total_steps = 6 if visualize else 5
    path_manager = LocalExecutionPathManager(output_dir, execution_id)
    path_manager.ensure_dirs()
    path_manager.write_json(path_manager.get_config_s3_key(), config)

    logging.info("Execution ID: %s", path_manager.execution_id)
    logging.info("Output directory: %s", path_manager.base_path)

    notes_config = deepcopy(config.get("notes_processing", {}))
    if skip_notes:
        notes_config["remove_notes_section"] = False
        notes_config.setdefault("frame_config", {})["remove_frame"] = False

    logging.info("Step 1/%d: Notes preprocessing", total_steps)
    notes_result = process_notes(image_path, notes_config, path_manager)

    processed_image_path = path_manager.output_dir / notes_result["processed_key"]
    original_image_path = image_path

    logging.info("Step 2/%d: Text detection (PaddleOCR on original image)", total_steps)
    text_results = detect_text(
        original_image_path,
        path_manager,
        notes_config=notes_config,
        use_gpu=use_gpu,
    )

    logging.info("Step 3/%d: Symbol detection (local PyTorch on processed image)", total_steps)
    processed_bytes = processed_image_path.read_bytes()
    symbol_results = detect_symbols(
        processed_bytes,
        model_dir=model_dir,
        path_manager=path_manager,
        config=config.get("symbol_detection", {}),
        notes_result=notes_result,
    )

    logging.info("Step 4/%d: Line detection", total_steps)
    line_results = detect_lines(
        processed_image_path=processed_image_path,
        text_results=text_results,
        symbol_results=symbol_results,
        notes_result=notes_result,
        line_config=config.get("line_detection", {}),
        path_manager=path_manager,
    )

    logging.info("Step 5/%d: Graph generation", total_steps)
    graph_results = generate_graph(
        image_key=image_path.name,
        text_results=text_results,
        symbol_results=symbol_results,
        line_results=line_results,
        notes_result=notes_result,
        graph_config=config.get("graph_generation", {}),
        path_manager=path_manager,
    )

    visualization_results: dict[str, Any] | None = None
    if visualize:
        logging.info("Step 6/%d: Graph visualization", total_steps)
        visualization_results = generate_visualizations(
            graph_data=graph_results["graph_data"],
            path_manager=path_manager,
            notes_result=notes_result,
        )

    summary = {
        "execution_id": path_manager.execution_id,
        "output_dir": str(path_manager.base_path),
        "input_image": str(image_path),
        "notes_processing": {
            "mode": notes_result.get("processing_mode"),
            "processed_image": str(processed_image_path),
        },
        "text_detection": {"count": len(text_results.get("text_elements", []))},
        "symbol_detection": {"count": symbol_results.get("num_detections", 0)},
        "line_detection": {"count": line_results.get("line_count", 0)},
        "graph": graph_results.get("graph_summary", {}),
        "artifacts": {
            "graph_data": str(path_manager.output_dir / graph_results["graph_data_key"]),
            "dexpi_xml": str(path_manager.output_dir / graph_results["dexpi_key"]),
            "text_results": str(
                path_manager.output_dir / path_manager.get_text_detection_results_s3_key()
            ),
            "symbol_results": str(
                path_manager.output_dir / path_manager.get_symbol_results_s3_key()
            ),
        },
    }
    if visualization_results:
        summary["visualization"] = {
            "count": visualization_results.get("count", 0),
            "paths": visualization_results.get("visualization_paths", {}),
        }
        summary["artifacts"].update(visualization_results.get("visualization_paths", {}))

    path_manager.write_json(path_manager.rel("execution_summary.json"), summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the P&ID digitization pipeline locally (no AWS)."
    )
    parser.add_argument("image", type=Path, help="Path to input P&ID image (PNG/JPG)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing extracted model.tar.gz artifacts",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional processing config JSON (defaults to cdk/lambda/input_validator/default_config.json)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for PaddleOCR (symbol detection still uses CUDA if available)",
    )
    parser.add_argument(
        "--skip-notes",
        action="store_true",
        help="Skip automatic notes/frame removal",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate physical layout and graph representation PNGs",
    )
    parser.add_argument(
        "--execution-id",
        default=None,
        help="Optional execution ID for output folder naming",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        model_dir = _resolve_model_dir(args.model_dir)
        config = _load_config(args.config, overrides=None)
        summary = run_pipeline(
            image_path=args.image,
            output_dir=args.output,
            model_dir=model_dir,
            config=config,
            use_gpu=not args.cpu,
            skip_notes=args.skip_notes,
            visualize=args.visualize,
            execution_id=args.execution_id,
        )
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1

    print(json.dumps(summary, indent=2))
    print(f"\nDone. Results written to: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
