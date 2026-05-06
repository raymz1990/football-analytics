#!/usr/bin/env python3
# main.py — Entry point do BetAnalytics Pro
import sys
import json
import argparse
from pathlib import Path
from src.utils import load_config, get_logger, ensure_dirs
from src.pipeline import run_full_pipeline, save_json
from src.minor_pipeline import run_minor_pipeline
from src.dashboard.generator import build_and_save_dashboard

logger = get_logger("main")
cfg = load_config()


def main():
    parser = argparse.ArgumentParser(description="BetAnalytics Pro — Football Value Betting System")
    parser.add_argument("--skip-collect",   action="store_true", help="Skip data collection (use cached)")
    parser.add_argument("--only-dashboard", action="store_true", help="Only rebuild dashboard from existing JSONs")
    parser.add_argument("--only-minor",     action="store_true", help="Only run minor leagues pipeline")
    parser.add_argument("--skip-minor",     action="store_true", help="Skip minor leagues pipeline")
    args = parser.parse_args()

    ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["dashboard_output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only_dashboard:
        data_path  = out_dir / "data.json"
        minor_path = out_dir / "minor_data.json"
        if not data_path.exists():
            logger.error("data.json not found.")
            sys.exit(1)
        with open(data_path) as f:
            data = json.load(f)
        if minor_path.exists():
            with open(minor_path) as f:
                data["minor"] = json.load(f)
        build_and_save_dashboard(data)
        return

    if args.only_minor:
        run_minor_pipeline()
        return

    logger.info("Running main pipeline...")
    data = run_full_pipeline()
    save_json(data, "data.json")

    if not args.skip_minor:
        logger.info("Running minor leagues pipeline...")
        try:
            minor_data = run_minor_pipeline()
            data["minor"] = minor_data
        except Exception as e:
            logger.error(f"Minor pipeline failed: {e}")
            data["minor"] = {}
    else:
        minor_path = out_dir / "minor_data.json"
        data["minor"] = json.load(open(minor_path)) if minor_path.exists() else {}

    logger.info("Building dashboard...")
    build_and_save_dashboard(data)
    logger.info("Done!")


if __name__ == "__main__":
    main()
