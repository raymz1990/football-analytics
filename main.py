#!/usr/bin/env python3
# main.py — Entry point do BetAnalytics Pro
import sys
import json
import argparse
from pathlib import Path
from src.utils import load_config, get_logger, ensure_dirs
from src.pipeline import run_full_pipeline, save_json
from src.dashboard.generator import build_and_save_dashboard

logger = get_logger("main")
cfg = load_config()


def main():
    parser = argparse.ArgumentParser(description="BetAnalytics Pro — Football Value Betting System")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection (use cached)")
    parser.add_argument("--only-dashboard", action="store_true", help="Only rebuild dashboard from existing data.json")
    args = parser.parse_args()

    ensure_dirs(cfg)

    if args.only_dashboard:
        data_path = Path(cfg["paths"]["dashboard_output"]) / "data.json"
        if not data_path.exists():
            logger.error("data.json not found. Run without --only-dashboard first.")
            sys.exit(1)
        with open(data_path) as f:
            data = json.load(f)
    else:
        logger.info("Running full pipeline...")
        data = run_full_pipeline()
        save_json(data, "data.json")

    logger.info("Building dashboard...")
    out = build_and_save_dashboard(data)
    logger.info(f"Done! Dashboard: {out}")


if __name__ == "__main__":
    main()
