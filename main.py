#!/usr/bin/env python3
# main.py — Entry point do BetAnalytics Pro
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import pytz

from src.utils import load_config, get_logger, ensure_dirs, today_str
from src.dashboard.generator import build_and_save_dashboard

logger = get_logger("main")
cfg = load_config()


def empty_data(date_str: str) -> dict:
    """Estrutura mínima para dashboard não quebrar."""
    return {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "total_fixtures": 0,
        "total_value_bets": 0,
        "hot_bets": [],
        "all_value_bets": [],
        "bets_by_league": {},
        "league_profiles": [],
        "recent_results": [],
        "metrics": {
            "total_bets": 0, "green": 0, "red": 0, "hit_rate": 0,
            "total_stake": 0, "total_pl": 0, "roi": 0, "yield_pct": 0,
            "max_drawdown": 0, "avg_clv": 0, "ev_avg": 0, "by_tier": {}
        },
        "minor": {
            "minor_fixtures": 0,
            "minor_value_bets": [],
            "minor_hot_bets": [],
            "minor_by_group": {},
            "minor_league_profiles": [],
        }
    }


def run_main_pipeline():
    from src.pipeline import run_full_pipeline, save_json
    logger.info("Running main pipeline...")
    data = run_full_pipeline()
    save_json(data, "data.json")
    return data


def run_minor_pipeline_safe():
    try:
        from src.minor_pipeline import run_minor_pipeline
        logger.info("Running minor leagues pipeline...")
        return run_minor_pipeline()
    except Exception as e:
        logger.error(f"Minor pipeline failed: {e}\n{traceback.format_exc()}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="BetAnalytics Pro")
    parser.add_argument("--skip-collect",   action="store_true")
    parser.add_argument("--only-dashboard", action="store_true")
    parser.add_argument("--only-minor",     action="store_true")
    parser.add_argument("--skip-minor",     action="store_true")
    args = parser.parse_args()

    ensure_dirs(cfg)
    out_dir = Path(cfg["paths"]["dashboard_output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = today_str()

    # ── Só rebuild do dashboard ───────────────────────────────────────────────
    if args.only_dashboard:
        data_path  = out_dir / "data.json"
        minor_path = out_dir / "minor_data.json"
        if not data_path.exists():
            logger.error("data.json not found. Running full pipeline instead.")
            data = empty_data(date_str)
        else:
            with open(data_path) as f:
                data = json.load(f)
        if minor_path.exists():
            with open(minor_path) as f:
                data["minor"] = json.load(f)
        build_and_save_dashboard(data)
        logger.info("Dashboard rebuilt.")
        return

    # ── Só pipeline minor ─────────────────────────────────────────────────────
    if args.only_minor:
        run_minor_pipeline_safe()
        return

    # ── Pipeline principal ────────────────────────────────────────────────────
    data = empty_data(date_str)

    try:
        from src.pipeline import run_full_pipeline, save_json
        data = run_full_pipeline()
        save_json(data, "data.json")
        logger.info(f"Main pipeline OK — fixtures: {data.get('total_fixtures',0)}, value bets: {data.get('total_value_bets',0)}")
    except Exception as e:
        logger.error(f"Main pipeline failed: {e}\n{traceback.format_exc()}")
        logger.warning("Continuing with empty data to generate dashboard anyway.")
        # Salva mesmo assim para não perder o estado
        data_path = out_dir / "data.json"
        with open(data_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, default=str)

    # ── Pipeline minor ────────────────────────────────────────────────────────
    if not args.skip_minor:
        minor_path = out_dir / "minor_data.json"
        minor_data = run_minor_pipeline_safe()
        if minor_data:
            data["minor"] = minor_data
            with open(minor_path, "w") as f:
                json.dump(minor_data, f, ensure_ascii=False, default=str)
        elif minor_path.exists():
            with open(minor_path) as f:
                data["minor"] = json.load(f)
    else:
        minor_path = out_dir / "minor_data.json"
        if minor_path.exists():
            with open(minor_path) as f:
                data["minor"] = json.load(f)

    # ── Sempre gera o dashboard ───────────────────────────────────────────────
    logger.info("Building dashboard...")
    build_and_save_dashboard(data)

    # ── Sumário final ─────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info(f"Date:           {data.get('date')}")
    logger.info(f"Fixtures:       {data.get('total_fixtures', 0)}")
    logger.info(f"Value Bets:     {data.get('total_value_bets', 0)}")
    minor = data.get("minor", {})
    logger.info(f"Minor Fixtures: {minor.get('minor_fixtures', 0)}")
    logger.info(f"Minor VBets:    {len(minor.get('minor_value_bets', []))}")
    logger.info(f"Dashboard:      {out_dir}/index.html")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
