# src/minor_pipeline.py — Pipeline para ligas menores
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.utils import load_config, get_logger, today_str, ensure_dirs
from src.collectors.minor_collector import run_minor_collection_pipeline, _default_profile
from src.models.minor_model import MinorLeagueModel, filter_minor_value_bets

logger = get_logger("minor_pipeline")
cfg    = load_config()


def process_minor_fixture(row: pd.Series, league_profile: Dict, odds: Dict) -> List[Dict]:
    """Processa um fixture de liga menor — sem histórico de time individual."""
    home = row["home_team"]
    away = row["away_team"]

    model  = MinorLeagueModel(league_profile)

    # Sem dados de times individuais → usa apenas o perfil da liga
    mu, nu = model.estimate_lambdas(
        home_goals_scored=[],
        home_goals_conceded=[],
        away_goals_scored=[],
        away_goals_conceded=[],
    )

    mu_ht, nu_ht = model.estimate_ht_lambdas(mu, nu)
    sim = model.simulate(mu, nu, mu_ht, nu_ht)

    fix_info = {
        "fixture_id":  str(row.get("fixture_id", "")),
        "home_team":   home,
        "away_team":   away,
        "league":      row.get("league_name", ""),
        "country":     row.get("country", ""),
        "group":       row.get("group", "other"),
        "kickoff":     str(row.get("date", "")),
        "mu_home_ft":  round(mu, 3),
        "mu_away_ft":  round(nu, 3),
        "mu_home_ht":  round(mu_ht, 3),
        "mu_away_ht":  round(nu_ht, 3),
    }

    bets = model.analyze_markets(sim, odds, fix_info)

    for b in bets:
        b.update({
            "fixture_id": fix_info["fixture_id"],
            "home_team":  home,
            "away_team":  away,
            "league":     fix_info["league"],
            "country":    fix_info["country"],
            "group":      fix_info["group"],
            "kickoff":    fix_info["kickoff"],
            "sim_avg_ft": round(sim["avg_goals_ft"], 2),
            "sim_avg_ht": round(sim["avg_goals_ht"], 2),
            "is_minor":   True,
        })

    return bets


def run_minor_pipeline() -> Dict:
    logger.info("=" * 60)
    logger.info("BetAnalytics Pro — Minor Leagues Pipeline")
    logger.info("=" * 60)
    ensure_dirs(cfg)

    # 1. Coleta
    logger.info("[1/3] Collecting minor league data...")
    collected        = run_minor_collection_pipeline()
    fixtures         = collected["fixtures"]
    odds_map         = collected["odds_map"]
    league_profiles  = collected["league_profiles"]

    if fixtures.empty:
        logger.warning("No minor league fixtures today.")
        return {
            "date":                  today_str(),
            "minor_fixtures":        0,
            "minor_value_bets":      [],
            "minor_hot_bets":        [],
            "minor_by_group":        {},
            "minor_league_profiles": [],
        }

    # 2. Analisa cada fixture
    logger.info(f"[2/3] Analyzing {len(fixtures)} minor fixtures...")
    all_value_bets: List[Dict] = []

    for _, row in fixtures.iterrows():
        lname   = row.get("league_name", "Unknown")
        profile = league_profiles.get(lname) or _default_profile(lname)
        key     = f"{str(row['home_team']).lower()}|{str(row['away_team']).lower()}"
        odds    = odds_map.get(key, {})

        try:
            bets   = process_minor_fixture(row, profile, odds)
            valued = filter_minor_value_bets(bets)
            all_value_bets.extend(valued)
        except Exception as e:
            logger.error(f"Error on {row.get('home_team')} vs {row.get('away_team')}: {e}")

    logger.info(f"[3/3] Minor value bets found: {len(all_value_bets)}")

    # Agrupa por região
    by_group: Dict[str, List] = {}
    for b in all_value_bets:
        g = b.get("group", "other")
        by_group.setdefault(g, []).append(b)

    result = {
        "date":                  today_str(),
        "minor_fixtures":        len(fixtures),
        "minor_value_bets":      all_value_bets,
        "minor_hot_bets":        sorted(all_value_bets, key=lambda x: x.get("ev_pct", 0), reverse=True)[:15],
        "minor_by_group":        by_group,
        "minor_league_profiles": list(league_profiles.values()),
    }

    # Salva JSON separado
    out_path = Path(cfg["paths"]["dashboard_output"]) / "minor_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Minor data saved: {out_path}")

    return result


if __name__ == "__main__":
    run_minor_pipeline()
