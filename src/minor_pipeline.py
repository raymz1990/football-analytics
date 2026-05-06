# src/minor_pipeline.py
# Pipeline completo para ligas menores.
# Integra coleta → modelagem → análise de value bets (1X2 + Over FT + Over HT).

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.utils import load_config, get_logger, today_str, ensure_dirs
from src.collectors.minor_collector import (
    run_minor_collection_pipeline,
    fetch_minor_league_history,
    build_minor_league_profile,
    _event_key,
)
from src.models.minor_model import MinorLeagueModel, filter_minor_value_bets

logger = get_logger("minor_pipeline")
cfg = load_config()


def get_team_history_from_league(history: pd.DataFrame, team_name: str,
                                  is_home: bool, n: int = 10) -> Dict:
    """Extrai estatísticas ofensivas/defensivas de um time do histórico."""
    if history.empty:
        return {"scored": [], "conceded": []}

    if is_home:
        rows = history[history["home_team"] == team_name].tail(n)
        scored   = list(rows["home_goals"].astype(float))
        conceded = list(rows["away_goals"].astype(float))
    else:
        rows = history[history["away_team"] == team_name].tail(n)
        scored   = list(rows["away_goals"].astype(float))
        conceded = list(rows["home_goals"].astype(float))

    # Pega também jogos como mandante+visitante misturados
    if len(scored) < 5:
        all_rows_home = history[history["home_team"] == team_name].tail(n)
        all_rows_away = history[history["away_team"] == team_name].tail(n)
        scored   = list(all_rows_home["home_goals"]) + list(all_rows_away["away_goals"])
        conceded = list(all_rows_home["away_goals"]) + list(all_rows_away["home_goals"])

    return {"scored": scored, "conceded": conceded}


def process_minor_fixture(row: pd.Series, league_history: pd.DataFrame,
                           league_profile: Dict, odds: Dict) -> List[Dict]:
    """
    Processa um único fixture de liga menor.
    Retorna lista de apostas analisadas.
    """
    home = row["home_team"]
    away = row["away_team"]

    # Histórico dos times
    home_data = get_team_history_from_league(league_history, home, is_home=True)
    away_data = get_team_history_from_league(league_history, away, is_home=False)

    model = MinorLeagueModel(league_profile)

    # Lambdas FT
    mu_home, mu_away = model.estimate_lambdas(
        home_goals_scored=home_data["scored"],
        home_goals_conceded=home_data["conceded"],
        away_goals_scored=away_data["scored"],
        away_goals_conceded=away_data["conceded"],
    )

    # Lambdas HT
    mu_ht_home, mu_ht_away = model.estimate_ht_lambdas(mu_home, mu_away)

    # Simulação Monte Carlo
    sim = model.simulate(mu_home, mu_away, mu_ht_home, mu_ht_away)

    # Contexto para o builder
    fix_info = {
        "fixture_id":  str(row.get("fixture_id", "")),
        "home_team":   home,
        "away_team":   away,
        "league":      row.get("league_name", ""),
        "country":     row.get("country", ""),
        "group":       row.get("group", ""),
        "kickoff":     str(row.get("date", "")),
        "mu_home_ft":  round(mu_home, 3),
        "mu_away_ft":  round(mu_away, 3),
        "mu_home_ht":  round(mu_ht_home, 3),
        "mu_away_ht":  round(mu_ht_away, 3),
        "n_history":   len(league_history),
    }

    # Análise de mercados
    bets = model.analyze_markets(sim, odds, fix_info)

    # Adiciona metadados do fixture
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
    """Pipeline principal para ligas menores."""
    logger.info("=" * 60)
    logger.info("BetAnalytics Pro — Minor Leagues Pipeline")
    logger.info("=" * 60)

    ensure_dirs(cfg)

    # 1. Coleta
    logger.info("[1/3] Collecting minor league data...")
    collected = run_minor_collection_pipeline()
    fixtures: pd.DataFrame = collected["fixtures"]
    odds_map: Dict = collected["odds_map"]
    league_profiles: Dict = collected["league_profiles"]

    if fixtures.empty:
        logger.warning("No minor league fixtures today.")
        return {
            "date": today_str(),
            "minor_fixtures": 0,
            "minor_value_bets": [],
            "minor_league_profiles": [],
        }

    # 2. Processar fixture por fixture
    logger.info(f"[2/3] Analyzing {len(fixtures)} minor league fixtures...")
    all_value_bets: List[Dict] = []

    # Carrega histórico por liga (reutiliza entre fixtures da mesma liga)
    league_histories: Dict[int, pd.DataFrame] = {}

    for _, row in fixtures.iterrows():
        lid = row["league_id"]
        lname = row["league_name"]

        # Carrega histórico do disco (já coletado)
        if lid not in league_histories:
            hist_path = (Path(cfg["paths"]["data_historical"])
                         / f"minor_{lid}_{lname[:20].replace(' ','_')}.csv")
            if hist_path.exists():
                league_histories[lid] = pd.read_csv(hist_path)
            else:
                league_histories[lid] = pd.DataFrame()

        history = league_histories[lid]
        profile = league_profiles.get(lname, {})
        if not profile:
            from src.collectors.minor_collector import _default_minor_profile
            profile = _default_minor_profile(lname)

        # Busca odds do evento
        key = _event_key(row["home_team"], row["away_team"])
        odds = odds_map.get(key, {})

        try:
            bets = process_minor_fixture(row, history, profile, odds)
            value_bets = filter_minor_value_bets(bets)
            all_value_bets.extend(value_bets)
        except Exception as e:
            logger.error(f"Error processing {row['home_team']} vs {row['away_team']}: {e}")

    # 3. Montar output
    logger.info(f"[3/3] Found {len(all_value_bets)} value bets in minor leagues.")

    # Agrupa por grupo geográfico
    by_group: Dict[str, List] = {}
    for b in all_value_bets:
        g = b.get("group", "other")
        by_group.setdefault(g, []).append(b)

    result = {
        "date":                  today_str(),
        "minor_fixtures":        len(fixtures),
        "minor_value_bets":      all_value_bets,
        "minor_hot_bets":        sorted(all_value_bets, key=lambda x: x["ev_pct"], reverse=True)[:15],
        "minor_by_group":        by_group,
        "minor_league_profiles": [
            {**v, "league": k} for k, v in league_profiles.items()
        ],
        "minor_fixtures_list":   fixtures.to_dict("records"),
    }

    # Salva JSON
    out_path = Path(cfg["paths"]["dashboard_output"]) / "minor_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Minor data saved: {out_path}")

    return result


if __name__ == "__main__":
    run_minor_pipeline()
