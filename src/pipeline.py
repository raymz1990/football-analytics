# src/pipeline.py — Orquestrador principal
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.utils import load_config, get_logger, today_str, ensure_dirs, implied_prob, expected_value, kelly_fraction
from src.collectors.collector import run_collection_pipeline
from src.models.statistical import PoissonModel, MonteCarloSimulator, BayesianUpdater, DynamicElo, GlickoModel
from src.models.ml_models import EnsembleModel
from src.tracking.tracker import BetTracker, LeagueProfiler

logger = get_logger("pipeline")
cfg = load_config()
VB  = cfg["value_betting"]


def load_historical_matches() -> pd.DataFrame:
    path = Path(cfg["paths"]["data_historical"]) / "matches.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "home_team","away_team","home_goals","away_goals","league_name","date","league_id"
    ])


def get_league_avg(historical: pd.DataFrame, league_id: int) -> dict:
    """Médias da liga a partir do histórico."""
    sub = historical[historical.get("league_id", pd.Series()) == league_id] if "league_id" in historical.columns else historical
    if sub.empty or len(sub) < 5:
        return {"home": 1.35, "away": 1.10, "total": 2.45}
    return {
        "home":  float(sub["home_goals"].mean()),
        "away":  float(sub["away_goals"].mean()),
        "total": float((sub["home_goals"] + sub["away_goals"]).mean()),
    }


def estimate_lambdas(row: pd.Series, historical: pd.DataFrame) -> tuple:
    """Estima mu (casa) e nu (fora) com shrinkage Bayesian."""
    home, away = row["home_team"], row["away_team"]
    lg_avg = get_league_avg(historical, row.get("league_id", 0))

    def team_avg(team, side):
        col_gf = "home_goals" if side == "home" else "away_goals"
        col_ga = "away_goals" if side == "home" else "home_goals"
        key    = "home_team" if side == "home" else "away_team"
        sub = historical[historical[key] == team].tail(10) if not historical.empty else pd.DataFrame()
        if sub.empty:
            return lg_avg["home"] if side == "home" else lg_avg["away"], \
                   lg_avg["away"] if side == "home" else lg_avg["home"]
        w = min(len(sub) / 10, 0.85)
        gf = w * float(sub[col_gf].mean()) + (1 - w) * (lg_avg["home"] if side == "home" else lg_avg["away"])
        ga = w * float(sub[col_ga].mean()) + (1 - w) * (lg_avg["away"] if side == "home" else lg_avg["home"])
        return gf, ga

    home_gf, home_ga = team_avg(home, "home")
    away_gf, away_ga = team_avg(away, "away")

    mu = np.clip(home_gf * (away_ga / max(lg_avg["away"], 0.1)) * 1.10, 0.3, 5.0)
    nu = np.clip(away_gf * (home_ga / max(lg_avg["home"], 0.1)),       0.3, 4.5)
    return float(mu), float(nu)


def simulate_fixture(mu: float, nu: float) -> dict:
    mc = MonteCarloSimulator(n_simulations=cfg["models"]["monte_carlo_simulations"])
    return mc.simulate(mu, nu)


def build_bets(row: pd.Series, sim: dict, odds: dict,
               mu: float, nu: float) -> List[dict]:
    """Gera lista de apostas analisadas para todos os mercados disponíveis."""
    bets = []

    def add(market, outcome, odd, model_p):
        if not odd or odd <= 1.0 or model_p <= 0:
            return
        imp  = implied_prob(odd)
        ev   = expected_value(model_p, odd)
        bets.append({
            "fixture_id":   str(row.get("fixture_id", "")),
            "home_team":    row["home_team"],
            "away_team":    row["away_team"],
            "league":       row.get("league_name", ""),
            "country":      row.get("country", ""),
            "kickoff":      str(row.get("date", "")),
            "market":       market,
            "outcome":      outcome,
            "period":       "FT",
            "bet365_odd":   round(odd, 3),
            "implied_prob": round(imp, 4),
            "model_prob":   round(model_p, 4),
            "ev":           round(ev, 4),
            "ev_pct":       round(ev * 100, 2),
            "kelly_pct":    round(kelly_fraction(model_p, odd) * 100, 2),
            "edge":         round(model_p - imp, 4),
            "mu_home_ft":   round(mu, 3),
            "mu_away_ft":   round(nu, 3),
            "convergence":  0,
            "is_minor":     False,
        })

    # 1X2
    add("1X2 — Vitória Casa",  "home",  odds.get("home_win"), sim.get("home_win", 0))
    add("1X2 — Empate",        "draw",  odds.get("draw"),     sim.get("draw", 0))
    add("1X2 — Vitória Fora",  "away",  odds.get("away_win"), sim.get("away_win", 0))

    # Over/Under
    for line, key in [(0.5,"05"),(1.5,"15"),(2.5,"25"),(3.5,"35")]:
        add(f"Over {line}",  f"over_{key}",  odds.get(f"over_{key}"),  sim.get(f"over_{key}", 0))
        add(f"Under {line}", f"under_{key}", odds.get(f"under_{key}"), sim.get(f"under_{key}", 0))

    # BTTS
    add("BTTS — Sim", "btts_yes", odds.get("btts_yes"), sim.get("btts", 0))
    add("BTTS — Não", "btts_no",  odds.get("btts_no"),  1 - sim.get("btts", 0))

    return bets


def classify_tier(ev_pct: float, prob: float) -> tuple:
    t = VB["tiers"]
    if ev_pct >= t["elite"]["ev_min"]    and prob >= t["elite"]["prob_min"]:    return "Elite",    "🔥"
    if ev_pct >= t["strong"]["ev_min"]   and prob >= t["strong"]["prob_min"]:   return "Forte",    "⚡"
    if ev_pct >= t["moderate"]["ev_min"] and prob >= t["moderate"]["prob_min"]: return "Moderada", "🟡"
    return "Sem valor", "❌"


def filter_value_bets(bets: List[dict]) -> List[dict]:
    out = []
    for b in bets:
        if (b["ev_pct"]     >= VB["min_ev_pct"] and
            b["model_prob"] >= VB["min_probability"] and
            b["bet365_odd"] >= VB["min_odds"]):
            tier, emoji = classify_tier(b["ev_pct"], b["model_prob"])
            b["tier"]       = tier
            b["tier_emoji"] = emoji
            out.append(b)
    return sorted(out, key=lambda x: x["ev_pct"], reverse=True)


def save_json(data: dict, filename: str):
    path = Path(cfg["paths"]["dashboard_output"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved: {path}")


def run_full_pipeline() -> dict:
    logger.info("=" * 60)
    logger.info("BetAnalytics Pro — Main Pipeline")
    logger.info("=" * 60)
    ensure_dirs(cfg)

    # 1. Coleta
    logger.info("[1/4] Collecting data...")
    collected = run_collection_pipeline()
    fixtures: pd.DataFrame = collected["fixtures"]
    odds_map: dict          = collected.get("odds_map", {})

    if fixtures.empty:
        logger.warning("No fixtures collected today — dashboard will show 0 fixtures.")

    # 2. Histórico
    logger.info("[2/4] Loading historical data...")
    historical = load_historical_matches()
    logger.info(f"Historical matches loaded: {len(historical)}")

    # 3. Modelos
    logger.info("[3/4] Training models...")
    elo_model = DynamicElo(k=cfg["models"]["elo_k_factor"])
    if len(historical) >= cfg["models"]["min_matches_for_model"]:
        try:
            elo_model.fit_history(historical)
        except Exception as e:
            logger.warning(f"Elo fit failed: {e}")

    # 4. Analisa cada fixture
    logger.info(f"[4/4] Analyzing {len(fixtures)} fixtures...")
    all_bets: List[dict] = []

    for _, row in fixtures.iterrows():
        try:
            mu, nu = estimate_lambdas(row, historical)
            sim    = simulate_fixture(mu, nu)

            # Busca odds por par de times
            key  = f"{row['home_team'].lower()}|{row['away_team'].lower()}"
            odds = odds_map.get(key, {})

            bets = build_bets(row, sim, odds, mu, nu)
            all_bets.extend(bets)
        except Exception as e:
            logger.error(f"Error on {row.get('home_team')} vs {row.get('away_team')}: {e}")

    value_bets = filter_value_bets(all_bets)
    logger.info(f"Value bets found: {len(value_bets)} (from {len(all_bets)} analyzed)")

    # Tracking
    tracker = BetTracker()
    for b in value_bets:
        try:
            tracker.register_bet(b)
        except Exception:
            pass
    metrics = tracker.compute_metrics(days=30)

    # League profiles
    profiler = LeagueProfiler()
    league_profiles = profiler.profile_all(historical) if not historical.empty else []

    # Fixtures como lista para o dashboard
    fixtures_list = fixtures.to_dict("records") if not fixtures.empty else []

    # Recentes
    recent_df = tracker.get_recent(20)
    recent    = recent_df.to_dict("records") if not recent_df.empty else []

    hot_bets   = sorted(value_bets, key=lambda x: x["ev_pct"], reverse=True)[:10]

    result = {
        "generated_at":    datetime.now().isoformat(),
        "date":            today_str(),
        "total_fixtures":  len(fixtures),
        "total_value_bets":len(value_bets),
        "hot_bets":        hot_bets,
        "all_value_bets":  value_bets,
        "fixtures_list":   fixtures_list,
        "bets_by_league":  {},
        "metrics":         metrics,
        "league_profiles": league_profiles,
        "recent_results":  recent,
        "minor":           {},
    }

    logger.info("=" * 60)
    logger.info(f"Pipeline complete — fixtures: {len(fixtures)}, value bets: {len(value_bets)}")
    return result
