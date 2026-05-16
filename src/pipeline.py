# src/pipeline.py
import json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from src.utils import load_config, get_logger, today_str, ensure_dirs, implied_prob, expected_value, kelly_fraction
from src.collectors.collector import run_collection_pipeline, estimate_odds_poisson
from src.models.statistical import MonteCarloSimulator
from src.tracking.tracker import BetTracker, LeagueProfiler

logger = get_logger("pipeline")
cfg    = load_config()
VB     = cfg["value_betting"]

# ── Médias de referência por liga ─────────────────────────────────────────────
LEAGUE_DEFAULTS = {
    "Premier League":      {"home": 1.54, "away": 1.33, "total": 2.87},
    "La Liga":             {"home": 1.42, "away": 1.19, "total": 2.61},
    "Bundesliga":          {"home": 1.72, "away": 1.40, "total": 3.12},
    "Serie A":             {"home": 1.38, "away": 1.15, "total": 2.53},
    "Ligue 1":             {"home": 1.35, "away": 1.09, "total": 2.44},
    "Brasileirão Série A": {"home": 1.28, "away": 1.03, "total": 2.31},
    "Champions League":    {"home": 1.50, "away": 1.20, "total": 2.70},
    "Primeira Liga":       {"home": 1.40, "away": 1.15, "total": 2.55},
    "Eredivisie":          {"home": 1.65, "away": 1.35, "total": 3.00},
    "Championship":        {"home": 1.45, "away": 1.25, "total": 2.70},
    "_default":            {"home": 1.40, "away": 1.15, "total": 2.55},
}


def load_historical() -> pd.DataFrame:
    p = Path(cfg["paths"]["data_historical"]) / "matches.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(
        columns=["home_team","away_team","home_goals","away_goals","league_name","date"])


def estimate_lambdas(home: str, away: str, league: str,
                     historical: pd.DataFrame) -> tuple:
    lg  = LEAGUE_DEFAULTS.get(league, LEAGUE_DEFAULTS["_default"])
    avg_h, avg_a = lg["home"], lg["away"]

    def team_stats(team, side):
        if historical.empty: return avg_h if side=="home" else avg_a, avg_a if side=="home" else avg_h
        col_gf = "home_goals" if side=="home" else "away_goals"
        col_ga = "away_goals" if side=="home" else "home_goals"
        col_tm = "home_team"  if side=="home" else "away_team"
        sub = historical[historical[col_tm]==team].tail(8)
        if len(sub) < 3: return avg_h if side=="home" else avg_a, avg_a if side=="home" else avg_h
        w   = min(len(sub)/8, 0.80)
        gf  = w*float(sub[col_gf].mean()) + (1-w)*(avg_h if side=="home" else avg_a)
        ga  = w*float(sub[col_ga].mean()) + (1-w)*(avg_a if side=="home" else avg_h)
        return gf, ga

    h_att, h_def = team_stats(home, "home")
    a_att, a_def = team_stats(away, "away")
    mu = float(np.clip(h_att * (a_def/max(avg_a,0.1)) * 1.10, 0.40, 5.0))
    nu = float(np.clip(a_att * (h_def/max(avg_h,0.1)),        0.30, 4.5))
    return mu, nu


def classify_tier(ev_pct, prob):
    t = VB["tiers"]
    if ev_pct >= t["elite"]["ev_min"]    and prob >= t["elite"]["prob_min"]:    return "Elite",    "🔥"
    if ev_pct >= t["strong"]["ev_min"]   and prob >= t["strong"]["prob_min"]:   return "Forte",    "⚡"
    if ev_pct >= t["moderate"]["ev_min"] and prob >= t["moderate"]["prob_min"]: return "Moderada", "🟡"
    return "Sem valor", ""


def analyze_fixture(row: pd.Series, sim: dict, odds: dict,
                    mu: float, nu: float, has_real_odds: bool) -> List[dict]:
    bets = []
    is_estimated = odds.get("_estimated", False) or not has_real_odds

    markets = [
        ("1X2 — Vitória Casa", "home",     "home_win",  sim["home_win"]),
        ("1X2 — Empate",       "draw",     "draw",      sim["draw"]),
        ("1X2 — Vitória Fora", "away",     "away_win",  sim["away_win"]),
        ("Over 0.5",           "over_05",  "over_05",   sim["over_05"]),
        ("Over 1.5",           "over_15",  "over_15",   sim["over_15"]),
        ("Over 2.5",           "over_25",  "over_25",   sim["over_25"]),
        ("Under 2.5",          "under_25", "under_25",  1-sim["over_25"]),
        ("Over 3.5",           "over_35",  "over_35",   sim["over_35"]),
        ("Under 3.5",          "under_35", "under_35",  1-sim["over_35"]),
        ("BTTS — Sim",         "btts_yes", "btts_yes",  sim["btts"]),
    ]

    for label, outcome, odd_key, model_p in markets:
        odd = odds.get(odd_key)
        if not odd or odd <= 1.0 or model_p <= 0:
            continue
        imp   = implied_prob(odd)
        ev    = expected_value(model_p, odd)
        kelly = kelly_fraction(model_p, odd)

        # Com odds estimadas (Poisson): exibe previsão mas NÃO classifica como value bet
        # EV positivo real só existe comparando modelo vs mercado real
        if is_estimated:
            tier, emoji = "Previsão", "📊"
            # Inclui mesmo sem EV positivo — é informação de probabilidade, não aposta
        else:
            tier, emoji = classify_tier(ev*100, model_p)
            if tier == "Sem valor":
                continue

        bets.append({
            "fixture_id":    str(row.get("fixture_id","")),
            "home_team":     row["home_team"],
            "away_team":     row["away_team"],
            "league":        row.get("league_name",""),
            "country":       row.get("country",""),
            "kickoff":       str(row.get("date","")),
            "market":        label,
            "outcome":       outcome,
            "period":        "FT",
            "bet365_odd":    round(odd, 3),
            "implied_prob":  round(imp, 4),
            "model_prob":    round(model_p, 4),
            "ev_pct":        round(ev*100, 2),
            "kelly_pct":     round(kelly*100, 2),
            "edge":          round(model_p-imp, 4),
            "mu_home_ft":    round(mu, 3),
            "mu_away_ft":    round(nu, 3),
            "convergence":   0,
            "tier":          tier,
            "tier_emoji":    emoji,
            "is_estimated":  is_estimated,
            "is_minor":      False,
            "odds_source":   "poisson_estimate" if is_estimated else "bet365_real",
        })

    return bets


def save_json(data, filename):
    p = Path(cfg["paths"]["dashboard_output"]) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved: {p}")


def run_full_pipeline() -> dict:
    logger.info("="*60)
    logger.info("BetAnalytics Pro — Main Pipeline")
    logger.info("="*60)
    ensure_dirs(cfg)

    # 1. Coleta
    logger.info("[1/4] Collecting data...")
    collected      = run_collection_pipeline()
    fixtures       = collected["fixtures"]
    odds_map       = collected.get("odds_map", {})
    has_real_odds  = collected.get("has_real_odds", False)

    # 2. Histórico
    logger.info("[2/4] Loading historical data...")
    historical = load_historical()
    logger.info(f"Historical: {len(historical)} matches")

    # 3. Analisa
    logger.info(f"[3/4] Analyzing {len(fixtures)} fixtures (real odds: {has_real_odds})...")
    mc = MonteCarloSimulator(cfg["models"]["monte_carlo_simulations"])
    all_bets: List[dict] = []

    for _, row in fixtures.iterrows():
        try:
            mu, nu = estimate_lambdas(row["home_team"], row["away_team"],
                                       row.get("league_name",""), historical)
            sim    = mc.simulate(mu, nu)

            # Tenta odds reais primeiro
            key_exact = f"{row['home_team'].lower()}|{row['away_team'].lower()}"
            odds = odds_map.get(key_exact, {})

            # Fuzzy match se não encontrar
            if not odds:
                for k in odds_map:
                    h, a = k.split("|")
                    if (h in row["home_team"].lower() or row["home_team"].lower() in h) and \
                       (a in row["away_team"].lower() or row["away_team"].lower() in a):
                        odds = odds_map[k]
                        break

            # Fallback: odds estimadas via Poisson
            if not odds:
                odds = estimate_odds_poisson(mu, nu)
                logger.debug(f"  Poisson odds fallback: {row['home_team']} vs {row['away_team']}")

            bets = analyze_fixture(row, sim, odds, mu, nu, has_real_odds)
            all_bets.extend(bets)
        except Exception as e:
            logger.error(f"Error: {row.get('home_team')} vs {row.get('away_team')}: {e}")

    logger.info(f"[4/4] Value bets: {len(all_bets)}")

    # 4. Tracking
    tracker = BetTracker()
    for b in all_bets:
        try: tracker.register_bet(b)
        except: pass
    metrics = tracker.compute_metrics(30)

    profiler       = LeagueProfiler()
    league_profiles = profiler.profile_all(historical) if not historical.empty else []
    recent          = tracker.get_recent(20).to_dict("records") if not tracker.get_recent(20).empty else []
    hot_bets        = sorted(all_bets, key=lambda x: x["ev_pct"], reverse=True)[:10]

    result = {
        "generated_at":    datetime.now().isoformat(),
        "date":            today_str(),
        "total_fixtures":  len(fixtures),
        "total_value_bets":len(all_bets),
        "has_real_odds":   has_real_odds,
        "hot_bets":        hot_bets,
        "all_value_bets":  all_bets,
        "bets_by_league":  {},
        "metrics":         metrics,
        "league_profiles": league_profiles,
        "recent_results":  recent,
        "minor":           {},
    }

    logger.info("="*60)
    logger.info(f"Fixtures: {len(fixtures)} | Value Bets: {len(all_bets)} | Real Odds: {has_real_odds}")
    return result
