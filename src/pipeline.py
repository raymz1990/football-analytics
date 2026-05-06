# src/pipeline.py — Orquestrador principal do sistema
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from src.utils import load_config, get_logger, today_str, ensure_dirs, implied_prob
from src.collectors.collector import run_collection_pipeline, fetch_odds_movement
from src.models.statistical import PoissonModel, ZeroInflatedPoisson, MonteCarloSimulator, BayesianUpdater, DynamicElo, GlickoModel
from src.models.ml_models import EnsembleModel, TeamStyleClusterer
from src.markets.analyzer import MarketAnalyzer, filter_value_bets, OddsMovementAnalyzer, PlayerPropsAnalyzer
from src.tracking.tracker import BetTracker, LeagueProfiler, RefereeModel

logger = get_logger("pipeline")
cfg = load_config()


def load_historical_matches() -> pd.DataFrame:
    path = Path(cfg["paths"]["data_historical"]) / "matches.csv"
    if path.exists():
        return pd.read_csv(path)
    # DataFrame mínimo para não quebrar
    return pd.DataFrame(columns=[
        "home_team", "away_team", "home_goals", "away_goals",
        "league_name", "date", "league_id"
    ])


def build_mock_odds(fixture_row: pd.Series) -> Dict:
    """Odds placeholder — substituir por dados reais da API."""
    return {
        "home_win": 2.10, "draw": 3.40, "away_win": 3.60,
        "btts_yes": 1.85, "btts_no": 1.95,
        "over_05": 1.05, "under_05": 10.0,
        "over_15": 1.30, "under_15": 3.50,
        "over_25": 1.85, "under_25": 1.95,
        "over_35": 3.20, "under_35": 1.35,
        "cards_over_45": 1.80, "cards_under_45": 2.00,
    }


def build_fixture_info(row: pd.Series, historical: pd.DataFrame) -> Dict:
    """Extrai features de contexto do fixture."""
    home = row["home_team"]
    away = row["away_team"]

    # Últimos jogos
    home_hist = historical[
        (historical["home_team"] == home) | (historical["away_team"] == home)
    ].tail(15)
    away_hist = historical[
        (historical["home_team"] == away) | (historical["away_team"] == away)
    ].tail(15)

    def form_pts(df, team):
        pts = 0
        for _, r in df.iterrows():
            hg, ag = int(r.get("home_goals", 0)), int(r.get("away_goals", 0))
            is_home = r["home_team"] == team
            if (is_home and hg > ag) or (not is_home and ag > hg): pts += 3
            elif hg == ag: pts += 1
        return pts / (len(df) * 3) if len(df) > 0 else 0.5

    return {
        "home_form_5": form_pts(home_hist.tail(5), home),
        "away_form_5": form_pts(away_hist.tail(5), away),
        "home_form_10": form_pts(home_hist.tail(10), home),
        "away_form_10": form_pts(away_hist.tail(10), away),
        "home_goals_scored_avg": float(home_hist[home_hist["home_team"] == home]["home_goals"].mean() or 1.35),
        "home_goals_conceded_avg": float(home_hist[home_hist["home_team"] == home]["away_goals"].mean() or 1.10),
        "away_goals_scored_avg": float(away_hist[away_hist["away_team"] == away]["away_goals"].mean() or 1.10),
        "away_goals_conceded_avg": float(away_hist[away_hist["away_team"] == away]["home_goals"].mean() or 1.35),
        "home_xg_avg": 1.40, "away_xg_avg": 1.10,
        "home_xg_conceded_avg": 1.10, "away_xg_conceded_avg": 1.40,
        "home_elo": 1500.0, "away_elo": 1500.0, "elo_diff": 0.0,
        "home_home_win_rate": 0.45, "away_away_win_rate": 0.30,
        "home_home_goals_avg": 1.50, "away_away_goals_avg": 1.10,
        "home_injured_count": 0, "away_injured_count": 0,
        "home_injury_impact": 0.0, "away_injury_impact": 0.0,
        "ref_avg_cards": 4.5, "ref_avg_goals": 2.5, "ref_home_win_rate": 0.45,
        "wind_kmh": 10.0, "temp_c": 20.0,
        "home_days_rest": 7, "away_days_rest": 7,
        "league_avg_goals": 2.5, "league_btts_rate": 0.50, "league_over25_rate": 0.52,
        "home_corners_avg": 5.5, "away_corners_avg": 4.5,
    }


def run_models_for_fixture(row: pd.Series, historical: pd.DataFrame,
                           poisson_model: PoissonModel,
                           elo_model: DynamicElo,
                           glicko_model: GlickoModel) -> Dict:
    """Executa todos os modelos para um fixture."""
    home, away = row["home_team"], row["away_team"]
    fix_info = build_fixture_info(row, historical)

    # Lambdas via Poisson (com fallback)
    try:
        mu, nu = poisson_model.predict_mu_nu(home, away)
    except Exception:
        mu, nu = 1.35, 1.10

    # Bayesian
    bayes = BayesianUpdater()
    home_goals_hist = list(
        historical[historical["home_team"] == home]["home_goals"].dropna()
    )
    away_goals_hist = list(
        historical[historical["away_team"] == away]["away_goals"].dropna()
    )
    mu_bayes = bayes.predict_lambda(home_goals_hist, 1.35)
    nu_bayes = bayes.predict_lambda(away_goals_hist, 1.10)

    # ZIP
    zip_model = ZeroInflatedPoisson()
    zip_matrix = zip_model.predict_matrix(mu, nu)
    zip_probs = {
        "home": float(np.tril(zip_matrix, -1).sum()),
        "draw": float(np.trace(zip_matrix)),
        "away": float(np.triu(zip_matrix, 1).sum()),
    }

    # Monte Carlo
    mc = MonteCarloSimulator(n_simulations=cfg["models"]["monte_carlo_simulations"])
    mc_results = mc.simulate(mu, nu)

    # Poisson 1x2
    try:
        poisson_1x2 = poisson_model.predict_1x2(home, away)
    except Exception:
        poisson_1x2 = {"home": 0.45, "draw": 0.28, "away": 0.27}

    # Elo
    elo_1x2 = elo_model.predict_1x2(home, away)

    # Glicko
    glicko_home_win = glicko_model.predict_win_prob(home, away)
    glicko_1x2 = {
        "home": glicko_home_win * 0.85,
        "draw": 0.25,
        "away": (1 - glicko_home_win) * 0.85,
    }

    # Monte Carlo 1x2
    mc_1x2 = {
        "home": mc_results["home_win"],
        "draw": mc_results["draw"],
        "away": mc_results["away_win"],
    }

    # Bayesian 1x2 (usando mu_bayes)
    mc_bayes_results = mc.simulate(mu_bayes, nu_bayes)
    bayes_1x2 = {
        "home": mc_bayes_results["home_win"],
        "draw": mc_bayes_results["draw"],
        "away": mc_bayes_results["away_win"],
    }

    all_predictions = {
        "poisson": poisson_1x2,
        "zip": zip_probs,
        "monte_carlo": mc_1x2,
        "elo": elo_1x2,
        "glicko": glicko_1x2,
        "bayesian": bayes_1x2,
    }

    # Ensemble blend
    ensemble = EnsembleModel()
    ensemble_probs = ensemble.blend(all_predictions)

    return {
        "predictions": all_predictions,
        "ensemble": ensemble_probs,
        "mc_results": mc_results,
        "fix_info": fix_info,
        "mu": mu, "nu": nu,
    }


def analyze_fixture(row: pd.Series, model_results: Dict,
                    odds: Dict, ensemble: EnsembleModel) -> List[Dict]:
    """Analisa todos os mercados de um fixture e retorna value bets."""
    fix_info = model_results["fix_info"]
    mc = model_results["mc_results"]
    ens_probs = model_results["ensemble"]
    all_preds = model_results["predictions"]

    analyzer = MarketAnalyzer(
        ensemble_predictions=ens_probs,
        monte_carlo=mc,
        odds_row=odds,
        fixture_info=fix_info,
    )

    all_bets = analyzer.run_all()

    for bet in all_bets:
        bet["fixture_id"] = str(row.get("fixture_id", ""))
        bet["home_team"] = row["home_team"]
        bet["away_team"] = row["away_team"]
        bet["league"] = row.get("league_name", "")
        bet["kickoff"] = str(row.get("date", ""))
        bet["convergence"] = ensemble.convergence_score(all_preds, bet["outcome"])
        bet["model_std"] = ensemble.std_across_models(all_preds, bet.get("outcome", "home"))
        bet["clv_expected"] = bet["ev_pct"] * 0.6  # proxy
        bet["models_detail"] = {m: p.get(bet["outcome"], 0) for m, p in all_preds.items()}

    return filter_value_bets(all_bets)


def build_daily_output(value_bets: List[Dict], fixtures: pd.DataFrame,
                       metrics: Dict, league_profiles: List[Dict],
                       tracker: BetTracker) -> Dict:
    """Monta estrutura de dados para o dashboard."""
    date_str = today_str()

    # Hot bets (top 10)
    hot_bets = sorted(value_bets, key=lambda x: x.get("ev_pct", 0), reverse=True)[:10]

    # Ranking por league
    bets_by_league: Dict[str, List] = {}
    for b in value_bets:
        lg = b.get("league", "Outros")
        bets_by_league.setdefault(lg, []).append(b)

    # Recent results
    recent_df = tracker.get_recent(20)
    recent = recent_df.to_dict("records") if not recent_df.empty else []

    return {
        "generated_at": datetime.now().isoformat(),
        "date": date_str,
        "total_fixtures": len(fixtures),
        "total_value_bets": len(value_bets),
        "hot_bets": hot_bets,
        "all_value_bets": value_bets,
        "bets_by_league": bets_by_league,
        "metrics": metrics,
        "league_profiles": league_profiles,
        "recent_results": recent,
    }


def save_json(data: Dict, filename: str):
    path = Path(cfg["paths"]["dashboard_output"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved: {path}")


def run_full_pipeline():
    """Entry point principal — executa pipeline completo diário."""
    logger.info("=" * 60)
    logger.info("BetAnalytics Pro — Daily Pipeline Started")
    logger.info("=" * 60)

    ensure_dirs(cfg)

    # 1. Coleta
    logger.info("[1/6] Collecting data...")
    try:
        collected = run_collection_pipeline()
        fixtures = collected["fixtures"]
    except Exception as e:
        logger.error(f"Collection failed: {e}. Using empty fixtures.")
        fixtures = pd.DataFrame(columns=[
            "fixture_id", "date", "league_id", "league_name", "country",
            "home_team", "home_id", "away_team", "away_id", "venue", "referee", "status"
        ])

    # 2. Load histórico
    logger.info("[2/6] Loading historical data...")
    historical = load_historical_matches()

    # 3. Treinar modelos
    logger.info("[3/6] Training models...")
    poisson_model = PoissonModel()
    elo_model = DynamicElo(k=cfg["models"]["elo_k_factor"])
    glicko_model = GlickoModel()

    if len(historical) >= cfg["models"]["min_matches_for_model"]:
        try:
            poisson_model.fit(historical)
            elo_model.fit_history(historical)
        except Exception as e:
            logger.warning(f"Model fitting error: {e}")

    ensemble = EnsembleModel()

    # 4. Analisar fixtures
    logger.info("[4/6] Analyzing fixtures...")
    all_value_bets = []

    for _, row in fixtures.iterrows():
        try:
            odds = build_mock_odds(row)
            model_results = run_models_for_fixture(
                row, historical, poisson_model, elo_model, glicko_model
            )
            bets = analyze_fixture(row, model_results, odds, ensemble)
            all_value_bets.extend(bets)
        except Exception as e:
            logger.error(f"Error on fixture {row.get('home_team')} vs {row.get('away_team')}: {e}")

    logger.info(f"Found {len(all_value_bets)} value bets")

    # 5. Tracking
    logger.info("[5/6] Tracking & metrics...")
    tracker = BetTracker()
    for bet in all_value_bets:
        tracker.register_bet(bet, stake_units=bet.get("kelly_pct", 1.0))

    metrics = tracker.compute_metrics(days=30)

    # League profiles
    profiler = LeagueProfiler()
    league_profiles = profiler.profile_all(historical) if not historical.empty else []

    # 6. Build output
    logger.info("[6/6] Building output...")
    output = build_daily_output(all_value_bets, fixtures, metrics, league_profiles, tracker)

    save_json(output, "data.json")
    logger.info("Pipeline complete. Dashboard data ready.")
    return output


if __name__ == "__main__":
    run_full_pipeline()
