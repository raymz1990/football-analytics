# src/tracking/tracker.py — Tracking de resultados, ROI e CLV
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.utils import get_logger, load_config, today_str

logger = get_logger("tracking")
cfg = load_config()
HIST_DIR = Path(cfg["paths"]["data_historical"])


class BetTracker:
    """Registra apostas, busca resultados e calcula métricas."""

    COLUMNS = [
        "bet_id", "date", "fixture_id", "home_team", "away_team",
        "league", "market", "outcome", "bet365_odd", "model_prob",
        "ev_pct", "kelly_pct", "tier", "stake_units",
        "result", "profit_loss", "roi", "clv_realized",
        "home_score", "away_score",
    ]

    def __init__(self):
        self.history_path = HIST_DIR / "bet_history.csv"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if self.history_path.exists():
            return pd.read_csv(self.history_path, parse_dates=["date"])
        return pd.DataFrame(columns=self.COLUMNS)

    def _save(self):
        self.df.to_csv(self.history_path, index=False)

    # ── REGISTRO ─────────────────────────────────────────────────────────────

    def register_bet(self, bet: Dict, stake_units: float = 1.0) -> str:
        bet_id = f"{bet['fixture_id']}_{bet['outcome']}_{today_str()}"
        if bet_id in self.df.get("bet_id", pd.Series()).values:
            return bet_id

        row = {
            "bet_id": bet_id,
            "date": today_str(),
            "fixture_id": bet.get("fixture_id"),
            "home_team": bet.get("home_team"),
            "away_team": bet.get("away_team"),
            "league": bet.get("league"),
            "market": bet.get("market"),
            "outcome": bet.get("outcome"),
            "bet365_odd": bet.get("bet365_odd"),
            "model_prob": bet.get("model_prob"),
            "ev_pct": bet.get("ev_pct"),
            "kelly_pct": bet.get("kelly_pct"),
            "tier": bet.get("tier"),
            "stake_units": stake_units,
            "result": "PENDING",
            "profit_loss": None,
            "roi": None,
            "clv_realized": None,
            "home_score": None,
            "away_score": None,
        }

        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._save()
        return bet_id

    # ── SETTLEMENT ───────────────────────────────────────────────────────────

    def settle_bet(self, bet_id: str, won: bool,
                   home_score: int, away_score: int,
                   closing_odd: Optional[float] = None):
        idx = self.df[self.df["bet_id"] == bet_id].index
        if idx.empty:
            logger.warning(f"Bet {bet_id} not found")
            return

        i = idx[0]
        stake = float(self.df.at[i, "stake_units"])
        odd = float(self.df.at[i, "bet365_odd"])

        if won:
            pl = stake * (odd - 1)
            result = "GREEN ✅"
        else:
            pl = -stake
            result = "RED ❌"

        clv = 0.0
        if closing_odd and closing_odd > 0:
            clv = ((odd / closing_odd) - 1) * 100

        self.df.at[i, "result"] = result
        self.df.at[i, "profit_loss"] = round(pl, 4)
        self.df.at[i, "roi"] = round(pl / stake * 100, 2)
        self.df.at[i, "clv_realized"] = round(clv, 2)
        self.df.at[i, "home_score"] = home_score
        self.df.at[i, "away_score"] = away_score
        self._save()
        logger.info(f"Settled {bet_id}: {'WIN' if won else 'LOSS'} | P/L: {pl:+.2f}u")

    def settle_from_result(self, fixture_id: str, home_score: int, away_score: int,
                           closing_odds: Dict[str, float] = None):
        """Liquida automaticamente todas as apostas de um fixture."""
        pending = self.df[
            (self.df["fixture_id"] == fixture_id) &
            (self.df["result"] == "PENDING")
        ]

        for _, row in pending.iterrows():
            won = self._check_outcome(
                row["outcome"], home_score, away_score
            )
            co = (closing_odds or {}).get(row["outcome"])
            self.settle_bet(row["bet_id"], won, home_score, away_score, co)

    def _check_outcome(self, outcome: str, hg: int, ag: int) -> bool:
        total = hg + ag
        mapping = {
            "home": hg > ag,
            "draw": hg == ag,
            "away": ag > hg,
            "btts_yes": hg > 0 and ag > 0,
            "btts_no": not (hg > 0 and ag > 0),
            "over_05": total > 0,
            "over_15": total > 1,
            "over_25": total > 2,
            "over_35": total > 3,
            "under_05": total < 1,
            "under_15": total < 2,
            "under_25": total < 3,
            "under_35": total < 4,
        }
        return bool(mapping.get(outcome, False))

    # ── MÉTRICAS ──────────────────────────────────────────────────────────────

    def compute_metrics(self, days: int = 30) -> Dict:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = self.df[
            (pd.to_datetime(self.df["date"]) >= cutoff) &
            (self.df["result"] != "PENDING")
        ].copy()

        if df.empty:
            return self._empty_metrics()

        total_bets = len(df)
        green = len(df[df["result"].str.startswith("GREEN")])
        red = len(df[df["result"].str.startswith("RED")])

        total_stake = df["stake_units"].sum()
        total_pl = df["profit_loss"].sum()
        roi = (total_pl / total_stake * 100) if total_stake > 0 else 0

        # Yield = P/L / total apostado
        yield_pct = roi

        # Hit rate
        hit_rate = (green / total_bets * 100) if total_bets > 0 else 0

        # Drawdown máximo
        cumulative = df.sort_values("date")["profit_loss"].cumsum()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max).min()

        # CLV médio realizado
        clv_mean = df["clv_realized"].dropna().mean()

        # Por tier
        by_tier = {}
        for tier in ["Elite", "Forte", "Moderada"]:
            t_df = df[df["tier"] == tier]
            if not t_df.empty:
                t_pl = t_df["profit_loss"].sum()
                t_stake = t_df["stake_units"].sum()
                by_tier[tier] = {
                    "bets": len(t_df),
                    "roi": round(t_pl / t_stake * 100, 2) if t_stake > 0 else 0,
                    "hit_rate": round(len(t_df[t_df["result"].str.startswith("GREEN")]) / len(t_df) * 100, 1),
                }

        return {
            "period_days": days,
            "total_bets": total_bets,
            "green": green,
            "red": red,
            "hit_rate": round(hit_rate, 1),
            "total_stake": round(float(total_stake), 2),
            "total_pl": round(float(total_pl), 2),
            "roi": round(float(roi), 2),
            "yield_pct": round(float(yield_pct), 2),
            "max_drawdown": round(float(drawdown), 2),
            "avg_clv": round(float(clv_mean) if not np.isnan(clv_mean) else 0, 2),
            "by_tier": by_tier,
            "ev_avg": round(df["ev_pct"].mean(), 2),
        }

    def _empty_metrics(self) -> Dict:
        return {
            "period_days": 30, "total_bets": 0, "green": 0, "red": 0,
            "hit_rate": 0, "total_stake": 0, "total_pl": 0,
            "roi": 0, "yield_pct": 0, "max_drawdown": 0,
            "avg_clv": 0, "by_tier": {}, "ev_avg": 0
        }

    def get_recent(self, n: int = 20) -> pd.DataFrame:
        return self.df.tail(n).sort_values("date", ascending=False)

    def get_pending(self) -> pd.DataFrame:
        return self.df[self.df["result"] == "PENDING"]


# ═══════════════════════════════════════════════════════════════════════════════
# LEAGUE PROFILER
# ═══════════════════════════════════════════════════════════════════════════════

class LeagueProfiler:
    """Calcula perfil estatístico de cada liga."""

    def __init__(self, historical_dir: Path = HIST_DIR):
        self.hist_dir = historical_dir

    def compute_profile(self, matches: pd.DataFrame, league_name: str) -> Dict:
        if matches.empty:
            return {}

        total = matches["home_goals"] + matches["away_goals"]
        btts = ((matches["home_goals"] > 0) & (matches["away_goals"] > 0))
        over25 = total > 2

        return {
            "league": league_name,
            "matches_analyzed": len(matches),
            "avg_goals_pg": round(float(total.mean()), 2),
            "avg_home_goals": round(float(matches["home_goals"].mean()), 2),
            "avg_away_goals": round(float(matches["away_goals"].mean()), 2),
            "btts_pct": round(float(btts.mean() * 100), 1),
            "over25_pct": round(float(over25.mean() * 100), 1),
            "over15_pct": round(float((total > 1).mean() * 100), 1),
            "home_win_pct": round(float((matches["home_goals"] > matches["away_goals"]).mean() * 100), 1),
            "draw_pct": round(float((matches["home_goals"] == matches["away_goals"]).mean() * 100), 1),
            "away_win_pct": round(float((matches["away_goals"] > matches["home_goals"]).mean() * 100), 1),
            "avg_cards_pg": round(float(matches.get("total_cards", pd.Series([0]*len(matches))).mean()), 2),
        }

    def profile_all(self, all_matches: pd.DataFrame) -> List[Dict]:
        profiles = []
        for league in all_matches["league_name"].unique():
            lg_df = all_matches[all_matches["league_name"] == league]
            profiles.append(self.compute_profile(lg_df, league))
        return sorted(profiles, key=lambda x: x.get("avg_goals_pg", 0), reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REFEREE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RefereeModel:
    """Modela impacto disciplinar do árbitro."""

    def analyze(self, ref_history: pd.DataFrame) -> Dict:
        if ref_history.empty:
            return {
                "avg_goals": 2.5, "avg_cards": 4.5,
                "home_win_rate": 0.45, "n_matches": 0
            }

        return {
            "referee": ref_history.get("referee", ["N/A"])[0] if "referee" in ref_history else "N/A",
            "n_matches": len(ref_history),
            "avg_goals": round(float(ref_history["total_goals"].mean()), 2),
            "avg_home_goals": round(float(ref_history["home_goals"].mean()), 2),
            "avg_away_goals": round(float(ref_history["away_goals"].mean()), 2),
            "home_win_rate": round(float((ref_history["home_goals"] > ref_history["away_goals"]).mean()), 3),
            "over25_rate": round(float((ref_history["total_goals"] > 2).mean()), 3),
        }
