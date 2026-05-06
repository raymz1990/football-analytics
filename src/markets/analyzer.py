# src/markets/analyzer.py — Análise de mercados e value betting
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from src.utils import get_logger, implied_prob, expected_value, kelly_fraction, load_config

logger = get_logger("markets.analyzer")
cfg = load_config()
vb_cfg = cfg["value_betting"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValueBet:
    fixture_id: str
    home_team: str
    away_team: str
    league: str
    kickoff: str
    market: str
    outcome: str
    bet365_odd: float
    implied_prob: float
    model_prob: float
    ev_pct: float
    kelly_pct: float
    convergence: int
    model_std: float
    tier: str
    tier_emoji: str
    clv_expected: float
    models_detail: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET CALCULATORS
# ═══════════════════════════════════════════════════════════════════════════════

class MarketAnalyzer:

    def __init__(self, ensemble_predictions: Dict, monte_carlo: Dict,
                 odds_row: Dict, fixture_info: Dict):
        self.ens = ensemble_predictions   # {"home": p, "draw": p, "away": p}
        self.mc = monte_carlo             # Monte Carlo results dict
        self.odds = odds_row              # Bet365 odds by market/outcome
        self.fix = fixture_info

    # ── 1X2 ──────────────────────────────────────────────────────────────────

    def analyze_1x2(self) -> List[Dict]:
        results = []
        mapping = {
            "home": ("Vitória Casa", self.odds.get("home_win")),
            "draw": ("Empate", self.odds.get("draw")),
            "away": ("Vitória Fora", self.odds.get("away_win")),
        }
        for outcome, (label, odd) in mapping.items():
            if not odd or odd <= 1.0:
                continue
            model_p = self.ens.get(outcome, 0.0)
            results.append(self._build(f"1X2 — {label}", outcome, odd, model_p))
        return results

    # ── BTTS ─────────────────────────────────────────────────────────────────

    def analyze_btts(self) -> List[Dict]:
        results = []
        btts_yes_odd = self.odds.get("btts_yes")
        btts_no_odd = self.odds.get("btts_no")
        if btts_yes_odd:
            results.append(self._build("BTTS — Sim", "btts_yes", btts_yes_odd, self.mc.get("btts", 0)))
        if btts_no_odd:
            results.append(self._build("BTTS — Não", "btts_no", btts_no_odd, 1 - self.mc.get("btts", 0)))
        return results

    # ── OVER/UNDER ────────────────────────────────────────────────────────────

    def analyze_over_under(self) -> List[Dict]:
        results = []
        lines = [
            (0.5, "over_05", "under_05"),
            (1.5, "over_15", "under_15"),
            (2.5, "over_25", "under_25"),
            (3.5, "over_35", "under_35"),
        ]
        for line, over_key, under_key in lines:
            over_odd = self.odds.get(f"over_{int(line*10)}")
            under_odd = self.odds.get(f"under_{int(line*10)}")
            over_p = self.mc.get(f"over_{over_key.split('_')[1]}", 0)
            if over_odd:
                results.append(self._build(f"Over {line}", over_key, over_odd, over_p))
            if under_odd:
                results.append(self._build(f"Under {line}", under_key, under_odd, 1 - over_p))
        return results

    # ── ASIAN HANDICAP ────────────────────────────────────────────────────────

    def analyze_asian_handicap(self, handicaps: List[float] = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]) -> List[Dict]:
        results = []
        mc_home = self.mc.get("avg_home_goals", 1.4)
        mc_away = self.mc.get("avg_away_goals", 1.1)

        for hc in handicaps:
            odd = self.odds.get(f"ah_{hc}")
            if not odd:
                continue
            # Prob estimada via MC: home ganha considerando handicap
            mc_p = self.mc.get("home_win", 0.45) if hc < 0 else self.mc.get("away_win", 0.30)
            results.append(self._build(f"AH Casa {hc:+.1f}", f"ah_{hc}", odd, mc_p))
        return results

    # ── CORNERS ──────────────────────────────────────────────────────────────

    def analyze_corners(self) -> List[Dict]:
        results = []
        # Corners estimados via estilo de jogo (proxy simples)
        home_corners_avg = self.fix.get("home_corners_avg", 5.5)
        away_corners_avg = self.fix.get("away_corners_avg", 4.5)
        expected_corners = home_corners_avg + away_corners_avg

        for line in [8.5, 9.5, 10.5, 11.5]:
            # Aproximação Poisson para corners
            from scipy.stats import poisson
            p_over = 1 - poisson.cdf(int(line), expected_corners)
            odd = self.odds.get(f"corners_over_{int(line*10)}")
            if odd:
                results.append(self._build(f"Corners Over {line}", f"corners_over_{line}", odd, float(p_over)))
        return results

    # ── CARDS ─────────────────────────────────────────────────────────────────

    def analyze_cards(self) -> List[Dict]:
        results = []
        ref_avg = self.fix.get("ref_avg_cards", 4.5)
        odd_over = self.odds.get("cards_over_45")
        odd_under = self.odds.get("cards_under_45")

        from scipy.stats import poisson
        p_over = 1 - poisson.cdf(4, ref_avg)
        if odd_over:
            results.append(self._build("Cartões Over 4.5", "cards_over_45", odd_over, float(p_over)))
        if odd_under:
            results.append(self._build("Cartões Under 4.5", "cards_under_45", odd_under, 1 - float(p_over)))
        return results

    # ── BUILDER ──────────────────────────────────────────────────────────────

    def _build(self, market_label: str, outcome_key: str,
               odd: float, model_prob: float) -> Dict:
        imp = implied_prob(odd)
        ev = expected_value(model_prob, odd)
        kelly = kelly_fraction(model_prob, odd)
        return {
            "market": market_label,
            "outcome": outcome_key,
            "bet365_odd": odd,
            "implied_prob": imp,
            "model_prob": model_prob,
            "ev": ev,
            "ev_pct": ev * 100,
            "kelly_pct": kelly * 100,
            "edge": model_prob - imp,
        }

    def run_all(self) -> List[Dict]:
        all_bets = []
        all_bets.extend(self.analyze_1x2())
        all_bets.extend(self.analyze_btts())
        all_bets.extend(self.analyze_over_under())
        all_bets.extend(self.analyze_cards())
        return all_bets


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE BET FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def classify_tier(ev_pct: float, prob: float, convergence: int) -> Tuple[str, str]:
    tiers = vb_cfg["tiers"]
    if (ev_pct >= tiers["elite"]["ev_min"] and
            prob >= tiers["elite"]["prob_min"] and
            convergence >= tiers["elite"]["convergence_min"]):
        return "Elite", "🔥"
    elif (ev_pct >= tiers["strong"]["ev_min"] and
          prob >= tiers["strong"]["prob_min"] and
          convergence >= tiers["strong"]["convergence_min"]):
        return "Forte", "⚡"
    elif (ev_pct >= tiers["moderate"]["ev_min"] and
          prob >= tiers["moderate"]["prob_min"] and
          convergence >= tiers["moderate"]["convergence_min"]):
        return "Moderada", "🟡"
    return "Sem valor", "❌"


def filter_value_bets(bets: List[Dict]) -> List[Dict]:
    """Filtra apenas apostas com valor real."""
    filtered = []
    for b in bets:
        ev = b.get("ev_pct", 0)
        prob = b.get("model_prob", 0)
        odd = b.get("bet365_odd", 1.0)
        conv = b.get("convergence", 0)

        if (ev >= vb_cfg["min_ev_pct"] and
                prob >= vb_cfg["min_probability"] and
                odd >= vb_cfg["min_odds"] and
                conv >= vb_cfg["min_model_convergence"]):
            tier, emoji = classify_tier(ev, prob, conv)
            b["tier"] = tier
            b["tier_emoji"] = emoji
            filtered.append(b)

    return sorted(filtered, key=lambda x: x["ev_pct"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ODDS MOVEMENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class OddsMovementAnalyzer:
    def __init__(self):
        pass

    def analyze(self, movement_history: List[Dict]) -> Dict:
        """Analisa movimentação de odds."""
        if not movement_history:
            return {}

        prices = [m["price"] for m in movement_history]
        times = [m["timestamp"] for m in movement_history]

        return {
            "opening_price": prices[0],
            "current_price": prices[-1],
            "movement_pct": ((prices[-1] - prices[0]) / prices[0]) * 100,
            "direction": "DOWN" if prices[-1] < prices[0] else "UP",
            "max_price": max(prices),
            "min_price": min(prices),
            "n_changes": len(prices) - 1,
            "sharp_money": prices[-1] < prices[0] * 0.92,  # -8% = sharp money
            "clv_opportunity": prices[0] > prices[-1],
        }

    def estimate_clv(self, opening: float, closing: float) -> float:
        """Closing Line Value estimado."""
        if closing <= 1.0:
            return 0.0
        return ((opening / closing) - 1) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER PROPS ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class PlayerPropsAnalyzer:
    PROP_MARKETS = ["shots_total", "shots_on_target", "tackles", "fouls_committed"]

    def analyze_player(self, player_name: str, stat_key: str,
                       line: float, recent_5: List[float],
                       recent_10: List[float], home_avg: float,
                       away_avg: float, is_home: bool,
                       matchup_allowed: float) -> Dict:
        """Analisa prop de jogador."""
        if not recent_5:
            return {}

        avg_5 = np.mean(recent_5)
        avg_10 = np.mean(recent_10) if recent_10 else avg_5
        context_avg = home_avg if is_home else away_avg

        # Modelo simples: média ponderada
        est_value = 0.4 * avg_5 + 0.3 * avg_10 + 0.2 * context_avg + 0.1 * matchup_allowed

        # Trend
        if len(recent_5) >= 3:
            trend = np.polyfit(range(len(recent_5)), recent_5, 1)[0]
        else:
            trend = 0.0

        from scipy.stats import poisson
        p_over = float(1 - poisson.cdf(int(line), est_value))
        p_under = 1 - p_over

        return {
            "player": player_name,
            "stat": stat_key,
            "line": line,
            "est_value": round(est_value, 2),
            "avg_5": round(avg_5, 2),
            "avg_10": round(avg_10, 2),
            "context_avg": round(context_avg, 2),
            "matchup_allowed": round(matchup_allowed, 2),
            "trend": round(trend, 3),
            "trend_label": "📈 Alta" if trend > 0.1 else "📉 Baixa" if trend < -0.1 else "➡️ Estável",
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "hot": p_over > 0.65 or p_under > 0.65,
        }
