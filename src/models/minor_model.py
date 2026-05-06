# src/models/minor_model.py
# Modelo estatístico adaptado para ligas menores com dados esparsos.
# Estratégia: Bayesian shrinkage em direção ao perfil da liga.
# Mercados gerados: 1X2, Over/Under FT (0.5/1.5/2.5/3.5), Over/Under HT (0.5/1.5).

import numpy as np
from scipy.stats import poisson
from typing import Dict, List, Optional, Tuple
from src.utils import get_logger, implied_prob, expected_value, kelly_fraction, load_config

logger = get_logger("models.minor")
cfg = load_config()


class MinorLeagueModel:
    """
    Modelo simplificado para ligas com poucos dados históricos.

    Abordagem:
      1. Prior = perfil da liga (média gols, win rates históricas)
      2. Dados do time → atualiza via Bayesian shrinkage
      3. Gera lambdas (mu_home, mu_away) → Poisson + Monte Carlo
      4. Calcula HT separadamente via ratio HT/FT da liga
    """

    def __init__(self, league_profile: Dict):
        self.lp = league_profile
        self.n_sim = cfg["models"]["monte_carlo_simulations"]

    # ── Estima lambdas ───────────────────────────────────────────────────────

    def _shrink(self, team_obs: List[float], league_prior: float,
                weight_obs: float = None) -> float:
        """
        Combina observações do time com prior da liga.
        Quanto menos dados, mais pesa o prior.
        """
        n = len(team_obs)
        if n == 0:
            return league_prior
        w = weight_obs if weight_obs is not None else min(n / 10.0, 0.85)
        obs_mean = float(np.mean(team_obs))
        return w * obs_mean + (1 - w) * league_prior

    def estimate_lambdas(
        self,
        home_goals_scored:   List[float],
        home_goals_conceded: List[float],
        away_goals_scored:   List[float],
        away_goals_conceded: List[float],
        home_advantage_adj:  float = 0.10,
    ) -> Tuple[float, float]:
        """
        Estima lambda (esperança de gols) para cada time.
        mu_home = ataque_home * defesa_away * fator_casa
        mu_away = ataque_away * defesa_home
        """
        league_avg = self.lp.get("avg_goals_ft", 2.45)
        league_home = self.lp.get("avg_home_goals", 1.35)
        league_away = self.lp.get("avg_away_goals", 1.10)

        # Ataque e defesa com shrinkage
        home_att  = self._shrink(home_goals_scored,   league_home)
        home_def  = self._shrink(home_goals_conceded, league_away)
        away_att  = self._shrink(away_goals_scored,   league_away)
        away_def  = self._shrink(away_goals_conceded, league_home)

        # Normaliza por média da liga (Dixon-Coles style, simplificado)
        home_att_norm = home_att / max(league_home, 0.01)
        home_def_norm = home_def / max(league_away, 0.01)
        away_att_norm = away_att / max(league_away, 0.01)
        away_def_norm = away_def / max(league_home, 0.01)

        mu_home = home_att_norm * away_def_norm * league_home * (1 + home_advantage_adj)
        mu_away = away_att_norm * home_def_norm * league_away

        # Clipping razoável
        mu_home = float(np.clip(mu_home, 0.3, 4.5))
        mu_away = float(np.clip(mu_away, 0.3, 4.0))

        return mu_home, mu_away

    # ── HT lambda ────────────────────────────────────────────────────────────

    def estimate_ht_lambdas(self, mu_home_ft: float, mu_away_ft: float) -> Tuple[float, float]:
        """
        Estima lambdas para o HT.
        Usa ratio HT/FT da liga — gols não são uniformes nos 45min.
        Tipicamente HT ≈ 40-45% dos gols FT.
        """
        ratio = self.lp.get("ht_ft_ratio", 0.43)
        return mu_home_ft * ratio, mu_away_ft * ratio

    # ── Monte Carlo ──────────────────────────────────────────────────────────

    def simulate(self, mu_home: float, mu_away: float,
                 mu_ht_home: float, mu_ht_away: float) -> Dict:
        """
        10.000 simulações Poisson para FT e HT.
        Retorna probabilidades de todos os mercados foco.
        """
        rng = np.random.default_rng()

        # Full Time
        hg = rng.poisson(mu_home, self.n_sim)
        ag = rng.poisson(mu_away, self.n_sim)
        total_ft = hg + ag

        # Half Time
        hg_ht = rng.poisson(mu_ht_home, self.n_sim)
        ag_ht = rng.poisson(mu_ht_away, self.n_sim)
        total_ht = hg_ht + ag_ht

        return {
            # ── 1X2 FT ──
            "home_win":      float(np.mean(hg > ag)),
            "draw":          float(np.mean(hg == ag)),
            "away_win":      float(np.mean(ag > hg)),

            # ── Over/Under FT ──
            "over_05_ft":    float(np.mean(total_ft > 0)),
            "under_05_ft":   float(np.mean(total_ft <= 0)),
            "over_15_ft":    float(np.mean(total_ft > 1)),
            "under_15_ft":   float(np.mean(total_ft <= 1)),
            "over_25_ft":    float(np.mean(total_ft > 2)),
            "under_25_ft":   float(np.mean(total_ft <= 2)),
            "over_35_ft":    float(np.mean(total_ft > 3)),
            "under_35_ft":   float(np.mean(total_ft <= 3)),
            "over_45_ft":    float(np.mean(total_ft > 4)),

            # ── BTTS ──
            "btts_yes":      float(np.mean((hg > 0) & (ag > 0))),
            "btts_no":       float(np.mean(~((hg > 0) & (ag > 0)))),

            # ── Over/Under HT ──
            "over_05_ht":    float(np.mean(total_ht > 0)),
            "under_05_ht":   float(np.mean(total_ht <= 0)),
            "over_15_ht":    float(np.mean(total_ht > 1)),
            "under_15_ht":   float(np.mean(total_ht <= 1)),
            "over_25_ht":    float(np.mean(total_ht > 2)),
            "under_25_ht":   float(np.mean(total_ht <= 2)),

            # ── 1X2 HT ──
            "ht_home_win":   float(np.mean(hg_ht > ag_ht)),
            "ht_draw":       float(np.mean(hg_ht == ag_ht)),
            "ht_away_win":   float(np.mean(ag_ht > hg_ht)),

            # ── Diagnóstico ──
            "mu_home_ft":    mu_home,
            "mu_away_ft":    mu_away,
            "mu_home_ht":    mu_ht_home,
            "mu_away_ht":    mu_ht_away,
            "avg_goals_ft":  float(np.mean(total_ft)),
            "avg_goals_ht":  float(np.mean(total_ht)),
        }

    # ── Análise de mercados ───────────────────────────────────────────────────

    def analyze_markets(self, sim: Dict, odds: Dict,
                        fixture_info: Dict) -> List[Dict]:
        """
        Gera lista de análises de mercado comparando probabilidade
        do modelo vs probabilidade implícita da Bet365.
        Foco: 1X2, Over FT, Over HT.
        """
        results = []
        fix = fixture_info

        # ── 1X2 FT ──────────────────────────────────────────────
        market_map_ft = [
            ("1X2 — Vitória Casa",   "home",     "home_win",   odds.get("home_win")),
            ("1X2 — Empate",         "draw",     "draw",       odds.get("draw")),
            ("1X2 — Vitória Fora",   "away",     "away_win",   odds.get("away_win")),
        ]
        for label, outcome, sim_key, odd in market_map_ft:
            if odd and odd > 1.0:
                results.append(self._build(label, outcome, odd, sim[sim_key], fix, "FT"))

        # ── Over/Under FT ────────────────────────────────────────
        over_ft_map = [
            ("Over 0.5 FT",   "over_05_ft",  "over_05"),
            ("Under 0.5 FT",  "under_05_ft", "under_05"),
            ("Over 1.5 FT",   "over_15_ft",  "over_15"),
            ("Under 1.5 FT",  "under_15_ft", "under_15"),
            ("Over 2.5 FT",   "over_25_ft",  "over_25"),
            ("Under 2.5 FT",  "under_25_ft", "under_25"),
            ("Over 3.5 FT",   "over_35_ft",  "over_35"),
            ("Under 3.5 FT",  "under_35_ft", "under_35"),
        ]
        for label, sim_key, odds_key in over_ft_map:
            # Normaliza chave de odds (ex: "over_25" → busca "over_25" ou "over_250")
            odd = odds.get(odds_key) or odds.get(odds_key.replace(".", "") + "0")
            if odd and odd > 1.0:
                results.append(self._build(label, sim_key, odd, sim[sim_key], fix, "FT"))

        # ── Over/Under HT ────────────────────────────────────────
        over_ht_map = [
            ("Over 0.5 HT",  "over_05_ht",  "ht_over_05"),
            ("Under 0.5 HT", "under_05_ht", "ht_under_05"),
            ("Over 1.5 HT",  "over_15_ht",  "ht_over_15"),
            ("Under 1.5 HT", "under_15_ht", "ht_under_15"),
            ("Over 2.5 HT",  "over_25_ht",  "ht_over_25"),
        ]
        for label, sim_key, odds_key in over_ht_map:
            odd = odds.get(odds_key)
            if odd and odd > 1.0:
                results.append(self._build(label, sim_key, odd, sim[sim_key], fix, "HT"))

        # ── 1X2 HT ──────────────────────────────────────────────
        ht_1x2_map = [
            ("1X2 HT — Casa",  "ht_home", "ht_home_win",  odds.get("ht_home_win")),
            ("1X2 HT — Empate","ht_draw", "ht_draw",      odds.get("ht_draw")),
            ("1X2 HT — Fora",  "ht_away", "ht_away_win",  odds.get("ht_away_win")),
        ]
        for label, outcome, sim_key, odd in ht_1x2_map:
            if odd and odd > 1.0:
                results.append(self._build(label, outcome, odd, sim[sim_key], fix, "HT"))

        return results

    def _build(self, market_label: str, outcome: str,
               odd: float, model_prob: float,
               fix: Dict, period: str) -> Dict:
        imp = implied_prob(odd)
        ev = expected_value(model_prob, odd)
        kelly = kelly_fraction(model_prob, odd)
        return {
            "market":       market_label,
            "outcome":      outcome,
            "period":       period,
            "bet365_odd":   round(odd, 3),
            "implied_prob": round(imp, 4),
            "model_prob":   round(model_prob, 4),
            "ev":           round(ev, 4),
            "ev_pct":       round(ev * 100, 2),
            "kelly_pct":    round(kelly * 100, 2),
            "edge":         round(model_prob - imp, 4),
            # Contexto
            "mu_home_ft":   fix.get("mu_home_ft", 0),
            "mu_away_ft":   fix.get("mu_away_ft", 0),
            "mu_home_ht":   fix.get("mu_home_ht", 0),
            "mu_away_ht":   fix.get("mu_away_ht", 0),
        }


# ── Filtro de value bets para ligas menores ───────────────────────────────────

VB = cfg["value_betting"]


def filter_minor_value_bets(bets: List[Dict]) -> List[Dict]:
    """
    Ligas menores: threshold de EV um pouco mais conservador
    pois o modelo tem menos dados históricos.
    """
    filtered = []
    for b in bets:
        ev  = b.get("ev_pct", 0)
        prob = b.get("model_prob", 0)
        odd  = b.get("bet365_odd", 1.0)

        # Thresholds ligeiramente mais altos para compensar incerteza
        if ev >= VB["min_ev_pct"] + 2 and prob >= VB["min_probability"] and odd >= VB["min_odds"]:
            tier, emoji = _tier(ev, prob)
            b["tier"] = tier
            b["tier_emoji"] = emoji
            b["data_quality"] = "⚠️ Liga menor — modelo com dados limitados"
            filtered.append(b)

    return sorted(filtered, key=lambda x: x["ev_pct"], reverse=True)


def _tier(ev: float, prob: float) -> Tuple[str, str]:
    tiers = VB["tiers"]
    if ev >= tiers["elite"]["ev_min"] and prob >= tiers["elite"]["prob_min"]:
        return "Elite", "🔥"
    elif ev >= tiers["strong"]["ev_min"] and prob >= tiers["strong"]["prob_min"]:
        return "Forte", "⚡"
    return "Moderada", "🟡"
