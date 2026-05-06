# src/models/statistical.py — Modelos estatísticos clássicos
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import factorial
from typing import Tuple, Dict
from src.utils import get_logger

logger = get_logger("models.statistical")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. POISSON SIMPLES
# ═══════════════════════════════════════════════════════════════════════════════

class PoissonModel:
    """Modelo de Dixon-Coles com parâmetros de ataque/defesa por time."""

    def __init__(self):
        self.attack: Dict[str, float] = {}
        self.defense: Dict[str, float] = {}
        self.home_advantage: float = 0.0
        self.rho: float = 0.0  # correlação baixo placar

    def _tau(self, x: int, y: int, mu: float, nu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1 - mu * nu * rho
        elif x == 0 and y == 1:
            return 1 + mu * rho
        elif x == 1 and y == 0:
            return 1 + nu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _log_likelihood(self, params: np.ndarray, data: pd.DataFrame,
                        teams: list) -> float:
        n = len(teams)
        attack = {t: params[i] for i, t in enumerate(teams)}
        defense = {t: params[n + i] for i, t in enumerate(teams)}
        home_adv = params[2 * n]
        rho = params[2 * n + 1]

        ll = 0.0
        for _, row in data.iterrows():
            h, a = row["home_team"], row["away_team"]
            if h not in attack or a not in attack:
                continue
            mu = np.exp(attack[h] + defense[a] + home_adv)
            nu = np.exp(attack[a] + defense[h])
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            tau = self._tau(hg, ag, mu, nu, rho)
            p_h = poisson.pmf(hg, mu)
            p_a = poisson.pmf(ag, nu)
            if p_h > 0 and p_a > 0 and tau > 0:
                ll += np.log(tau) + np.log(p_h) + np.log(p_a)

        return -ll

    def fit(self, data: pd.DataFrame):
        teams = sorted(set(data["home_team"]) | set(data["away_team"]))
        n = len(teams)
        x0 = np.zeros(2 * n + 2)
        x0[:n] = 0.3   # attack init
        x0[n:2*n] = -0.3  # defense init
        x0[2*n] = 0.1     # home advantage
        x0[2*n+1] = -0.1  # rho

        result = minimize(
            self._log_likelihood, x0,
            args=(data, teams),
            method="L-BFGS-B",
            options={"maxiter": 200}
        )

        self.attack = {t: result.x[i] for i, t in enumerate(teams)}
        self.defense = {t: result.x[n + i] for i, t in enumerate(teams)}
        self.home_advantage = result.x[2 * n]
        self.rho = result.x[2 * n + 1]
        logger.info(f"Poisson model fitted. Home advantage: {self.home_advantage:.3f}, rho: {self.rho:.3f}")

    def predict_mu_nu(self, home: str, away: str) -> Tuple[float, float]:
        mu = np.exp(self.attack.get(home, 0) + self.defense.get(away, 0) + self.home_advantage)
        nu = np.exp(self.attack.get(away, 0) + self.defense.get(home, 0))
        return mu, nu

    def predict_matrix(self, home: str, away: str, max_goals: int = 8) -> np.ndarray:
        mu, nu = self.predict_mu_nu(home, away)
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                tau = self._tau(i, j, mu, nu, self.rho)
                matrix[i, j] = tau * poisson.pmf(i, mu) * poisson.pmf(j, nu)
        return matrix / matrix.sum()

    def predict_1x2(self, home: str, away: str) -> Dict[str, float]:
        m = self.predict_matrix(home, away)
        p_home = float(np.tril(m, -1).sum())
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        return {"home": p_home, "draw": p_draw, "away": p_away}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ZERO-INFLATED POISSON
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroInflatedPoisson:
    """ZIP: mistura de Poisson com massa extra em zero."""

    def __init__(self, pi: float = 0.05):
        self.pi = pi  # prob de excesso de zeros

    def pmf(self, k: int, mu: float) -> float:
        if k == 0:
            return self.pi + (1 - self.pi) * np.exp(-mu)
        return (1 - self.pi) * poisson.pmf(k, mu)

    def predict_matrix(self, mu: float, nu: float, max_goals: int = 8) -> np.ndarray:
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = self.pmf(i, mu) * self.pmf(j, nu)
        return matrix / matrix.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ELO DINÂMICO
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicElo:
    def __init__(self, k: float = 32, initial: float = 1500):
        self.ratings: Dict[str, float] = {}
        self.k = k
        self.initial = initial

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home: str, away: str, home_goals: int, away_goals: int):
        ra = self.get_rating(home) + 50  # home advantage
        rb = self.get_rating(away)
        ea = self.expected_score(ra, rb)

        if home_goals > away_goals:
            sa = 1.0
        elif home_goals == away_goals:
            sa = 0.5
        else:
            sa = 0.0

        goal_diff = abs(home_goals - away_goals)
        k_adj = self.k * (1 + 0.1 * min(goal_diff, 5))

        self.ratings[home] = ra + k_adj * (sa - ea)
        self.ratings[away] = rb + k_adj * ((1 - sa) - (1 - ea))

    def predict_1x2(self, home: str, away: str) -> Dict[str, float]:
        ra = self.get_rating(home) + 50
        rb = self.get_rating(away)
        p_home_win = self.expected_score(ra, rb)
        p_away_win = 1 - p_home_win
        # Aproximação de empate via fórmula de Hvattum
        p_draw = 1 - abs(p_home_win - 0.5) * 2 * 0.3
        p_draw = max(0.15, min(0.35, p_draw))
        p_home_win -= p_draw / 2
        p_away_win -= p_draw / 2
        total = p_home_win + p_draw + p_away_win
        return {"home": p_home_win/total, "draw": p_draw/total, "away": p_away_win/total}

    def fit_history(self, matches: pd.DataFrame):
        matches = matches.sort_values("date")
        for _, row in matches.iterrows():
            self.update(row["home_team"], row["away_team"],
                        int(row["home_goals"]), int(row["away_goals"]))
        logger.info(f"Elo fitted on {len(matches)} matches. Teams rated: {len(self.ratings)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GLICKO-2 SIMPLIFICADO
# ═══════════════════════════════════════════════════════════════════════════════

class GlickoModel:
    def __init__(self, mu0: float = 1500, rd0: float = 350, sigma0: float = 0.06):
        self.ratings: Dict[str, Dict] = {}
        self.mu0 = mu0
        self.rd0 = rd0
        self.sigma0 = sigma0

    def get(self, team: str) -> Dict:
        return self.ratings.get(team, {"mu": self.mu0, "rd": self.rd0, "sigma": self.sigma0})

    def _g(self, rd: float) -> float:
        return 1 / np.sqrt(1 + 3 * rd**2 / np.pi**2)

    def _E(self, mu: float, mu_j: float, rd_j: float) -> float:
        return 1 / (1 + np.exp(-self._g(rd_j) * (mu - mu_j)))

    def update(self, team: str, opponent: str, score: float):
        t = self.get(team)
        o = self.get(opponent)
        mu, rd = t["mu"], t["rd"]
        mu_j, rd_j = o["mu"], o["rd"]

        g_rd_j = self._g(rd_j)
        E = self._E(mu, mu_j, rd_j)
        v = 1 / (g_rd_j**2 * E * (1 - E))
        delta = v * g_rd_j * (score - E)

        new_rd = 1 / np.sqrt(1/rd**2 + 1/v)
        new_mu = mu + new_rd**2 * g_rd_j * (score - E)

        self.ratings[team] = {"mu": new_mu, "rd": new_rd, "sigma": t["sigma"]}

    def predict_win_prob(self, home: str, away: str) -> float:
        h = self.get(home)
        a = self.get(away)
        return self._E(h["mu"] + 50, a["mu"], a["rd"])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 10000):
        self.n = n_simulations

    def simulate(self, mu: float, nu: float, use_zip: bool = False,
                 zip_pi: float = 0.05) -> Dict[str, float]:
        rng = np.random.default_rng()

        if use_zip:
            def sample(lam):
                zeros = rng.random(self.n) < zip_pi
                counts = rng.poisson(lam, self.n)
                counts[zeros] = 0
                return counts
            home_goals = sample(mu)
            away_goals = sample(nu)
        else:
            home_goals = rng.poisson(mu, self.n)
            away_goals = rng.poisson(nu, self.n)

        results = {
            "home_win": float(np.mean(home_goals > away_goals)),
            "draw":     float(np.mean(home_goals == away_goals)),
            "away_win": float(np.mean(home_goals < away_goals)),
            "btts":     float(np.mean((home_goals > 0) & (away_goals > 0))),
            "over_05":  float(np.mean(home_goals + away_goals > 0)),
            "over_15":  float(np.mean(home_goals + away_goals > 1)),
            "over_25":  float(np.mean(home_goals + away_goals > 2)),
            "over_35":  float(np.mean(home_goals + away_goals > 3)),
            "avg_goals": float(np.mean(home_goals + away_goals)),
            "avg_home_goals": float(np.mean(home_goals)),
            "avg_away_goals": float(np.mean(away_goals)),
            "clean_sheet_home": float(np.mean(away_goals == 0)),
            "clean_sheet_away": float(np.mean(home_goals == 0)),
        }

        # Score distribution (top 10 scorelines)
        total = home_goals + away_goals
        for goals in range(7):
            results[f"total_{goals}"] = float(np.mean(total == goals))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BAYESIAN UPDATER
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianUpdater:
    """Atualiza prior de Poisson com resultados recentes via conjugado Gamma."""

    def __init__(self, prior_alpha: float = 3.0, prior_beta: float = 1.0):
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, observations: list) -> Tuple[float, float]:
        """Posterior Gamma após observações."""
        n = len(observations)
        total = sum(observations)
        post_alpha = self.alpha + total
        post_beta = self.beta + n
        return post_alpha, post_beta

    def posterior_mean(self, observations: list) -> float:
        a, b = self.update(observations)
        return a / b

    def predict_lambda(self, team_goals: list, league_avg: float = 1.35) -> float:
        """Combina dados do time com média da liga."""
        if not team_goals:
            return league_avg
        observed = self.posterior_mean(team_goals)
        weight = min(len(team_goals) / 10, 1.0)
        return weight * observed + (1 - weight) * league_avg
