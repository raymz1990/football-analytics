# src/models/ml_models.py — Machine Learning models
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
from src.utils import get_logger, load_config

logger = get_logger("models.ml")
cfg = load_config()
CACHE_DIR = Path(cfg["paths"]["models_cache"])


def _feature_columns() -> List[str]:
    return [
        # Forma recente
        "home_form_5", "away_form_5", "home_form_10", "away_form_10",
        # Gols
        "home_goals_scored_avg", "home_goals_conceded_avg",
        "away_goals_scored_avg", "away_goals_conceded_avg",
        # xG
        "home_xg_avg", "away_xg_avg",
        "home_xg_conceded_avg", "away_xg_conceded_avg",
        # Elo
        "home_elo", "away_elo", "elo_diff",
        # Casa/Fora
        "home_home_win_rate", "away_away_win_rate",
        "home_home_goals_avg", "away_away_goals_avg",
        # Lesões
        "home_injured_count", "away_injured_count",
        "home_injury_impact", "away_injury_impact",
        # Árbitro
        "ref_avg_cards", "ref_avg_goals", "ref_home_win_rate",
        # Clima
        "wind_kmh", "temp_c",
        # Descanso
        "home_days_rest", "away_days_rest",
        # Liga
        "league_avg_goals", "league_btts_rate", "league_over25_rate",
    ]


def build_feature_vector(match_data: Dict) -> np.ndarray:
    cols = _feature_columns()
    return np.array([match_data.get(c, 0.0) for c in cols], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticModel:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial"))
        ])
        self.classes_ = ["home", "draw", "away"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        cv = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")
        logger.info(f"Logistic CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        proba = self.model.predict_proba(X)[0]
        return {c: float(p) for c, p in zip(self.classes_, proba)}

    def save(self): joblib.dump(self.model, CACHE_DIR / "logistic.pkl")
    def load(self): self.model = joblib.load(CACHE_DIR / "logistic.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=20,
            n_jobs=-1, random_state=42, class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.classes_ = ["home", "draw", "away"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        cv = cross_val_score(self.model, X_scaled, y, cv=5, scoring="accuracy")
        logger.info(f"RandomForest CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        return {c: float(p) for c, p in zip(self.classes_, proba)}

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=_feature_columns()
        ).sort_values(ascending=False)

    def save(self):
        joblib.dump((self.model, self.scaler), CACHE_DIR / "rf.pkl")

    def load(self):
        self.model, self.scaler = joblib.load(CACHE_DIR / "rf.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_map = {"home": 0, "draw": 1, "away": 2}
        self.inv_label = {0: "home", 1: "draw", 2: "away"}

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        y_enc = np.array([self.label_map[v] for v in y])
        self.model.fit(X_s, y_enc, eval_set=[(X_s, y_enc)], verbose=False)
        logger.info("XGBoost fitted")

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        X_s = self.scaler.transform(X)
        proba = self.model.predict_proba(X_s)[0]
        return {self.inv_label[i]: float(p) for i, p in enumerate(proba)}

    def save(self):
        self.model.save_model(str(CACHE_DIR / "xgb.json"))
        joblib.dump(self.scaler, CACHE_DIR / "xgb_scaler.pkl")

    def load(self):
        self.model.load_model(str(CACHE_DIR / "xgb.json"))
        self.scaler = joblib.load(CACHE_DIR / "xgb_scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════════════

class LightGBMModel:
    def __init__(self):
        self.params = {
            "objective": "multiclass", "num_class": 3,
            "n_estimators": 500, "max_depth": 6,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "random_state": 42,
            "n_jobs": -1, "verbose": -1
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.scaler = StandardScaler()
        self.label_map = {"home": 0, "draw": 1, "away": 2}
        self.inv_label = {0: "home", 1: "draw", 2: "away"}

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        y_enc = np.array([self.label_map[v] for v in y])
        self.model.fit(X_s, y_enc)
        logger.info("LightGBM fitted")

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        X_s = self.scaler.transform(X)
        proba = self.model.predict_proba(X_s)[0]
        return {self.inv_label[i]: float(p) for i, p in enumerate(proba)}

    def save(self):
        joblib.dump((self.model, self.scaler), CACHE_DIR / "lgbm.pkl")

    def load(self):
        self.model, self.scaler = joblib.load(CACHE_DIR / "lgbm.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """Combina todos os modelos com pesos aprendidos."""

    MODELS = ["poisson", "zip", "monte_carlo", "elo", "glicko",
              "bayesian", "logistic", "random_forest", "xgboost", "lightgbm"]

    WEIGHTS = {
        "poisson": 0.15, "zip": 0.10, "monte_carlo": 0.10,
        "elo": 0.08, "glicko": 0.07, "bayesian": 0.10,
        "logistic": 0.08, "random_forest": 0.12,
        "xgboost": 0.10, "lightgbm": 0.10
    }

    def blend(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Blenda predições de múltiplos modelos com pesos."""
        outcomes = ["home", "draw", "away"]
        blended = {o: 0.0 for o in outcomes}
        total_w = 0.0

        for model_name, probs in predictions.items():
            w = self.WEIGHTS.get(model_name, 0.05)
            for o in outcomes:
                blended[o] += w * probs.get(o, 0.333)
            total_w += w

        # Normaliza
        for o in outcomes:
            blended[o] /= total_w

        return blended

    def convergence_score(self, predictions: Dict[str, Dict[str, float]],
                          outcome: str, threshold: float = 0.05) -> int:
        """Conta quantos modelos concordam sobre o resultado."""
        ensemble = self.blend(predictions)
        best = ensemble[outcome]
        count = 0
        for model_name, probs in predictions.items():
            if abs(probs.get(outcome, 0) - best) < threshold:
                count += 1
        return count

    def std_across_models(self, predictions: Dict[str, Dict[str, float]],
                          outcome: str) -> float:
        vals = [p.get(outcome, 0) for p in predictions.values()]
        return float(np.std(vals))


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM STYLE CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

class TeamStyleClusterer:
    """Agrupa times por estilo tático via K-Means."""

    STYLES = {
        0: "Posse & Construção",
        1: "Pressão Alta",
        2: "Contra-Ataque",
        3: "Defensivo Compacto",
        4: "Direto & Físico",
    }

    def __init__(self, n_clusters: int = 5):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.team_clusters: Dict[str, int] = {}

    def fit(self, team_stats: pd.DataFrame):
        """
        team_stats: DataFrame com colunas de estilo por time
        Exemplos: shots_pg, possession_avg, passes_pg, press_rate, etc.
        """
        feature_cols = [c for c in team_stats.columns if c != "team"]
        X = self.scaler.fit_transform(team_stats[feature_cols].fillna(0))
        labels = self.kmeans.fit_predict(X)
        for i, team in enumerate(team_stats["team"]):
            self.team_clusters[team] = int(labels[i])
        logger.info(f"Clustered {len(self.team_clusters)} teams into {self.kmeans.n_clusters} styles")

    def get_style(self, team: str) -> str:
        cluster = self.team_clusters.get(team, -1)
        return self.STYLES.get(cluster, "Desconhecido")

    def get_cluster(self, team: str) -> int:
        return self.team_clusters.get(team, -1)
