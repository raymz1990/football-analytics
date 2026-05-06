# src/utils.py — Shared utilities
import yaml
import logging
from pathlib import Path
from datetime import datetime
import pytz

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    return logging.getLogger(name)

def today_str(tz: str = "America/Sao_Paulo") -> str:
    return datetime.now(pytz.timezone(tz)).strftime("%Y-%m-%d")

def ensure_dirs(cfg: dict):
    for key in ["data_raw", "data_processed", "data_historical", "dashboard_output", "models_cache"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

def implied_prob(odd: float) -> float:
    """Converte odd decimal para probabilidade implícita."""
    if odd <= 1.0:
        return 1.0
    return 1.0 / odd

def expected_value(true_prob: float, odd: float) -> float:
    """EV = (prob_real * odd) - 1"""
    return (true_prob * odd) - 1.0

def kelly_fraction(true_prob: float, odd: float, kelly_pct: float = 0.25) -> float:
    """Critério de Kelly fracionado."""
    b = odd - 1
    q = 1 - true_prob
    k = (b * true_prob - q) / b
    return max(0.0, k * kelly_pct)
