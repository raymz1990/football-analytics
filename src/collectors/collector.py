# src/collectors/collector.py
# Fontes: football-data.org (fixtures) + The Odds API (odds)
# Fallback: odds estimadas via Poisson quando The Odds API não disponível

import os, time, requests, pandas as pd
from pathlib import Path
from src.utils import load_config, get_logger, today_str

logger   = get_logger("collector")
cfg      = load_config()
FD_KEY   = os.getenv("FOOTBALL_DATA_API_KEY", "")
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
HEADERS_FD = {"X-Auth-Token": FD_KEY}
BASE_FD    = cfg["apis"]["football_data"]["base_url"]
BASE_ODDS  = cfg["apis"]["odds_api"]["base_url"]

FD_COMPETITIONS = [
    ("PL",  "Premier League",      "England"),
    ("PD",  "La Liga",             "Spain"),
    ("BL1", "Bundesliga",          "Germany"),
    ("SA",  "Serie A",             "Italy"),
    ("FL1", "Ligue 1",             "France"),
    ("CL",  "Champions League",    "Europe"),
    ("PPL", "Primeira Liga",       "Portugal"),
    ("DED", "Eredivisie",          "Netherlands"),
    ("BSA", "Brasileirão Série A", "Brazil"),
    ("ELC", "Championship",        "England"),
]

ODDS_SPORT_KEYS = [
    "soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a",
    "soccer_germany_bundesliga", "soccer_france_ligue_one",
    "soccer_brazil_campeonato", "soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie", "soccer_turkey_super_league",
    "soccer_uefa_champs_league", "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league", "soccer_usa_mls",
    "soccer_argentina_primera_division", "soccer_efl_champ",
    "soccer_spain_segunda_division", "soccer_mexico_ligamx",
    "soccer_belgium_first_div", "soccer_greece_super_league",
    "soccer_scotland_premiership",
]


def _get(url, headers={}, params=None, retries=3, label="") -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 429:
                logger.warning(f"Rate limit — sleeping 65s ({label})")
                time.sleep(65); continue
            if r.status_code in (401, 403):
                logger.error(f"Auth {r.status_code} — {label}")
                return {}
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[{i+1}/{retries}] {label}: {e}")
            if i < retries - 1: time.sleep(2 ** i)
    return {}


# ── football-data.org ─────────────────────────────────────────────────────────

def fetch_fixtures_today() -> pd.DataFrame:
    date = today_str()
    logger.info(f"[football-data.org] {date} | key: {'✅' if FD_KEY else '❌'}")
    rows = []
    for code, name, country in FD_COMPETITIONS:
        data    = _get(f"{BASE_FD}/competitions/{code}/matches",
                       HEADERS_FD, {"dateFrom": date, "dateTo": date}, label=f"FD/{code}")
        matches = data.get("matches", [])
        logger.info(f"  {code}: {len(matches)} matches")
        for m in matches:
            if m.get("status") not in ("SCHEDULED","TIMED","IN_PLAY","PAUSED","LIVE"):
                continue
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            rows.append({
                "fixture_id":  m["id"],
                "date":        m.get("utcDate", ""),
                "league_id":   code,
                "league_name": name,
                "country":     country,
                "home_team":   home.get("shortName") or home.get("name","?"),
                "home_id":     home.get("id", 0),
                "away_team":   away.get("shortName") or away.get("name","?"),
                "away_id":     away.get("id", 0),
                "venue": "", "referee": "",
                "status":      m.get("status",""),
                "minor_focus": False,
                "source":      "football-data.org",
            })
        time.sleep(6.5)
    df = pd.DataFrame(rows)
    logger.info(f"[football-data.org] Total: {len(df)} fixtures")
    return df


# ── The Odds API ──────────────────────────────────────────────────────────────

def _check_odds_quota() -> bool:
    """Verifica se a chave ainda tem créditos."""
    r = _get(f"{BASE_ODDS}/sports", params={"apiKey": ODDS_KEY}, label="OddsAPI/sports")
    if not r:
        logger.warning("ODDS_API_KEY: inválida, expirada ou sem créditos")
        return False
    logger.info(f"ODDS_API_KEY: ✅ ({len(r)} sports disponíveis)")
    return True


def fetch_odds_bet365(skip_if_no_key: bool = True) -> pd.DataFrame:
    if not ODDS_KEY:
        logger.warning("ODDS_API_KEY não configurada")
        return pd.DataFrame()

    if not _check_odds_quota():
        return pd.DataFrame()

    logger.info("[The Odds API] Buscando odds Bet365...")
    rows = []
    for sk in ODDS_SPORT_KEYS:
        data = _get(f"{BASE_ODDS}/sports/{sk}/odds", params={
            "apiKey": ODDS_KEY, "regions": "eu,uk",
            "markets": "h2h,totals", "bookmakers": "bet365",
            "dateFormat": "iso",
        }, label=f"Odds/{sk}")
        if not isinstance(data, list) or not data:
            continue
        for ev in data:
            home, away = ev.get("home_team",""), ev.get("away_team","")
            for bm in ev.get("bookmakers",[]):
                if bm["key"] != "bet365": continue
                for mkt in bm.get("markets",[]):
                    for o in mkt.get("outcomes",[]):
                        rows.append({
                            "sport_key": sk, "event_id": ev["id"],
                            "home_team": home, "away_team": away,
                            "commence_time": ev.get("commence_time",""),
                            "market": mkt["key"], "outcome": o["name"],
                            "price": float(o["price"]), "point": o.get("point"),
                        })
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    logger.info(f"[The Odds API] {len(df)} odds records")
    return df


def build_odds_map(odds_df: pd.DataFrame) -> dict:
    omap = {}
    if odds_df.empty:
        return omap
    for _, r in odds_df.iterrows():
        key = f"{str(r['home_team']).lower()}|{str(r['away_team']).lower()}"
        omap.setdefault(key, {})
        mkt, outcome, price, pt = r["market"], str(r["outcome"]).lower(), float(r["price"]), r.get("point")
        if mkt == "h2h":
            if "draw" in outcome:            omap[key]["draw"]     = price
            elif r["outcome"] == r["home_team"]: omap[key]["home_win"] = price
            else:                            omap[key]["away_win"] = price
        elif mkt == "totals" and pt is not None:
            sfx = str(pt).replace(".","")
            omap[key][f"{'over' if 'over' in outcome else 'under'}_{sfx}"] = price
    return omap


# ── Odds estimadas via Poisson (fallback quando sem API de odds) ───────────────

def estimate_odds_poisson(mu: float, nu: float) -> dict:
    """
    Converte lambdas Poisson em odds implícitas com margem de 8%.
    Usado como fallback quando não há odds reais da Bet365.
    """
    import numpy as np
    from scipy.stats import poisson

    max_g = 9
    matrix = np.zeros((max_g, max_g))
    for i in range(max_g):
        for j in range(max_g):
            matrix[i,j] = poisson.pmf(i, mu) * poisson.pmf(j, nu)
    matrix /= matrix.sum()

    p_home = float(np.tril(matrix,-1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())

    total_goals = [(i+j, matrix[i,j]) for i in range(max_g) for j in range(max_g)]
    p_over = {
        "05": sum(p for g,p in total_goals if g > 0),
        "15": sum(p for g,p in total_goals if g > 1),
        "25": sum(p for g,p in total_goals if g > 2),
        "35": sum(p for g,p in total_goals if g > 3),
    }

    margin = 1.08  # margem bookmaker simulada

    def to_odd(p): return round(1 / (p * margin), 2) if p > 0 else 0.0

    return {
        "home_win":    to_odd(p_home),
        "draw":        to_odd(p_draw),
        "away_win":    to_odd(p_away),
        "over_05":     to_odd(p_over["05"]),
        "under_05":    to_odd(1-p_over["05"]),
        "over_15":     to_odd(p_over["15"]),
        "under_15":    to_odd(1-p_over["15"]),
        "over_25":     to_odd(p_over["25"]),
        "under_25":    to_odd(1-p_over["25"]),
        "over_35":     to_odd(p_over["35"]),
        "under_35":    to_odd(1-p_over["35"]),
        "_estimated":  True,   # flag: odds não são reais
    }


def _save_raw(df: pd.DataFrame, filename: str):
    path = Path(cfg["paths"]["data_raw"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_collection_pipeline() -> dict:
    logger.info("=== Starting collection pipeline ===")
    fixtures = fetch_fixtures_today()
    if not fixtures.empty:
        _save_raw(fixtures, f"fixtures_{today_str()}.csv")

    odds_df  = fetch_odds_bet365() if ODDS_KEY else pd.DataFrame()
    odds_map = build_odds_map(odds_df)
    if not odds_df.empty:
        _save_raw(odds_df, f"odds_{today_str()}.csv")

    real_odds = len(odds_map) > 0
    logger.info(f"=== Done: {len(fixtures)} fixtures | {len(odds_map)} com odds reais | fallback Poisson: {'NÃO' if real_odds else 'SIM'} ===")
    return {"fixtures": fixtures, "odds": odds_df, "odds_map": odds_map, "has_real_odds": real_odds}
