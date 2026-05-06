# src/collectors/minor_collector.py
# Coleta especializada para ligas menores com dados esparsos.
# Foco: 1X2, Over/Under FT, Over/Under HT.
# Estratégia: agregar múltiplas fontes para compensar falta de dados históricos.

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from src.utils import load_config, get_logger, today_str

logger = get_logger("collector.minor")
cfg = load_config()

HEADERS_AF = {
    "X-RapidAPI-Key": os.getenv("API_FOOTBALL_KEY", ""),
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
}
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
BASE_AF = cfg["apis"]["api_football"]["base_url"]
BASE_ODDS = cfg["apis"]["odds_api"]["base_url"]


def _get(url: str, headers: dict, params: dict = None, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[attempt {i+1}] {url} → {e}")
            time.sleep(2 ** i)
    return {}


# ── Helpers de liga ──────────────────────────────────────────────────────────

def get_all_minor_leagues() -> List[Dict]:
    """Retorna todas as ligas minor_focus=True do config."""
    minor_groups = ["minor_europe", "minor_oceania", "minor_americas", "minor_asia", "minor_africa"]
    leagues = []
    for group in minor_groups:
        for lg in cfg["leagues"].get(group, []):
            lg["group"] = group
            leagues.append(lg)
    # Também inclui tier_3, cups, womens com minor_focus=True
    for group in ["tier_3", "cups", "womens"]:
        for lg in cfg["leagues"].get(group, []):
            if lg.get("minor_focus", False):
                lg["group"] = group
                leagues.append(lg)
    return leagues


# ── Fixtures do dia (ligas menores) ──────────────────────────────────────────

def fetch_minor_fixtures_today() -> pd.DataFrame:
    """
    Busca todos os fixtures do dia para ligas menores via API-Football.
    Faz uma única chamada sem filtro de liga — mais eficiente para cobertura global.
    """
    date = today_str()
    url = f"{BASE_AF}/fixtures"
    data = _get(url, HEADERS_AF, params={
        "date": date,
        "timezone": "UTC",
        "status": "NS",   # Not Started
    })

    minor_ids = {lg["id"] for lg in get_all_minor_leagues()}

    rows = []
    for f in data.get("response", []):
        fix = f["fixture"]
        league = f["league"]
        teams = f["teams"]
        goals = f.get("goals", {})
        score = f.get("score", {})

        if league["id"] not in minor_ids:
            continue

        # Encontra metadados da liga
        lg_meta = next((l for l in get_all_minor_leagues() if l["id"] == league["id"]), {})

        rows.append({
            "fixture_id":   fix["id"],
            "date":         fix["date"],
            "league_id":    league["id"],
            "league_name":  league["name"],
            "country":      league["country"],
            "group":        lg_meta.get("group", "unknown"),
            "timezone":     lg_meta.get("timezone", "UTC"),
            "home_team":    teams["home"]["name"],
            "home_id":      teams["home"]["id"],
            "away_team":    teams["away"]["name"],
            "away_id":      teams["away"]["id"],
            "venue":        fix.get("venue", {}).get("name", ""),
            "referee":      fix.get("referee") or "",
            "status":       fix["status"]["short"],
            "minor_focus":  True,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Minor leagues fixtures today: {len(df)} ({df['league_name'].nunique() if not df.empty else 0} ligas)")
    return df


# ── Odds para ligas menores ───────────────────────────────────────────────────

def fetch_minor_odds(sport_key: str, regions: str = "eu,uk,au") -> List[Dict]:
    """
    Busca odds Bet365 para um sport_key específico via The Odds API.
    Mercados: h2h (1X2), totals (Over/Under FT).
    """
    url = f"{BASE_ODDS}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_KEY,
        "regions": regions,
        "markets": "h2h,totals",
        "bookmakers": "bet365",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    data = _get(url, {}, params=params)
    return data if isinstance(data, list) else []


def fetch_halftime_odds(sport_key: str) -> List[Dict]:
    """
    Busca odds de 1ª metade (HT) via The Odds API — mercado h2h_h1 + totals_h1.
    Nem todas as ligas menores têm esse mercado.
    """
    url = f"{BASE_ODDS}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_KEY,
        "regions": "eu,uk",
        "markets": "h2h_h1,totals_h1",
        "bookmakers": "bet365",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    data = _get(url, {}, params=params)
    return data if isinstance(data, list) else []


def build_minor_odds_map(fixtures: pd.DataFrame) -> Dict[str, Dict]:
    """
    Para cada fixture de liga menor, monta um dict de odds:
    {fixture_key: {home_win, draw, away_win, over_25, under_25, ht_over_15, ...}}
    """
    # Agrupa por odds_key da liga
    league_odds_keys = {}
    for lg in get_all_minor_leagues():
        league_odds_keys[lg["name"]] = lg.get("odds_key", "")

    odds_map: Dict[str, Dict] = {}
    fetched_keys = set()

    for _, row in fixtures.iterrows():
        sport_key = league_odds_keys.get(row["league_name"], "")
        if not sport_key or sport_key in fetched_keys:
            continue

        logger.info(f"  Fetching odds: {sport_key}")
        ft_odds = fetch_minor_odds(sport_key)
        ht_odds = fetch_halftime_odds(sport_key)
        fetched_keys.add(sport_key)

        # Mapeia por par de times
        for event in ft_odds:
            key = _event_key(event["home_team"], event["away_team"])
            odds_map.setdefault(key, {})
            for bm in event.get("bookmakers", []):
                if bm["key"] != "bet365":
                    continue
                for mkt in bm.get("markets", []):
                    _parse_market(odds_map[key], mkt, half=False)

        for event in ht_odds:
            key = _event_key(event["home_team"], event["away_team"])
            odds_map.setdefault(key, {})
            for bm in event.get("bookmakers", []):
                if bm["key"] != "bet365":
                    continue
                for mkt in bm.get("markets", []):
                    _parse_market(odds_map[key], mkt, half=True)

        time.sleep(0.5)  # respeita rate limit

    return odds_map


def _event_key(home: str, away: str) -> str:
    return f"{home.lower().strip()}|{away.lower().strip()}"


def _parse_market(target: dict, market: dict, half: bool):
    prefix = "ht_" if half else ""
    key = market.get("key", "")

    if key in ("h2h", "h2h_h1"):
        for o in market.get("outcomes", []):
            n = o["name"].lower()
            if "draw" in n:
                target[f"{prefix}draw"] = o["price"]
            elif o["name"] == market.get("home_team", ""):
                target[f"{prefix}home_win"] = o["price"]
            else:
                target[f"{prefix}away_win"] = o["price"]
        # fallback por posição
        outcomes = market.get("outcomes", [])
        if len(outcomes) >= 3 and f"{prefix}home_win" not in target:
            target[f"{prefix}home_win"] = outcomes[0]["price"]
            target[f"{prefix}draw"]     = outcomes[1]["price"]
            target[f"{prefix}away_win"] = outcomes[2]["price"]

    elif key in ("totals", "totals_h1"):
        for o in market.get("outcomes", []):
            pt = o.get("point", 2.5)
            name = o["name"].lower()
            label = f"{prefix}over_{str(pt).replace('.','')}" if "over" in name else f"{prefix}under_{str(pt).replace('.','')}"
            target[label] = o["price"]


# ── Histórico de ligas menores ────────────────────────────────────────────────

def fetch_minor_league_history(league_id: int, season: int = 2024,
                               max_rounds: int = 30) -> pd.DataFrame:
    """
    Busca histórico de resultados de uma liga menor.
    Limita a max_rounds para não estourar o rate limit.
    """
    url = f"{BASE_AF}/fixtures"
    data = _get(url, HEADERS_AF, params={
        "league": league_id,
        "season": season,
        "status": "FT",
    })

    rows = []
    for f in data.get("response", [])[:max_rounds * 2]:
        goals = f.get("goals", {})
        score = f.get("score", {})
        ht = score.get("halftime", {})
        teams = f["teams"]
        rows.append({
            "fixture_id":    f["fixture"]["id"],
            "date":          f["fixture"]["date"],
            "home_team":     teams["home"]["name"],
            "away_team":     teams["away"]["name"],
            "home_goals":    goals.get("home") or 0,
            "away_goals":    goals.get("away") or 0,
            "ht_home_goals": ht.get("home") or 0,
            "ht_away_goals": ht.get("away") or 0,
            "total_goals":   (goals.get("home") or 0) + (goals.get("away") or 0),
            "ht_total_goals": (ht.get("home") or 0) + (ht.get("away") or 0),
        })

    return pd.DataFrame(rows)


def build_minor_league_profile(history: pd.DataFrame, league_name: str) -> Dict:
    """
    Perfil estatístico simplificado para ligas menores — base para os modelos.
    """
    if history.empty:
        return _default_minor_profile(league_name)

    total = history["total_goals"]
    ht_total = history["ht_total_goals"]

    profile = {
        "league":           league_name,
        "n_matches":        len(history),
        # FT
        "avg_goals_ft":     round(float(total.mean()), 3),
        "avg_home_goals":   round(float(history["home_goals"].mean()), 3),
        "avg_away_goals":   round(float(history["away_goals"].mean()), 3),
        "over_05_ft":       round(float((total > 0).mean()), 3),
        "over_15_ft":       round(float((total > 1).mean()), 3),
        "over_25_ft":       round(float((total > 2).mean()), 3),
        "over_35_ft":       round(float((total > 3).mean()), 3),
        "btts_ft":          round(float(((history["home_goals"] > 0) & (history["away_goals"] > 0)).mean()), 3),
        "home_win_rate":    round(float((history["home_goals"] > history["away_goals"]).mean()), 3),
        "draw_rate":        round(float((history["home_goals"] == history["away_goals"]).mean()), 3),
        "away_win_rate":    round(float((history["away_goals"] > history["home_goals"]).mean()), 3),
        # HT
        "avg_goals_ht":     round(float(ht_total.mean()), 3),
        "over_05_ht":       round(float((ht_total > 0).mean()), 3),
        "over_15_ht":       round(float((ht_total > 1).mean()), 3),
        "over_25_ht":       round(float((ht_total > 2).mean()), 3),
        # Relação HT/FT
        "ht_ft_ratio":      round(float(ht_total.mean() / total.mean()), 3) if total.mean() > 0 else 0.45,
    }
    return profile


def _default_minor_profile(league_name: str) -> Dict:
    """Perfil padrão para ligas sem histórico suficiente."""
    return {
        "league": league_name, "n_matches": 0,
        "avg_goals_ft": 2.45, "avg_home_goals": 1.35, "avg_away_goals": 1.10,
        "over_05_ft": 0.92, "over_15_ft": 0.74, "over_25_ft": 0.52,
        "over_35_ft": 0.30, "btts_ft": 0.48,
        "home_win_rate": 0.45, "draw_rate": 0.26, "away_win_rate": 0.29,
        "avg_goals_ht": 1.05, "over_05_ht": 0.68, "over_15_ht": 0.35,
        "over_25_ht": 0.14, "ht_ft_ratio": 0.43,
    }


# ── Pipeline completo para ligas menores ──────────────────────────────────────

def run_minor_collection_pipeline() -> Dict:
    """
    Pipeline de coleta para ligas menores.
    Retorna: fixtures, odds_map, league_profiles.
    """
    logger.info("=== Minor Leagues Collection Pipeline ===")

    fixtures = fetch_minor_fixtures_today()

    if fixtures.empty:
        logger.warning("No minor league fixtures found today.")
        return {"fixtures": fixtures, "odds_map": {}, "league_profiles": {}}

    logger.info(f"Building odds map for {len(fixtures)} fixtures...")
    odds_map = build_minor_odds_map(fixtures)

    # Histórico e perfil por liga
    league_profiles = {}
    processed_leagues = set()

    for _, row in fixtures.iterrows():
        lid = row["league_id"]
        lname = row["league_name"]
        if lid in processed_leagues:
            continue
        processed_leagues.add(lid)

        logger.info(f"  Loading history: {lname}")
        hist = fetch_minor_league_history(lid, season=2024)
        if len(hist) < 5:
            hist2 = fetch_minor_league_history(lid, season=2023)
            hist = pd.concat([hist, hist2], ignore_index=True)

        profile = build_minor_league_profile(hist, lname)
        league_profiles[lname] = profile

        # Salva histórico
        hist_path = Path(cfg["paths"]["data_historical"]) / f"minor_{lid}_{lname[:20].replace(' ','_')}.csv"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(hist_path, index=False)

        time.sleep(0.3)

    logger.info(f"Minor pipeline done. {len(fixtures)} fixtures, {len(league_profiles)} leagues profiled.")
    return {
        "fixtures": fixtures,
        "odds_map": odds_map,
        "league_profiles": league_profiles,
    }
