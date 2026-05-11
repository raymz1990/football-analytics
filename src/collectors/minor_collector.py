# src/collectors/minor_collector.py
# Coleta para ligas menores via The Odds API (sport_keys)
# API-Football só é usado se a chave funcionar e por liga específica.

import os
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.utils import load_config, get_logger, today_str

logger           = get_logger("collector.minor")
cfg              = load_config()
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
ODDS_KEY         = os.getenv("ODDS_API_KEY", "")
BASE_AF          = cfg["apis"]["api_football"]["base_url"]
BASE_ODDS        = cfg["apis"]["odds_api"]["base_url"]
HEADERS_AF       = {
    "X-RapidAPI-Key":  API_FOOTBALL_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
}

# ── Sport keys das ligas menores na The Odds API ─────────────────────────────
MINOR_SPORT_KEYS = [
    # Europa
    "soccer_iceland_premier_league",
    "soccer_norway_eliteserien",
    "soccer_sweden_allsvenskan",
    "soccer_finland_veikkausliiga",
    "soccer_denmark_superliga",
    "soccer_poland_ekstraklasa",
    "soccer_scotland_premiership",
    "soccer_belgium_first_div",
    "soccer_greece_super_league",
    "soccer_switzerland_superleague",
    "soccer_russia_premier_league",
    # Oceania / Ásia-Pacífico
    "soccer_australia_aleague",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    # Américas
    "soccer_argentina_primera_division",
    "soccer_chile_primera_division",
    "soccer_colombia_primera_a",
    "soccer_usa_usl_championship",
    "soccer_canada_premier_league",
    # África / Oriente Médio
    "soccer_south_africa_premier_division",
    "soccer_saudi_arabias_pro_league",
    "soccer_israel_premier_league",
]


def _get(url, headers, params=None, retries=3, label="") -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code in (403, 401):
                logger.error(f"Auth error {r.status_code} — {label}")
                return {}
            if r.status_code == 429:
                logger.warning(f"Rate limit — sleeping 61s ({label})")
                time.sleep(61)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[attempt {i+1}/{retries}] {label}: {e}")
            if i < retries - 1:
                time.sleep(2 ** i)
    return {}


def fetch_minor_odds_by_sport(sport_key: str) -> List[dict]:
    """Busca odds Bet365 para um sport_key — mercados h2h e totals."""
    url    = f"{BASE_ODDS}/sports/{sport_key}/odds"
    params = {
        "apiKey":     ODDS_KEY,
        "regions":    "eu,uk,au",
        "markets":    "h2h,totals",
        "bookmakers": "bet365",
        "dateFormat": "iso",
    }
    data = _get(url, {}, params=params, label=f"Minor/{sport_key}")
    return data if isinstance(data, list) else []


def fetch_minor_ht_odds(sport_key: str) -> List[dict]:
    """Busca odds de HT quando disponível."""
    url    = f"{BASE_ODDS}/sports/{sport_key}/odds"
    params = {
        "apiKey":     ODDS_KEY,
        "regions":    "eu,uk",
        "markets":    "h2h_h1,totals_h1",
        "bookmakers": "bet365",
        "dateFormat": "iso",
    }
    data = _get(url, {}, params=params, label=f"MinorHT/{sport_key}")
    return data if isinstance(data, list) else []


def _parse_events(events: List[dict], is_ht: bool = False) -> List[dict]:
    """Extrai fixtures e odds de uma lista de eventos da Odds API."""
    prefix = "ht_" if is_ht else ""
    rows   = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        odds = {}
        for bm in event.get("bookmakers", []):
            if bm["key"] != "bet365":
                continue
            for mkt in bm.get("markets", []):
                key = mkt.get("key", "")
                for o in mkt.get("outcomes", []):
                    name  = o["name"].lower()
                    price = o["price"]
                    pt    = o.get("point")
                    if key in ("h2h", "h2h_h1"):
                        if "draw" in name:
                            odds[f"{prefix}draw"] = price
                        elif o["name"] == home:
                            odds[f"{prefix}home_win"] = price
                        else:
                            odds[f"{prefix}away_win"] = price
                    elif key in ("totals", "totals_h1") and pt is not None:
                        sfx = str(pt).replace(".", "")
                        if "over" in name:
                            odds[f"{prefix}over_{sfx}"] = price
                        else:
                            odds[f"{prefix}under_{sfx}"] = price
        rows.append({
            "home_team":     home,
            "away_team":     away,
            "commence_time": event.get("commence_time", ""),
            "odds":          odds,
        })
    return rows


def _default_profile(league_name: str) -> Dict:
    return {
        "league": league_name, "n_matches": 0,
        "avg_goals_ft": 2.45, "avg_home_goals": 1.35, "avg_away_goals": 1.10,
        "over_05_ft": 0.92, "over_15_ft": 0.74, "over_25_ft": 0.52,
        "over_35_ft": 0.30, "btts_ft": 0.48,
        "home_win_rate": 0.45, "draw_rate": 0.26, "away_win_rate": 0.29,
        "avg_goals_ht": 1.05, "over_05_ht": 0.68, "over_15_ht": 0.35,
        "over_25_ht": 0.14, "ht_ft_ratio": 0.43,
    }


def run_minor_collection_pipeline() -> Dict:
    """Pipeline de coleta para ligas menores via The Odds API."""
    logger.info("=== Minor Leagues Collection Pipeline ===")

    if not ODDS_KEY:
        logger.warning("ODDS_API_KEY not set — cannot fetch minor league odds")
        return {"fixtures": pd.DataFrame(), "odds_map": {}, "league_profiles": {}}

    all_fixtures   = []
    odds_map       = {}
    league_profiles = {}

    for sport_key in MINOR_SPORT_KEYS:
        logger.info(f"  Processing: {sport_key}")

        # FT odds
        ft_events = fetch_minor_odds_by_sport(sport_key)
        # HT odds
        ht_events = fetch_minor_ht_odds(sport_key)

        # Parse HT odds
        ht_map = {}
        for row in _parse_events(ht_events, is_ht=True):
            k = f"{row['home_team'].lower()}|{row['away_team'].lower()}"
            ht_map[k] = row["odds"]

        for row in _parse_events(ft_events, is_ht=False):
            home = row["home_team"]
            away = row["away_team"]
            key  = f"{home.lower()}|{away.lower()}"

            # Combina FT + HT odds
            combined_odds = {**row["odds"], **ht_map.get(key, {})}
            odds_map[key] = combined_odds

            # Infere nome da liga pelo sport_key
            league_name = sport_key.replace("soccer_", "").replace("_", " ").title()
            country     = sport_key.split("_")[1].title() if "_" in sport_key else ""

            all_fixtures.append({
                "fixture_id":  f"{sport_key}_{key}",
                "date":        row["commence_time"],
                "league_id":   sport_key,
                "league_name": league_name,
                "country":     country,
                "home_team":   home,
                "home_id":     0,
                "away_team":   away,
                "away_id":     0,
                "venue":       "",
                "referee":     "",
                "status":      "NS",
                "minor_focus": True,
                "group":       _infer_group(sport_key),
                "timezone":    "UTC",
                "source":      "odds-api",
            })

            # Perfil da liga (default — sem histórico no free tier)
            if league_name not in league_profiles:
                league_profiles[league_name] = _default_profile(league_name)

        time.sleep(0.3)

    fixtures = pd.DataFrame(all_fixtures) if all_fixtures else pd.DataFrame()
    logger.info(f"Minor pipeline: {len(fixtures)} fixtures, {len(odds_map)} odds events, {len(league_profiles)} leagues")
    return {
        "fixtures":        fixtures,
        "odds_map":        odds_map,
        "league_profiles": league_profiles,
    }


def _infer_group(sport_key: str) -> str:
    k = sport_key.lower()
    if any(x in k for x in ["iceland","norway","sweden","finland","denmark","poland",
                              "scotland","belgium","greece","switzerland","russia"]):
        return "minor_europe"
    if any(x in k for x in ["australia","japan","korea"]):
        return "minor_oceania"
    if any(x in k for x in ["argentina","chile","colombia","usa","canada","brazil"]):
        return "minor_americas"
    if any(x in k for x in ["saudi","israel","china","india","thailand","indonesia"]):
        return "minor_asia"
    if any(x in k for x in ["south_africa","egypt","morocco","nigeria","ghana"]):
        return "minor_africa"
    return "other"


def get_all_minor_leagues():
    return []
