# src/collectors/collector.py — Data collection pipeline
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.utils import load_config, get_logger, today_str

logger = get_logger("collector")
cfg = load_config()

HEADERS_FD = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_API_KEY", "")}
HEADERS_AF = {
    "X-RapidAPI-Key": os.getenv("API_FOOTBALL_KEY", ""),
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
WEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")


def _get(url: str, headers: dict, params: dict = None, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"Request failed ({i+1}/{retries}): {e}")
            time.sleep(2 ** i)
    return {}


# ─── Fixtures ────────────────────────────────────────────────────────────────

def fetch_fixtures_today() -> pd.DataFrame:
    """Busca todos os jogos do dia via API-Football."""
    date = today_str()
    url = f"{cfg['apis']['api_football']['base_url']}/fixtures"
    data = _get(url, HEADERS_AF, params={"date": date, "timezone": "America/Sao_Paulo"})
    
    rows = []
    for f in data.get("response", []):
        fix = f["fixture"]
        teams = f["teams"]
        league = f["league"]
        rows.append({
            "fixture_id": fix["id"],
            "date": fix["date"],
            "league_id": league["id"],
            "league_name": league["name"],
            "country": league["country"],
            "home_team": teams["home"]["name"],
            "home_id": teams["home"]["id"],
            "away_team": teams["away"]["name"],
            "away_id": teams["away"]["id"],
            "venue": fix.get("venue", {}).get("name", ""),
            "referee": fix.get("referee", ""),
            "status": fix["status"]["short"],
        })
    
    df = pd.DataFrame(rows)
    _save_raw(df, f"fixtures_{date}.csv")
    logger.info(f"Fetched {len(df)} fixtures for {date}")
    return df


# ─── Odds ────────────────────────────────────────────────────────────────────

def fetch_odds_bet365(sport: str = "soccer") -> pd.DataFrame:
    """Busca odds Bet365 para jogos do dia."""
    url = f"{cfg['apis']['odds_api']['base_url']}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_KEY,
        "regions": "eu",
        "markets": "h2h,totals,btts",
        "bookmakers": "bet365",
        "dateFormat": "iso",
    }
    data = _get(url, {}, params=params)
    
    rows = []
    for event in data if isinstance(data, list) else []:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        for bm in event.get("bookmakers", []):
            if bm["key"] != "bet365":
                continue
            for market in bm.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id": event["id"],
                        "home_team": home,
                        "away_team": away,
                        "commence_time": commence,
                        "market": market["key"],
                        "outcome": outcome["name"],
                        "price": outcome["price"],
                        "point": outcome.get("point", None),
                    })
    
    df = pd.DataFrame(rows)
    _save_raw(df, f"odds_{today_str()}.csv")
    logger.info(f"Fetched {len(df)} odds records")
    return df


def fetch_odds_movement(event_id: str) -> list:
    """Histórico de movimento de odds para um evento."""
    url = f"{cfg['apis']['odds_api']['base_url']}/sports/soccer/odds-history"
    params = {"apiKey": ODDS_KEY, "eventId": event_id, "bookmakers": "bet365"}
    data = _get(url, {}, params=params)
    return data if isinstance(data, list) else []


# ─── Historical Stats ─────────────────────────────────────────────────────────

def fetch_team_stats(team_id: int, league_id: int, season: int = 2024) -> dict:
    url = f"{cfg['apis']['api_football']['base_url']}/teams/statistics"
    data = _get(url, HEADERS_AF, params={
        "team": team_id, "league": league_id, "season": season
    })
    return data.get("response", {})


def fetch_recent_matches(team_id: int, last: int = 15) -> pd.DataFrame:
    url = f"{cfg['apis']['api_football']['base_url']}/fixtures"
    data = _get(url, HEADERS_AF, params={"team": team_id, "last": last})
    rows = []
    for f in data.get("response", []):
        goals = f.get("goals", {})
        teams = f["teams"]
        is_home = teams["home"]["id"] == team_id
        rows.append({
            "fixture_id": f["fixture"]["id"],
            "date": f["fixture"]["date"],
            "is_home": is_home,
            "team_goals": goals.get("home" if is_home else "away", 0),
            "opp_goals": goals.get("away" if is_home else "home", 0),
            "result": ("W" if (goals.get("home",0) > goals.get("away",0)) == is_home else
                       "D" if goals.get("home",0) == goals.get("away",0) else "L"),
        })
    return pd.DataFrame(rows)


def fetch_xg_data(fixture_id: int) -> dict:
    """xG via API-Football statistics endpoint."""
    url = f"{cfg['apis']['api_football']['base_url']}/fixtures/statistics"
    data = _get(url, HEADERS_AF, params={"fixture": fixture_id})
    xg = {}
    for team_data in data.get("response", []):
        team_name = team_data["team"]["name"]
        for stat in team_data.get("statistics", []):
            if stat["type"] == "Expected Goals":
                xg[team_name] = float(stat["value"] or 0)
    return xg


# ─── Injuries & Suspensions ───────────────────────────────────────────────────

def fetch_injuries(team_id: int, fixture_id: int) -> pd.DataFrame:
    url = f"{cfg['apis']['api_football']['base_url']}/injuries"
    data = _get(url, HEADERS_AF, params={"team": team_id, "fixture": fixture_id})
    rows = []
    for p in data.get("response", []):
        player = p.get("player", {})
        rows.append({
            "player_id": player.get("id"),
            "player_name": player.get("name"),
            "type": p.get("type", ""),
            "reason": p.get("reason", ""),
        })
    return pd.DataFrame(rows)


# ─── Referee ──────────────────────────────────────────────────────────────────

def fetch_referee_history(referee_name: str, season: int = 2024) -> pd.DataFrame:
    """Busca histórico de partidas do árbitro."""
    url = f"{cfg['apis']['api_football']['base_url']}/fixtures"
    # Filtra por árbitro nos fixtures da temporada
    data = _get(url, HEADERS_AF, params={"season": season, "timezone": "UTC"})
    rows = []
    for f in data.get("response", []):
        if referee_name.lower() in (f["fixture"].get("referee") or "").lower():
            goals = f.get("goals", {})
            rows.append({
                "fixture_id": f["fixture"]["id"],
                "date": f["fixture"]["date"],
                "home_goals": goals.get("home", 0),
                "away_goals": goals.get("away", 0),
                "total_goals": (goals.get("home", 0) or 0) + (goals.get("away", 0) or 0),
            })
    return pd.DataFrame(rows)


# ─── Weather ──────────────────────────────────────────────────────────────────

def fetch_weather(city: str) -> dict:
    if not WEATHER_KEY or not city:
        return {}
    url = f"{cfg['apis']['openweather']['base_url']}/weather"
    data = _get(url, {}, params={"q": city, "appid": WEATHER_KEY, "units": "metric"})
    return {
        "temp_c": data.get("main", {}).get("temp"),
        "condition": data.get("weather", [{}])[0].get("main", ""),
        "wind_kmh": data.get("wind", {}).get("speed", 0) * 3.6,
        "humidity": data.get("main", {}).get("humidity"),
    }


# ─── Player Props ─────────────────────────────────────────────────────────────

def fetch_player_stats(player_id: int, league_id: int, season: int = 2024) -> dict:
    url = f"{cfg['apis']['api_football']['base_url']}/players"
    data = _get(url, HEADERS_AF, params={
        "id": player_id, "league": league_id, "season": season
    })
    resp = data.get("response", [])
    return resp[0] if resp else {}


def fetch_lineup(fixture_id: int) -> dict:
    url = f"{cfg['apis']['api_football']['base_url']}/fixtures/lineups"
    data = _get(url, HEADERS_AF, params={"fixture": fixture_id})
    return data.get("response", [])


# ─── Persistence ─────────────────────────────────────────────────────────────

def _save_raw(df: pd.DataFrame, filename: str):
    path = Path(cfg["paths"]["data_raw"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.debug(f"Saved raw data: {path}")


def load_historical(filename: str) -> pd.DataFrame:
    path = Path(cfg["paths"]["data_historical"]) / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_collection_pipeline() -> dict:
    """Executa pipeline completo de coleta."""
    logger.info("=== Starting data collection pipeline ===")
    
    fixtures = fetch_fixtures_today()
    odds = fetch_odds_bet365()
    
    results = {
        "fixtures": fixtures,
        "odds": odds,
        "team_stats": {},
        "injuries": {},
        "weather": {},
    }
    
    for _, row in fixtures.iterrows():
        fid = row["fixture_id"]
        
        # Team stats
        for side in ["home", "away"]:
            tid = row[f"{side}_id"]
            key = f"{side}_{tid}"
            results["team_stats"][key] = fetch_team_stats(tid, row["league_id"])
            time.sleep(0.5)
        
        # Injuries
        results["injuries"][fid] = {
            "home": fetch_injuries(row["home_id"], fid),
            "away": fetch_injuries(row["away_id"], fid),
        }
        
        # Weather
        if row.get("venue"):
            results["weather"][fid] = fetch_weather(row.get("country", ""))
    
    logger.info("=== Collection pipeline complete ===")
    return results
