# src/collectors/collector.py
import os
import time
import requests
import pandas as pd
from pathlib import Path
from src.utils import load_config, get_logger, today_str

logger   = get_logger("collector")
cfg      = load_config()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
ODDS_KEY         = os.getenv("ODDS_API_KEY", "")
WEATHER_KEY      = os.getenv("OPENWEATHER_KEY", "")
FD_KEY           = os.getenv("FOOTBALL_DATA_API_KEY", "")

HEADERS_AF = {
    "X-RapidAPI-Key":  API_FOOTBALL_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
}
HEADERS_FD = {"X-Auth-Token": FD_KEY}

BASE_AF   = cfg["apis"]["api_football"]["base_url"]
BASE_ODDS = cfg["apis"]["odds_api"]["base_url"]
BASE_FD   = cfg["apis"]["football_data"]["base_url"]


def _check_keys():
    logger.info(f"FOOTBALL_DATA_API_KEY: {'SET ✅' if FD_KEY   else 'MISSING ❌'}")
    logger.info(f"API_FOOTBALL_KEY:      {'SET ✅' if API_FOOTBALL_KEY else 'MISSING ❌'}")
    logger.info(f"ODDS_API_KEY:          {'SET ✅' if ODDS_KEY else 'MISSING ❌'}")


def _get(url, headers, params=None, retries=3, label="") -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 429:
                wait = 61
                logger.warning(f"Rate limit on {label or url} — sleeping {wait}s")
                time.sleep(wait)
                continue
            if r.status_code in (403, 401):
                logger.error(f"Auth error {r.status_code} on {label or url} — check API key/plan")
                return {}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[attempt {i+1}/{retries}] {label or url}: {e}")
            if i < retries - 1:
                time.sleep(2 ** i)
    return {}


# ═══════════════════════════════════════════════════════════════════
# SOURCE 1: football-data.org  (free, sem precisar de plano pago)
# Cobre: PL, La Liga, Serie A, Bundesliga, Ligue 1, CL, EL
# ═══════════════════════════════════════════════════════════════════

# IDs das ligas no football-data.org
FD_COMPETITIONS = [
    "PL",   # Premier League
    "PD",   # La Liga
    "SA",   # Serie A
    "BL1",  # Bundesliga
    "FL1",  # Ligue 1
    "CL",   # Champions League
    "EL",   # Europa League
    "PPL",  # Primeira Liga
    "DED",  # Eredivisie
    "BSA",  # Brasileirão
]

def fetch_fixtures_football_data() -> pd.DataFrame:
    """
    Busca fixtures do dia via football-data.org.
    Plano free permite 10 req/min e cobre as principais ligas.
    """
    date = today_str()
    logger.info(f"[football-data.org] Fetching fixtures for {date}...")
    rows = []

    for comp in FD_COMPETITIONS:
        url  = f"{BASE_FD}/competitions/{comp}/matches"
        data = _get(url, HEADERS_FD,
                    params={"dateFrom": date, "dateTo": date},
                    label=f"FD/{comp}")
        matches = data.get("matches", [])
        logger.info(f"  {comp}: {len(matches)} matches")

        for m in matches:
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            rows.append({
                "fixture_id":  m.get("id"),
                "date":        m.get("utcDate", ""),
                "league_id":   comp,
                "league_name": m.get("competition", {}).get("name", comp),
                "country":     m.get("area", {}).get("name", ""),
                "home_team":   home.get("shortName") or home.get("name", ""),
                "home_id":     home.get("id"),
                "away_team":   away.get("shortName") or away.get("name", ""),
                "away_id":     away.get("id"),
                "venue":       "",
                "referee":     "",
                "status":      m.get("status", ""),
                "minor_focus": False,
                "source":      "football-data.org",
            })
        time.sleep(6.5)  # respeita 10 req/min

    df = pd.DataFrame(rows)
    logger.info(f"[football-data.org] Total: {len(df)} fixtures")
    return df


# ═══════════════════════════════════════════════════════════════════
# SOURCE 2: API-Football via RapidAPI (por liga, não por data global)
# Só usa se a chave funcionar — testa primeiro
# ═══════════════════════════════════════════════════════════════════

# Ligas tier 1-2 com IDs no API-Football
AF_LEAGUE_IDS = [
    39,   # Premier League
    140,  # La Liga
    135,  # Serie A
    78,   # Bundesliga
    61,   # Ligue 1
    71,   # Brasileirão A
    94,   # Primeira Liga
    88,   # Eredivisie
    203,  # Süper Lig
    179,  # Scottish Prem
    253,  # MLS
    128,  # Argentine Primera
    292,  # K League 1
    98,   # J1 League
    235,  # Russian Premier
    2,    # Champions League
    3,    # Europa League
    848,  # Conference League
]

def _test_api_football() -> bool:
    """Testa se a chave do API-Football funciona usando endpoint /timezone (mais leve)."""
    data = _get(f"{BASE_AF}/timezone", HEADERS_AF, label="AF/status", retries=1)
    # Plano free retorna lista de timezones — qualquer resposta válida = chave OK
    ok = isinstance(data.get("response"), list) and len(data.get("response", [])) > 0
    logger.info(f"[API-Football] Key test: {'OK ✅' if ok else 'FAILED ❌'} — response: {str(data)[:80]}")
    return ok


def fetch_fixtures_api_football() -> pd.DataFrame:
    """Busca fixtures por liga (contorna a restrição do plano free)."""
    date = today_str()
    season = date[:4]  # ex: "2026"
    rows = []

    for league_id in AF_LEAGUE_IDS:
        data = _get(f"{BASE_AF}/fixtures", HEADERS_AF,
                    params={"league": league_id, "date": date, "season": season,
                            "timezone": "America/Sao_Paulo"},
                    label=f"AF/league/{league_id}")

        if not data:
            continue

        errors = data.get("errors", {})
        if errors:
            logger.warning(f"  League {league_id} errors: {errors}")
            continue

        response = data.get("response", [])
        for f in response:
            fix    = f["fixture"]
            teams  = f["teams"]
            league = f["league"]
            rows.append({
                "fixture_id":  fix["id"],
                "date":        fix["date"],
                "league_id":   league["id"],
                "league_name": league["name"],
                "country":     league["country"],
                "home_team":   teams["home"]["name"],
                "home_id":     teams["home"]["id"],
                "away_team":   teams["away"]["name"],
                "away_id":     teams["away"]["id"],
                "venue":       (fix.get("venue") or {}).get("name", ""),
                "referee":     fix.get("referee") or "",
                "status":      fix["status"]["short"],
                "minor_focus": False,
                "source":      "api-football",
            })
        time.sleep(0.8)

    df = pd.DataFrame(rows)
    logger.info(f"[API-Football] Total: {len(df)} fixtures")
    return df


# ═══════════════════════════════════════════════════════════════════
# SOURCE 3: The Odds API  (odds Bet365)
# Mercados: h2h (1X2) + totals (Over/Under)
# ═══════════════════════════════════════════════════════════════════

# sport_keys da The Odds API para as principais ligas
ODDS_SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_brazil_campeonato",
    "soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie",
    "soccer_turkey_super_league",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_uefa_conference_league",
    "soccer_usa_mls",
    "soccer_argentina_primera_division",
    "soccer_efl_champ",
    "soccer_spain_segunda_division",
]

def fetch_odds_bet365() -> pd.DataFrame:
    """
    Busca odds Bet365 via The Odds API.
    Mercados: h2h (1X2) e totals (Over/Under).
    Nota: 'btts' não é suportado nessa API — removido.
    """
    logger.info("[The Odds API] Fetching Bet365 odds...")
    rows = []

    for sport_key in ODDS_SPORT_KEYS:
        url    = f"{BASE_ODDS}/sports/{sport_key}/odds"
        params = {
            "apiKey":     ODDS_KEY,
            "regions":    "eu",
            "markets":    "h2h,totals",   # btts removido — causa 422
            "bookmakers": "bet365",
            "dateFormat": "iso",
        }
        data = _get(url, {}, params=params, label=f"Odds/{sport_key}")
        if not isinstance(data, list):
            continue

        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            for bm in event.get("bookmakers", []):
                if bm["key"] != "bet365":
                    continue
                for mkt in bm.get("markets", []):
                    for outcome in mkt.get("outcomes", []):
                        rows.append({
                            "sport_key":     sport_key,
                            "event_id":      event["id"],
                            "home_team":     home,
                            "away_team":     away,
                            "commence_time": event.get("commence_time", ""),
                            "market":        mkt["key"],
                            "outcome":       outcome["name"],
                            "price":         outcome["price"],
                            "point":         outcome.get("point"),
                        })
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    logger.info(f"[The Odds API] Total: {len(df)} odds records across {df['sport_key'].nunique() if not df.empty else 0} sports")
    _save_raw(df, f"odds_{today_str()}.csv")
    return df


def build_odds_map(odds_df: pd.DataFrame) -> dict:
    """Constrói dict {home|away: {market: odd}} para lookup rápido."""
    omap = {}
    if odds_df.empty:
        return omap

    for _, row in odds_df.iterrows():
        key = f"{str(row['home_team']).lower()}|{str(row['away_team']).lower()}"
        omap.setdefault(key, {})
        mkt     = row["market"]
        outcome = str(row["outcome"]).lower()
        price   = float(row["price"])
        pt      = row.get("point")

        if mkt == "h2h":
            if "draw" in outcome:
                omap[key]["draw"] = price
            elif outcome == str(row["home_team"]).lower():
                omap[key]["home_win"] = price
            else:
                omap[key]["away_win"] = price

        elif mkt == "totals" and pt is not None:
            suffix = str(pt).replace(".", "")
            if "over" in outcome:
                omap[key][f"over_{suffix}"] = price
            else:
                omap[key][f"under_{suffix}"] = price

    # Fallback h2h por posição (quando os nomes não batem)
    h2h_events = odds_df[odds_df["market"] == "h2h"].groupby(["home_team", "away_team"])
    for (home, away), grp in h2h_events:
        key = f"{home.lower()}|{away.lower()}"
        prices = grp.sort_values("outcome")["price"].tolist()
        if len(prices) >= 3 and "home_win" not in omap.get(key, {}):
            omap.setdefault(key, {})
            omap[key].setdefault("home_win", prices[0])
            omap[key].setdefault("draw",     prices[1])
            omap[key].setdefault("away_win", prices[2])

    return omap


# ═══════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════

def _save_raw(df: pd.DataFrame, filename: str):
    path = Path(cfg["paths"]["data_raw"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_historical(filename: str) -> pd.DataFrame:
    path = Path(cfg["paths"]["data_historical"]) / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_collection_pipeline() -> dict:
    logger.info("=== Starting collection pipeline ===")
    _check_keys()

    # --- Fixtures: tenta football-data.org primeiro (mais confiável no free) ---
    fixtures = pd.DataFrame()

    if FD_KEY:
        fixtures = fetch_fixtures_football_data()
    else:
        logger.warning("FOOTBALL_DATA_API_KEY not set — skipping football-data.org")

    # Se API-Football funcionar, complementa com mais ligas
    if API_FOOTBALL_KEY:
        af_ok = _test_api_football()
        if af_ok:
            af_fixtures = fetch_fixtures_api_football()
            if not af_fixtures.empty:
                # Remove duplicatas por fixture_id
                combined = pd.concat([fixtures, af_fixtures], ignore_index=True)
                fixtures = combined.drop_duplicates(
                    subset=["home_team", "away_team", "date"], keep="first"
                )
                logger.info(f"Combined fixtures (FD + AF): {len(fixtures)}")
    else:
        logger.warning("API_FOOTBALL_KEY not set — skipping API-Football")

    # --- Odds ---
    odds_df  = pd.DataFrame()
    odds_map = {}
    if ODDS_KEY:
        odds_df  = fetch_odds_bet365()
        odds_map = build_odds_map(odds_df)
    else:
        logger.warning("ODDS_API_KEY not set — no odds data")

    logger.info(f"=== Collection done: {len(fixtures)} fixtures, {len(odds_map)} events with odds ===")
    return {
        "fixtures": fixtures,
        "odds":     odds_df,
        "odds_map": odds_map,
    }
