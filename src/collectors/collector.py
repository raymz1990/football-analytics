# src/collectors/collector.py
# Fontes: football-data.org (fixtures) + The Odds API (odds Bet365)
# API-Football removida — plano free RapidAPI instável e desnecessário.

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

# ── Competições football-data.org (plano free) ────────────────────────────────
# Plano free cobre: PL, PD, BL1, SA, FL1, CL, PPL, DED, BSA, ELC
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

# ── Sport keys validados da The Odds API ──────────────────────────────────────
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
    "soccer_uefa_europa_conference_league",
    "soccer_usa_mls",
    "soccer_argentina_primera_division",
    "soccer_efl_champ",
    "soccer_spain_segunda_division",
    "soccer_russia_premier_league",
    "soccer_mexico_ligamx",
    "soccer_belgium_first_div",
    "soccer_greece_super_league",
    "soccer_scotland_premiership",
    "soccer_spl",
]


def _get(url, headers={}, params=None, retries=3, label="") -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=20)
            if r.status_code == 429:
                logger.warning(f"Rate limit — sleeping 65s ({label})")
                time.sleep(65)
                continue
            if r.status_code in (401, 403):
                logger.error(f"Auth error {r.status_code} — {label} (check key/plan)")
                return {}
            if r.status_code == 404:
                logger.debug(f"404 Not Found — {label}")
                return {}
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"[{i+1}/{retries}] {label}: {e}")
            if i < retries - 1:
                time.sleep(2 ** i)
    return {}


# ── Fixtures via football-data.org ────────────────────────────────────────────

def fetch_fixtures_today() -> pd.DataFrame:
    date = today_str()
    logger.info(f"[football-data.org] Fetching fixtures for {date}...")
    logger.info(f"  FOOTBALL_DATA_API_KEY: {'SET ✅' if FD_KEY else 'MISSING ❌'}")
    rows = []

    for code, name, country in FD_COMPETITIONS:
        data = _get(
            f"{BASE_FD}/competitions/{code}/matches",
            HEADERS_FD,
            params={"dateFrom": date, "dateTo": date},
            label=f"FD/{code}"
        )
        matches = data.get("matches", [])
        logger.info(f"  {code} ({name}): {len(matches)} matches")

        for m in matches:
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            # Só inclui jogos agendados (SCHEDULED / TIMED)
            if m.get("status") not in ("SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"):
                continue
            rows.append({
                "fixture_id":   m["id"],
                "date":         m.get("utcDate", ""),
                "league_id":    code,
                "league_name":  name,
                "country":      country,
                "home_team":    home.get("shortName") or home.get("name", "?"),
                "home_id":      home.get("id", 0),
                "away_team":    away.get("shortName") or away.get("name", "?"),
                "away_id":      away.get("id", 0),
                "venue":        "",
                "referee":      "",
                "status":       m.get("status", ""),
                "minor_focus":  False,
                "source":       "football-data.org",
            })
        time.sleep(6.5)  # respeita 10 req/min do plano free

    df = pd.DataFrame(rows)
    logger.info(f"[football-data.org] Total: {len(df)} fixtures today")
    return df


# ── Odds via The Odds API ─────────────────────────────────────────────────────

def fetch_odds_bet365() -> pd.DataFrame:
    logger.info(f"[The Odds API] Fetching Bet365 odds...")
    logger.info(f"  ODDS_API_KEY: {'SET ✅' if ODDS_KEY else 'MISSING ❌'}")
    rows = []
    found_keys = []

    for sport_key in ODDS_SPORT_KEYS:
        params = {
            "apiKey":     ODDS_KEY,
            "regions":    "eu,uk",
            "markets":    "h2h,totals",
            "bookmakers": "bet365",
            "dateFormat": "iso",
        }
        data = _get(f"{BASE_ODDS}/sports/{sport_key}/odds",
                    params=params, label=f"Odds/{sport_key}")

        if not isinstance(data, list) or not data:
            continue

        found_keys.append(sport_key)
        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            for bm in event.get("bookmakers", []):
                if bm["key"] != "bet365":
                    continue
                for mkt in bm.get("markets", []):
                    for o in mkt.get("outcomes", []):
                        rows.append({
                            "sport_key":     sport_key,
                            "event_id":      event["id"],
                            "home_team":     home,
                            "away_team":     away,
                            "commence_time": event.get("commence_time", ""),
                            "market":        mkt["key"],
                            "outcome":       o["name"],
                            "price":         float(o["price"]),
                            "point":         o.get("point"),
                        })
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if found_keys:
        logger.info(f"[The Odds API] ✅ {len(df)} odds records from: {', '.join(found_keys)}")
    else:
        logger.warning("[The Odds API] 0 events — odds not yet open for today's matches")
        logger.info("  → Normal se o pipeline rodar antes das 10h BRT ou em dia sem jogos")
    return df


def build_odds_map(odds_df: pd.DataFrame) -> dict:
    """Dict {home_lower|away_lower: {market: odd}} para lookup rápido."""
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
            elif row["outcome"] == row["home_team"]:
                omap[key]["home_win"] = price
            else:
                omap[key]["away_win"] = price
        elif mkt == "totals" and pt is not None:
            sfx = str(pt).replace(".", "")
            tag = "over" if "over" in outcome else "under"
            omap[key][f"{tag}_{sfx}"] = price

    # Fallback: atribui h2h por posição quando nomes não batem exatamente
    for key in list(omap.keys()):
        o = omap[key]
        if "home_win" not in o and "draw" not in o and "away_win" not in o:
            prices = [v for k, v in o.items() if isinstance(v, float)]
            if len(prices) >= 3:
                omap[key]["home_win"] = prices[0]
                omap[key]["draw"]     = prices[1]
                omap[key]["away_win"] = prices[2]

    return omap


# ── Persistência ──────────────────────────────────────────────────────────────

def _save_raw(df: pd.DataFrame, filename: str):
    path = Path(cfg["paths"]["data_raw"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_historical(filename: str) -> pd.DataFrame:
    path = Path(cfg["paths"]["data_historical"]) / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ── Pipeline de coleta ────────────────────────────────────────────────────────

def run_collection_pipeline() -> dict:
    logger.info("=== Starting collection pipeline ===")

    fixtures = fetch_fixtures_today()
    if not fixtures.empty:
        _save_raw(fixtures, f"fixtures_{today_str()}.csv")

    odds_df  = fetch_odds_bet365() if ODDS_KEY else pd.DataFrame()
    odds_map = build_odds_map(odds_df)
    if not odds_df.empty:
        _save_raw(odds_df, f"odds_{today_str()}.csv")

    logger.info(f"=== Collection done: {len(fixtures)} fixtures, {len(odds_map)} events with odds ===")
    return {"fixtures": fixtures, "odds": odds_df, "odds_map": odds_map}
