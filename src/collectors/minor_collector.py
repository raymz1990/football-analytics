# src/collectors/minor_collector.py — sport_keys validados da documentação oficial
import os, time, requests, pandas as pd
from typing import Dict, List
from src.utils import load_config, get_logger

logger   = get_logger("collector.minor")
cfg      = load_config()
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
BASE     = cfg["apis"]["odds_api"]["base_url"]

# ── Sport keys 100% validados em the-odds-api.com/sports-odds-data/sports-apis.html
MINOR_SPORT_KEYS = [
    # Europa
    "soccer_austria_bundesliga",
    "soccer_belgium_first_div",
    "soccer_denmark_superliga",
    "soccer_finland_veikkausliiga",
    "soccer_france_ligue_two",
    "soccer_germany_bundesliga2",
    "soccer_germany_liga3",
    "soccer_greece_super_league",
    "soccer_italy_serie_b",
    "soccer_league_of_ireland",
    "soccer_norway_eliteserien",
    "soccer_poland_ekstraklasa",
    "soccer_russia_premier_league",
    "soccer_scotland_premiership",       # chave correta para Scottish Prem
    "soccer_spl",                        # Scottish Premier League (alternativa)
    "soccer_spain_segunda_division",
    "soccer_sweden_allsvenskan",
    "soccer_sweden_superettan",
    "soccer_switzerland_superleague",
    "soccer_turkey_super_league",
    "soccer_efl_champ",
    "soccer_england_league1",
    "soccer_england_league2",
    # Américas
    "soccer_argentina_primera_division",
    "soccer_brazil_serie_b",
    "soccer_chile_campeonato",           # correto (era chile_primera_division — errado)
    "soccer_conmebol_copa_libertadores",
    "soccer_conmebol_copa_sudamericana",
    "soccer_mexico_ligamx",
    "soccer_usa_mls",
    "soccer_concacaf_leagues_cup",
    # Ásia / Oceania
    "soccer_australia_aleague",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_china_superleague",
    "soccer_saudi_arabia_pro_league",    # correto (era saudi_arabias — errado)
    # Copas
    "soccer_fa_cup",
    "soccer_germany_dfb_pokal",
    "soccer_italy_coppa_italia",
    "soccer_spain_copa_del_rey",
    "soccer_france_coupe_de_france",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_champs_league_qualification",
]


def _get(url, params, label="", retries=2) -> list:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 404:
                logger.debug(f"404 — key not active today: {label}")
                return []
            if r.status_code in (401, 403):
                logger.error(f"Auth error {r.status_code}: {label}")
                return []
            if r.status_code == 422:
                logger.debug(f"422 — market not supported: {label}")
                return []
            if r.status_code == 429:
                logger.warning(f"Rate limit — sleeping 61s ({label})")
                time.sleep(61)
                continue
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"[{i+1}/{retries}] {label}: {e}")
            if i < retries - 1:
                time.sleep(2)
    return []


def _parse(events: list, ht: bool = False) -> Dict[str, dict]:
    """Extrai odds de uma lista de eventos → {home|away: {market: odd}}"""
    prefix = "ht_" if ht else ""
    result = {}
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        key  = f"{home.lower()}|{away.lower()}"
        odds = {}
        for bm in ev.get("bookmakers", []):
            if bm["key"] != "bet365":
                continue
            for mkt in bm.get("markets", []):
                mk = mkt.get("key", "")
                for o in mkt.get("outcomes", []):
                    name  = o["name"].lower()
                    price = float(o["price"])
                    pt    = o.get("point")
                    if mk in ("h2h", "h2h_h1"):
                        if "draw" in name:
                            odds[f"{prefix}draw"] = price
                        elif o["name"] == home:
                            odds[f"{prefix}home_win"] = price
                        else:
                            odds[f"{prefix}away_win"] = price
                    elif mk in ("totals", "totals_h1") and pt is not None:
                        sfx = str(pt).replace(".", "")
                        tag = "over" if "over" in name else "under"
                        odds[f"{prefix}{tag}_{sfx}"] = price
        if odds:
            result.setdefault(key, {}).update(odds)
            result[key]["home_team"]     = home
            result[key]["away_team"]     = away
            result[key]["commence_time"] = ev.get("commence_time", "")
            result[key]["sport_key"]     = ev.get("sport_key", "")
    return result


def _infer_group(sport_key: str) -> str:
    k = sport_key
    if any(x in k for x in ["austria","belgium","denmark","finland","france_ligue_two",
                              "germany_bundesliga2","germany_liga3","greece","italy_serie_b",
                              "ireland","norway","poland","russia","scotland","spl",
                              "spain_segunda","sweden","switzerland","turkey","efl","england"]):
        return "minor_europe"
    if any(x in k for x in ["australia","japan","korea","china","saudi"]):
        return "minor_oceania"
    if any(x in k for x in ["argentina","brazil_serie_b","chile","libertadores",
                              "sudamericana","mexico","mls","concacaf"]):
        return "minor_americas"
    return "cups"


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
    logger.info("=== Minor Leagues Collection Pipeline ===")

    if not ODDS_KEY:
        logger.warning("ODDS_API_KEY not set")
        return {"fixtures": pd.DataFrame(), "odds_map": {}, "league_profiles": {}}

    all_events:    Dict[str, dict] = {}
    league_profiles: Dict[str, dict] = {}
    found = 0

    for sk in MINOR_SPORT_KEYS:
        base_params = {"apiKey": ODDS_KEY, "regions": "eu,uk,au",
                       "markets": "h2h,totals", "bookmakers": "bet365",
                       "dateFormat": "iso", "oddsFormat": "decimal"}

        ft = _get(f"{BASE}/sports/{sk}/odds", base_params, label=sk)
        if ft:
            parsed = _parse(ft, ht=False)
            for k, v in parsed.items():
                all_events.setdefault(k, {}).update(v)
            found += len(parsed)
            logger.info(f"  ✅ {sk}: {len(parsed)} events")

            # Tenta HT odds (falha silenciosamente se não disponível)
            ht_params = {**base_params, "markets": "h2h_h1,totals_h1",
                         "regions": "eu,uk"}
            ht = _get(f"{BASE}/sports/{sk}/odds", ht_params, label=f"{sk}/HT")
            if ht:
                ht_parsed = _parse(ht, ht=True)
                for k, v in ht_parsed.items():
                    all_events.setdefault(k, {}).update(v)

        time.sleep(0.25)

    # Monta DataFrame de fixtures
    rows = []
    for key, ev in all_events.items():
        sk   = ev.get("sport_key", "")
        name = sk.replace("soccer_","").replace("_"," ").title()
        rows.append({
            "fixture_id":  key,
            "date":        ev.get("commence_time", ""),
            "league_id":   sk,
            "league_name": name,
            "country":     name,
            "home_team":   ev.get("home_team", ""),
            "home_id":     0,
            "away_team":   ev.get("away_team", ""),
            "away_id":     0,
            "venue": "", "referee": "", "status": "NS",
            "minor_focus": True,
            "group":       _infer_group(sk),
            "timezone":    "UTC",
            "source":      "odds-api",
        })
        if name not in league_profiles:
            league_profiles[name] = _default_profile(name)

    # odds_map: chave = home|away, valor = {market: odd}
    odds_map = {k: {mk: v for mk, v in ev.items()
                    if mk not in ("home_team","away_team","commence_time","sport_key")}
                for k, ev in all_events.items()}

    fixtures = pd.DataFrame(rows) if rows else pd.DataFrame()
    logger.info(f"Minor pipeline: {len(fixtures)} fixtures, {len(league_profiles)} leagues, {found} events with odds")
    return {"fixtures": fixtures, "odds_map": odds_map, "league_profiles": league_profiles}
