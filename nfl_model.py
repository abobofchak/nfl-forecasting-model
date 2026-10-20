"""
NFL model v15
  Ratings : Elo + EPA/play (garbage-time filtered, pass/rush split) + multi-window form + QB ratings
  Context : rest/bye/short week, divisional, surface, roof, wind/cold, travel distance, time zones
  Output  : walk-forward backtest, feature-group ablation, next-week predictions,
            a ranked list of the games the model is most confident in, a National Weather
            Service forecast for next week's outdoor games, and one JSON "game card" per
            matchup (for the app) with injuries, weather, QBs, team stats and a points-by-feature
            explanation of every predicted margin, a likely range for each margin, a "what if a
            different QB starts" table, a prediction log that grades the model live, and a
            weekly power-rankings file with week-over-week movement, and category rankings
            (passing/rushing offense and defense, QB) that also feed each game's matchup edges,
            and a separate over/under model that predicts each game's total points. Weather is
            fetched for the app and feeds ONLY the over/under model, never the spread model.
            v11 adds five fact sections to every game card for the app (betting, scoring and pace,
            rankings and stats, situation, people and history). None of them feeds the models.
            v12 adds angle chips on every card, results.json (for the app's pick tracker),
            trends.json (league and team trends) and weather_watch.json.
            v13 adds standings.json and one page per team (cards/teams/KC.json and so on).
            v14 adds playoff_odds.json (a season simulation), highlights.json (game of the week) and a
            recap for every finished game (cards/recaps/). None of them feeds the models.

pip install nflreadpy pandas numpy scikit-learn pyarrow polars
Run weekly:  python nfl_model.py

NOTE: the first run after upgrading re-downloads play-by-play for every season
(new cache files), so it will be slow once. Later runs use the cache.
"""
import csv
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- settings
_today = datetime.now()
# Auto-detects the season (new season starts in March); set env NFL_SEASON to force one.
CURRENT_SEASON = int(os.environ.get("NFL_SEASON") or (_today.year if _today.month >= 3 else _today.year - 1))
START_SEASON = 2011     # first season loaded (first usable season is START+1)
TEST_FROM = 2014        # walk-forward backtest on seasons >= this (more seasons = more trustworthy)
MIN_TRAIN = 500         # min games needed before a walk-forward fit

CORE_STATS = ["margin", "epa_off", "epa_def"]   # epa_def = EPA allowed (lower is better)
VARIANTS = ["r8", "ewm", "blend"]
EXTRA_STATS = ["epa_off_pass", "epa_def_pass", "epa_off_rush", "epa_def_rush"]
EXTRA_VARIANTS = ["ewm"]                        # fewer windows for these, to limit noise
TOTAL_STATS = ["pf", "pa", "plays_off", "pass_rate_off"]   # scoring and pace, used by the over/under model

GARBAGE_TIME_FILTER = True    # ignore plays when win probability is <5% or >95%
USE_MARKET_FEATURES = False   # True = feed the spread line (and posted total) to the models as a feature.
                              # Accuracy goes up, but "edge vs the line" then only means
                              # "what the model adds on top of the line".
RUN_ABLATION = os.environ.get("NFL_ABLATION", "1") != "0"   # test each feature group (extra ~1 min); env NFL_ABLATION=0 skips
TOP_N = 5                     # games shown in the "most confident" lists

MODEL_VERSION = "v15"
USE_NWS_FORECAST = True       # pull the National Weather Service forecast for next week's outdoor games
NWS_USER_AGENT = (os.environ.get("NWS_USER_AGENT")
                  or "NFLModelApp/1.0 (your-email@example.com)")   # the NWS asks for contact info: put yours here or set env NWS_USER_AGENT
EXPORT_CARDS = True           # write one JSON game card per matchup for the app
CARD_DIR = "cards"
LOG_PREDICTIONS = True        # save each prediction before kickoff and grade it afterward
LOG_PATH = "prediction_log.csv"
QB_SHRINK_K = 150             # QB rating: dropbacks of "average" mixed into every QB's record
QB_PRIOR = 0.0                # QB rating: the average EPA/dropback it shrinks toward
INCLUDE_MARKET_LINES = False  # True = cards also carry the spread/moneylines and the model-vs-line gap

CACHE_DIR = "cache"
CACHE_TAG = "v3gt" if GARBAGE_TIME_FILTER else "v3"

# Override the assumed starting QB for upcoming games. By default the model
# assumes each team starts whoever started its last game. Use the name as it
# appears in play-by-play, e.g. {"KC": "P.Mahomes", "DAL": "C.Rush"}
QB_OVERRIDE = {}

# True = the model works out each team's starting QB by itself: it reads the weekly depth chart,
# skips a QB the injury report lists as Out or Doubtful, and falls back to whoever started the last
# game if the depth chart can't be read. QB_OVERRIDE (above) always wins over this.
AUTO_QB = True

# The schedule has no forecast for future games, so upcoming outdoor games are treated as
# neutral weather. Override with a forecast, e.g. {"2026_05_NYG_WAS": {"temp": 38, "wind": 18}}
WEATHER_OVERRIDE = {}

# Home city (lat, lon, UTC offset in standard time). Includes old abbreviations.
TEAM_GEO = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.76, -84.40, -5), "BAL": (39.28, -76.62, -5),
    "BUF": (42.77, -78.79, -5), "CAR": (35.23, -80.85, -5), "CHI": (41.86, -87.62, -6),
    "CIN": (39.10, -84.52, -5), "CLE": (41.51, -81.70, -5), "DAL": (32.75, -97.09, -6),
    "DEN": (39.74, -105.02, -7), "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5), "JAX": (30.32, -81.64, -5),
    "KC": (39.05, -94.48, -6), "LA": (33.95, -118.34, -8), "STL": (38.63, -90.19, -6),
    "LAC": (33.95, -118.34, -8), "SD": (32.78, -117.12, -8), "LV": (36.09, -115.18, -8),
    "OAK": (37.75, -122.20, -8), "MIA": (25.96, -80.24, -5), "MIN": (44.97, -93.26, -6),
    "NE": (42.09, -71.26, -5), "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5),
    "NYJ": (40.81, -74.07, -5), "PHI": (39.90, -75.17, -5), "PIT": (40.45, -80.02, -5),
    "SEA": (47.60, -122.33, -8), "SF": (37.40, -121.97, -8), "TB": (27.98, -82.50, -5),
    "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
}
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
TEAM_CITIES = {   # where each team's home games are played
    "ARI": "Glendale, AZ", "ATL": "Atlanta, GA", "BAL": "Baltimore, MD", "BUF": "Orchard Park, NY",
    "CAR": "Charlotte, NC", "CHI": "Chicago, IL", "CIN": "Cincinnati, OH", "CLE": "Cleveland, OH",
    "DAL": "Arlington, TX", "DEN": "Denver, CO", "DET": "Detroit, MI", "GB": "Green Bay, WI",
    "HOU": "Houston, TX", "IND": "Indianapolis, IN", "JAX": "Jacksonville, FL", "KC": "Kansas City, MO",
    "LA": "Inglewood, CA", "LAC": "Inglewood, CA", "LV": "Las Vegas, NV", "MIA": "Miami Gardens, FL",
    "MIN": "Minneapolis, MN", "NE": "Foxborough, MA", "NO": "New Orleans, LA", "NYG": "East Rutherford, NJ",
    "NYJ": "East Rutherford, NJ", "PHI": "Philadelphia, PA", "PIT": "Pittsburgh, PA", "SEA": "Seattle, WA",
    "SF": "Santa Clara, CA", "TB": "Tampa, FL", "TEN": "Nashville, TN", "WAS": "Landover, MD",
}


# ---------- Play-by-play: team EPA + QB dropbacks (cached per season) ----------
def load_pbp_tables(seasons):
    import polars as pl
    os.makedirs(CACHE_DIR, exist_ok=True)
    epa_frames, qb_frames = [], []
    for s in seasons:
        p_epa = f"{CACHE_DIR}/epa_{CACHE_TAG}_{s}.parquet"
        p_qb = f"{CACHE_DIR}/qb_{s}.parquet"
        if s < CURRENT_SEASON and os.path.exists(p_epa) and os.path.exists(p_qb):
            epa_frames.append(pd.read_parquet(p_epa))
            qb_frames.append(pd.read_parquet(p_qb))
            continue
        try:
            pbp = (
                nfl.load_pbp([s])
                .filter((pl.col("season_type") == "REG") & pl.col("epa").is_not_null())
                .select(["game_id", "posteam", "defteam", "epa", "pass", "rush",
                         "qb_dropback", "passer_player_id", "passer_player_name",
                         "wp", "qb_kneel", "qb_spike"])
                .to_pandas()
            )
        except Exception as e:  # e.g. season has no play-by-play yet
            print(f"No pbp for {s}: {e}")
            continue

        base_plays = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1))
                         & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]
        # pace uses every play (garbage time included), since it is about how many snaps a team runs
        pace = (base_plays.groupby(["game_id", "posteam"])
                .agg(plays_off=("epa", "size"), pass_rate_off=("pass", "mean"))
                .reset_index().rename(columns={"posteam": "team"}))
        plays = base_plays
        if GARBAGE_TIME_FILTER:
            plays = plays[plays["wp"].isna() | plays["wp"].between(0.05, 0.95)]
        passes, rushes = plays[plays["pass"] == 1], plays[plays["rush"] == 1]

        def agg(d, side, name):
            return (d.groupby(["game_id", side]).epa.mean().rename(name)
                    .reset_index().rename(columns={side: "team"}))

        parts = [agg(plays, "posteam", "epa_off"), agg(plays, "defteam", "epa_def"),
                 agg(passes, "posteam", "epa_off_pass"), agg(passes, "defteam", "epa_def_pass"),
                 agg(rushes, "posteam", "epa_off_rush"), agg(rushes, "defteam", "epa_def_rush")]
        epa = parts[0]
        for part in parts[1:] + [pace]:
            epa = epa.merge(part, on=["game_id", "team"], how="outer")

        # QB ratings use every dropback (no garbage-time filter)
        qbp = pbp[(pbp.qb_dropback == 1) & pbp.passer_player_id.notna()]
        qb = (qbp.groupby(["game_id", "posteam", "passer_player_id"])
              .agg(n_db=("epa", "size"), sum_epa=("epa", "sum"), qb_name=("passer_player_name", "first"))
              .reset_index().rename(columns={"posteam": "team", "passer_player_id": "qb_id"}))
        if s < CURRENT_SEASON:
            epa.to_parquet(p_epa)
            qb.to_parquet(p_qb)
        epa_frames.append(epa)
        qb_frames.append(qb)
    return pd.concat(epa_frames, ignore_index=True), pd.concat(qb_frames, ignore_index=True)


# ---------- Elo ratings (pre-game, no leakage) ----------
def add_elo(df, k=20, hfa=48, revert=1 / 3):
    ratings, last_season, pre = {}, {}, {}
    for r in df.sort_values(["gameday", "game_id"]).itertuples():
        for t in (r.home_team, r.away_team):
            if t not in ratings:
                ratings[t] = 1500.0
            elif last_season[t] != r.season:  # new season: pull toward the mean
                ratings[t] = 1500 + (ratings[t] - 1500) * (1 - revert)
            last_season[t] = r.season
        hr, ar = ratings[r.home_team], ratings[r.away_team]
        pre[r.Index] = (hr, ar)
        if pd.notna(r.result):  # only update on played games
            h = 0 if r.location == "Neutral" else hfa
            diff = hr + h - ar
            exp = 1 / (1 + 10 ** (-diff / 400))
            actual = 1.0 if r.result > 0 else 0.0 if r.result < 0 else 0.5
            m = abs(r.result)
            win_diff = diff if r.result > 0 else -diff
            mult = np.log(m + 1) * 2.2 / (win_diff * 0.001 + 2.2) if m > 0 else 1.0
            delta = k * mult * (actual - exp)
            ratings[r.home_team] += delta
            ratings[r.away_team] -= delta
    df["home_elo"] = df.index.map(lambda i: pre[i][0])
    df["away_elo"] = df.index.map(lambda i: pre[i][1])
    return df


# ---------- Schedule context: rest, roof, weather, travel, time zones ----------
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(a))


def add_schedule_features(df, weather=None):
    df = df.copy()

    # Rest: bye = coming off 13+ days (not week 1), short week = 5 days or fewer
    hr = df.home_rest.fillna(7).clip(3, 14)
    ar = df.away_rest.fillna(7).clip(3, 14)
    df["home_rest"], df["away_rest"] = hr, ar
    later = df.week > 1
    df["home_bye"] = ((hr >= 13) & later).astype(int)
    df["away_bye"] = ((ar >= 13) & later).astype(int)
    df["home_short"] = (hr <= 5).astype(int)
    df["away_short"] = (ar <= 5).astype(int)

    # Context
    df["div_game"] = pd.to_numeric(df.div_game, errors="coerce").fillna(0).astype(int)
    roof = df.roof.fillna("outdoors").astype(str).str.lower()
    df["dome"] = roof.isin(["dome", "closed"]).astype(int)
    df["grass"] = df.surface.fillna("").astype(str).str.lower().str.contains("grass").astype(int)

    # Weather (only matters outdoors; future games have no forecast, so they get 0 = neutral)
    temp = pd.to_numeric(df.temp, errors="coerce")
    wind = pd.to_numeric(df.wind, errors="coerce")
    for gid, w in (weather or {}).items():   # forecasts and manual overrides for upcoming games
        m = df.game_id == gid
        if w.get("temp") is not None:
            temp = temp.mask(m, w["temp"])
        if w.get("wind") is not None:
            wind = wind.mask(m, w["wind"])
    outdoor = (df.dome == 0).to_numpy()
    df["wind_mph"] = np.where(outdoor, wind.fillna(0), 0.0)
    df["cold"] = np.where(outdoor & temp.notna(), np.clip((45 - temp) / 10, 0, None), 0.0)

    # Travel and time zones (neutral-site games: venue unknown, so travel is set to 0)
    geo = pd.DataFrame(TEAM_GEO, index=["lat", "lon", "tz"]).T
    unknown = (set(df.home_team) | set(df.away_team)) - set(geo.index)
    if unknown:
        print(f"Warning: no location data for {sorted(unknown)}; travel features set to 0 for them")
    neutral = (df.location == "Neutral").to_numpy()
    h, a = df.home_team, df.away_team
    miles = haversine_miles(a.map(geo.lat), a.map(geo.lon), h.map(geo.lat), h.map(geo.lon))
    tz_shift = (h.map(geo.tz) - a.map(geo.tz)).to_numpy(dtype=float)   # + = away team travelled east
    df["travel_k"] = np.where(neutral | np.isnan(miles), 0.0, miles / 1000)
    df["tz_shift"] = np.where(neutral | np.isnan(tz_shift), 0.0, tz_shift)

    # West-coast-style body-clock penalty: away team travelled east for an early kickoff.
    # gametime is Eastern, so the away team's body clock reads ET hour + (away UTC offset + 5).
    et_hour = pd.to_numeric(df.gametime.astype(str).str.slice(0, 2), errors="coerce").fillna(13)
    away_body = et_hour + a.map(geo.tz).fillna(-5) + 5
    df["west_early"] = ((away_body < 11) & (df.tz_shift > 0) & ~neutral).astype(int)
    return df


# ---------- Team form: 3 windows per stat ----------
def add_team_features(long):
    long = long.sort_values(["team", "gameday"]).reset_index(drop=True)
    long["gp"] = long.groupby(["team", "season"]).cumcount()  # games played so far this season
    for s in CORE_STATS + EXTRA_STATS + TOTAL_STATS:
        prior = (long.groupby(["team", "season"])[s].mean().reset_index()
                 .assign(season=lambda d: d.season + 1)
                 .rename(columns={s: f"{s}_prior"}))  # last season's full-year average
        long = long.merge(prior, on=["team", "season"], how="left")
        g = long.groupby("team")[s]
        long[f"{s}_r8"] = g.transform(lambda x: x.shift().rolling(8, min_periods=3).mean())
        long[f"{s}_ewm"] = g.transform(lambda x: x.shift().ewm(halflife=12, min_periods=3).mean())
        std = long.groupby(["team", "season"])[s].transform(lambda x: x.shift().expanding().mean())
        # blend: early season leans on last year, shifts to this year as games are played
        w = long.gp / (long.gp + 8)
        long[f"{s}_blend"] = np.where(
            long.gp == 0, long[f"{s}_prior"], w * std.fillna(0) + (1 - w) * long[f"{s}_prior"]
        )
    return long



# ---------- Automatic starting QB: depth chart + injury report ----------
def auto_qb_starters(upcoming):
    """{team: (gsis_id, name)} for next week's games, or {} if the depth chart can't be read.
    Takes each team's QB1 from the newest depth chart up to this week, then steps down the chart past
    any QB the injury report lists as Out or Doubtful. Teams with no usable answer are left out and
    keep the old rule (whoever started their last game)."""
    if not AUTO_QB or upcoming is None or upcoming.empty:
        return {}
    try:
        season, week = int(upcoming.season.iloc[0]), int(upcoming.week.iloc[0])
        teams = set(upcoming.home_team) | set(upcoming.away_team)
        dc = nfl.load_depth_charts([season]).to_pandas()
        tcol = next((c for c in ("club_code", "team") if c in dc.columns), None)
        pcol = next((c for c in ("depth_position", "depth_chart_position", "position") if c in dc.columns), None)
        if tcol is None or pcol is None or "gsis_id" not in dc.columns or "depth_team" not in dc.columns:
            return {}
        dc = dc[dc[pcol].astype(str).str.upper() == "QB"].copy()
        if "week" in dc.columns:
            wk = pd.to_numeric(dc.week, errors="coerce")
            dc = dc[wk <= week]
            if dc.empty:
                return {}
            dc = dc[pd.to_numeric(dc.week, errors="coerce") == pd.to_numeric(dc.week, errors="coerce").max()]
        for c in ("dt", "update_dt", "last_updated"):          # several snapshots in a week: newest only
            if c in dc.columns and dc[c].notna().any():
                dc = dc[dc[c] == dc[c].max()]
                break
        dc["rank"] = pd.to_numeric(dc.depth_team, errors="coerce")
        dc = dc.dropna(subset=["rank", "gsis_id"]).sort_values("rank")
        namecol = next((c for c in ("full_name", "football_name") if c in dc.columns), None)

        out_ids = set()
        try:
            inj = nfl.load_injuries([season]).to_pandas()
            inj = inj[(inj.week == week) & inj.report_status.isin(["Out", "Doubtful"])]
            out_ids = set(inj.gsis_id.dropna()) if "gsis_id" in inj.columns else set()
        except Exception:
            pass

        picks = {}
        for team in teams:
            cand = dc[dc[tcol] == team].drop_duplicates("gsis_id")
            for r in cand.itertuples():
                if r.gsis_id in out_ids:
                    continue
                picks[team] = (str(r.gsis_id), str(getattr(r, namecol)) if namecol else str(r.gsis_id))
                break
        return picks
    except Exception as e:
        print(f"  (auto starting-QB lookup unavailable, using last game's starter: {e})")
        return {}

# ---------- QB ratings: career EPA/dropback, shrunk toward average ----------
def add_qb_features(long, qb_all, k=QB_SHRINK_K, prior=QB_PRIOR, auto=None):
    qb_all = qb_all.merge(long[["game_id", "team", "gameday"]].drop_duplicates(),
                          on=["game_id", "team"], how="left")
    qb_all = qb_all.sort_values(["qb_id", "gameday"])
    qb_all["cn"] = qb_all.groupby("qb_id").n_db.cumsum() - qb_all.n_db        # dropbacks before this game
    qb_all["cs"] = qb_all.groupby("qb_id").sum_epa.cumsum() - qb_all.sum_epa  # EPA before this game
    qb_all["rating"] = (qb_all.cs + k * prior) / (qb_all.cn + k)

    starters = (qb_all.sort_values("n_db").groupby(["game_id", "team"]).tail(1)
                [["game_id", "team", "qb_id", "rating"]].rename(columns={"rating": "qb_rating_played"}))
    tot = qb_all.groupby("qb_id")[["n_db", "sum_epa"]].sum()
    final_rating = (tot.sum_epa + k * prior) / (tot.n_db + k)  # rating as of today
    id2name = qb_all.drop_duplicates("qb_id", keep="last").set_index("qb_id").qb_name
    name2id = (qb_all.sort_values("gameday").drop_duplicates("qb_name", keep="last")
               .set_index("qb_name").qb_id)

    long = long.sort_values(["team", "gameday"]).reset_index(drop=True)
    long = long.merge(starters, on=["game_id", "team"], how="left")
    unplayed = long.margin.isna()
    last = long.groupby("team").qb_id.ffill()  # last known starter
    ov = long.team.map({t: name2id.get(n) for t, n in QB_OVERRIDE.items()})
    auto = auto or {}
    au = long.team.map({t: v[0] for t, v in auto.items()})
    for _id, _nm in auto.values():             # a starter with no pass attempts yet still needs a name
        if _id not in id2name.index:
            id2name.loc[_id] = _nm
    # priority for upcoming games: your override, then the depth chart, then last game's starter
    long["qb_used"] = long.qb_id.where(~unplayed, ov.fillna(au).fillna(last))
    rating = long.qb_rating_played.where(~unplayed, long.qb_used.map(final_rating))
    long["qb_rating"] = rating.fillna(prior)
    prev = long.groupby("team").qb_used.shift(1)
    long["qb_chg"] = ((long.qb_used != prev) & prev.notna() & long.qb_used.notna()).astype(int)
    long["qb_name"] = long.qb_used.map(id2name)
    return long


def report_auto_qbs(long, auto):
    """Print where the automatic pick differs from the old 'last game's starter' rule, so you can sanity-check it."""
    if not auto:
        return
    last = (long[long.margin.notna()].dropna(subset=["qb_id"]).sort_values("gameday").groupby("team").qb_id.last())
    diff = [(t, v[1]) for t, v in sorted(auto.items()) if last.get(t) is not None and last.get(t) != v[0]]
    print(f"\nAuto starting QBs: {len(auto)} teams from the depth chart/injury report.")
    for t, n in diff:
        print(f"  {t}: {n} (different from last game's starter)")


def next_week_games(df):
    """The next unplayed week of games."""
    up = df[df.result.isna()].sort_values("gameday")
    if up.empty:
        return up
    first = up.iloc[0]
    return up[(up.season == first.season) & (up.week == first.week)].copy()


def team_snapshot(long, feat_cols, at=None):
    """Each team's ratings going into its next game. at=(season, week) instead gives the state
    at the start of that week (a team on a bye that week simply carries its current ratings)."""
    if at is None:
        rows = long[long.margin.isna()]
    else:
        s, w = at
        rows = long[(long.season > s) | ((long.season == s) & (long.week >= w))]
    nxt = rows.sort_values("gameday").groupby("team").head(1).set_index("team")
    snap = nxt[[c for c in feat_cols if c in nxt.columns]].copy()
    snap["elo"] = nxt["elo"]
    snap["qb_id"] = nxt["qb_used"]   # assumed starter
    snap["qb_name"] = nxt["qb_name"]
    last_qb = (long[long.margin.notna()].dropna(subset=["qb_id"]).sort_values("gameday")
               .groupby("team").qb_id.last())
    snap["last_qb_id"] = snap.index.map(last_qb)   # started the team's most recent played game
    return snap


def build_dataset():
    seasons = list(range(START_SEASON, CURRENT_SEASON + 1))
    df = nfl.load_schedules(seasons).to_pandas()
    df = df[df.game_type == "REG"].copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.reset_index(drop=True)

    df = add_elo(df)

    # Weather for next week's games: NWS forecast, then any manual overrides on top
    weather = {gid: dict(w) for gid, w in fetch_forecasts(next_week_games(df)).items()}
    for gid, w in WEATHER_OVERRIDE.items():
        weather[gid] = {**weather.get(gid, {}), **w, "source": "manual_override"}
    df = add_schedule_features(df, weather)

    epa, qb_all = load_pbp_tables(seasons)
    qb_dir = build_qb_directory(qb_all, df)

    # One row per team per game (home and away stacked)
    base_cols = ["game_id", "season", "gameday"]
    home = df[base_cols + ["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"})
    away = df[base_cols + ["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"})
    long = pd.concat([home, away], ignore_index=True)

    long["margin"] = long.pf - long.pa
    long = long.merge(epa, on=["game_id", "team"], how="left")
    long = add_team_features(long)
    auto_qb = auto_qb_starters(next_week_games(df))
    long = add_qb_features(long, qb_all, auto=auto_qb)
    report_auto_qbs(long, auto_qb)
    elo_map = pd.concat([
        df[["game_id", "home_team", "home_elo"]].rename(columns={"home_team": "team", "home_elo": "elo"}),
        df[["game_id", "away_team", "away_elo"]].rename(columns={"away_team": "team", "away_elo": "elo"}),
    ])
    long = long.merge(elo_map, on=["game_id", "team"], how="left").merge(
        df[["game_id", "week"]], on="game_id", how="left")

    core_feat = [f"{s}_{v}" for s in CORE_STATS for v in VARIANTS] + ["qb_rating"]
    extra_feat = [f"{s}_{v}" for s in EXTRA_STATS for v in EXTRA_VARIANTS]
    total_feat = [f"{s}_{v}" for s in TOTAL_STATS for v in VARIANTS]
    feat_cols = core_feat + extra_feat
    keep = ["game_id", "team"] + feat_cols + total_feat + ["qb_chg", "qb_name"]
    df = df.merge(long[keep].add_prefix("h_"), left_on=["game_id", "home_team"],
                  right_on=["h_game_id", "h_team"])
    df = df.merge(long[keep].add_prefix("a_"), left_on=["game_id", "away_team"],
                  right_on=["a_game_id", "a_team"])

    # ----- spread model inputs: home minus away. Weather is deliberately NOT used here. -----
    for c in feat_cols:
        df[f"{c}_diff"] = df[f"h_{c}"] - df[f"a_{c}"]
    df["elo_diff"] = df.home_elo - df.away_elo
    df["rest_diff"] = df["home_rest"] - df["away_rest"]
    df["neutral"] = (df.location == "Neutral").astype(int)

    base = ([f"{c}_diff" for c in core_feat]
            + ["elo_diff", "rest_diff", "neutral", "h_qb_chg", "a_qb_chg"])
    if USE_MARKET_FEATURES:
        base.append("spread_line")
    groups = {
        "rest/bye": ["home_bye", "away_bye", "home_short", "away_short"],
        "div/roof/surface": ["div_game", "dome", "grass"],
        "travel/time zone": ["travel_k", "tz_shift", "west_early"],
        "pass/rush EPA": [f"{c}_diff" for c in extra_feat],
    }

    # ----- over/under model inputs: combined for both teams, plus pace and weather -----
    df["total_actual"] = df.home_score + df.away_score
    if "total_line" not in df.columns:
        df["total_line"] = np.nan
    for v in VARIANTS:
        # each team's expected points = average of its scoring and the opponent's points allowed
        df[f"exp_total_{v}"] = (df[f"h_pf_{v}"] + df[f"a_pa_{v}"] + df[f"a_pf_{v}"] + df[f"h_pa_{v}"]) / 2
        df[f"epa_off_sum_{v}"] = df[f"h_epa_off_{v}"] + df[f"a_epa_off_{v}"]
        df[f"epa_def_sum_{v}"] = df[f"h_epa_def_{v}"] + df[f"a_epa_def_{v}"]   # EPA allowed: higher = more scoring
    for s in EXTRA_STATS:
        df[f"{s}_sum_ewm"] = df[f"h_{s}_ewm"] + df[f"a_{s}_ewm"]
    df["plays_sum_ewm"] = df.h_plays_off_ewm + df.a_plays_off_ewm
    df["pass_rate_avg_ewm"] = (df.h_pass_rate_off_ewm + df.a_pass_rate_off_ewm) / 2
    df["qb_rating_sum"] = df.h_qb_rating + df.a_qb_rating
    df["short_any"] = df.home_short + df.away_short
    df["wind_x_pass"] = df.wind_mph / 10 * df.pass_rate_avg_ewm   # wind hurts pass-heavy games most
    df["cold_x_pass"] = df.cold * df.pass_rate_avg_ewm

    total_base = [f"exp_total_{v}" for v in VARIANTS] + ["dome", "neutral"]
    if USE_MARKET_FEATURES:
        total_base.append("total_line")
    total_groups = {
        "EPA (overall)": [f"epa_off_sum_{v}" for v in VARIANTS] + [f"epa_def_sum_{v}" for v in VARIANTS],
        "pass/rush EPA": [f"{s}_sum_ewm" for s in EXTRA_STATS],
        "pace": ["plays_sum_ewm", "pass_rate_avg_ewm"],
        "quarterbacks": ["qb_rating_sum"],
        "context": ["grass", "div_game", "short_any"],
        "weather": ["wind_mph", "cold", "wind_x_pass", "cold_x_pass"],
    }
    all_cols = (base + [c for g in groups.values() for c in g]
                + total_base + [c for g in total_groups.values() for c in g])
    up = next_week_games(df)
    prev_at = (int(up.season.iloc[0]), int(up.week.iloc[0]) - 1) if len(up) and int(up.week.iloc[0]) > 1 else None
    snap_cols = feat_cols + [f"{s}_r8" for s in EXTRA_STATS]   # extra last-8-game columns for category rankings
    extras = {"snapshot": team_snapshot(long, snap_cols),
              "snapshot_prev": team_snapshot(long, snap_cols, at=prev_at) if prev_at else None,
              "weather": weather, "qb_dir": qb_dir,
              "total_base": total_base, "total_groups": total_groups,
              "epa_by_game": long.set_index(["game_id", "team"])[["epa_off", "epa_def", "plays_off", "pass_rate_off"]].to_dict("index")}
    extras["schedule"] = df[["game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
                             "home_score", "away_score", "result", "spread_line", "total_line"]].copy()
    return df.dropna(subset=all_cols), base, groups, extras


# ---------- Modeling helpers ----------
def implied(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def payout(ml):  # profit per 1 unit staked on a win
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, 100 / (-ml), ml / 100)


def make_models():
    reg = make_pipeline(StandardScaler(), Ridge(alpha=100))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    return reg, clf


# ---------- Walk-forward backtest ----------
def walk_forward(played, X_cols):
    """For each week, train only on games played before it, then predict that week."""
    out = []
    for _, wk in played[played.season >= TEST_FROM].groupby(["season", "week"]):
        train = played[played.gameday < wk.gameday.min()]
        if len(train) < MIN_TRAIN:
            continue
        reg, clf = make_models()
        reg.fit(train[X_cols], train.result)
        clf.fit(train[X_cols], train.home_win)
        w = wk.copy()
        w["pred_margin"] = reg.predict(wk[X_cols])
        w["p_home_model"] = clf.predict_proba(wk[X_cols])[:, 1]
        out.append(w)
    return pd.concat(out)


def backtest_report(wf):
    ph, pa_ = implied(wf.home_moneyline), implied(wf.away_moneyline)
    wf["p_mkt"] = ph / (ph + pa_)
    print(f"Walk-forward backtest: {len(wf)} games, seasons {wf.season.min()}-{wf.season.max()}")
    print(f"Spread MAE  model {mean_absolute_error(wf.result, wf.pred_margin):.2f} | "
          f"market {mean_absolute_error(wf.result, wf.spread_line):.2f}")
    print(f"Win logloss model {log_loss(wf.home_win, wf.p_home_model):.4f} | "
          f"market {log_loss(wf.home_win, wf.p_mkt):.4f}")

    print("\nSpread bets at -110 (bet when |model - line| >= threshold):")
    wf["edge"] = wf.pred_margin - wf.spread_line
    wf["cm"] = wf.result - wf.spread_line  # >0 means home covered
    for t in (0, 1, 2, 3):
        b = wf[(wf.edge.abs() >= t) & (wf.cm != 0) & (wf.edge != 0)]
        win = np.where(b.edge > 0, b.cm > 0, b.cm < 0)
        profit = np.where(win, 100 / 110, -1.0)
        print(f"  >= {t} pts: {len(b):4d} bets | win {win.mean():.1%} | ROI {profit.mean():+.1%}")

    print("\nMoneyline bets at market odds (bet when |model - market prob| >= threshold):")
    wf["ml_edge"] = wf.p_home_model - wf.p_mkt
    for t in (0.02, 0.05, 0.08):
        b = wf[wf.ml_edge.abs() >= t]
        home = b.ml_edge > 0
        win = np.where(home, b.home_win == 1, b.home_win == 0)
        pay = np.where(home, payout(b.home_moneyline), payout(b.away_moneyline))
        profit = np.where(win, pay, -1.0)
        print(f"  >= {t:.0%}: {len(b):4d} bets | win {win.mean():.1%} | ROI {profit.mean():+.1%}")

    print("\nCalibration (model home-win prob vs actual):")
    bins = pd.cut(wf.p_home_model, [0, .3, .4, .5, .6, .7, 1])
    print(wf.groupby(bins, observed=True).agg(games=("home_win", "size"),
          predicted=("p_home_model", "mean"), actual=("home_win", "mean")).round(3).to_string())

    # How often the model's pick wins straight up, by how confident it was
    print("\nStraight-up accuracy by model confidence (pick = side the model gives >50%):")
    p_pick = np.maximum(wf.p_home_model, 1 - wf.p_home_model)
    model_right = (wf.p_home_model > 0.5) == (wf.home_win == 1)
    mkt_right = (wf.p_mkt > 0.5) == (wf.home_win == 1)
    rows = []
    for lo in (0.5, 0.6, 0.7, 0.8):
        m = p_pick >= lo
        rows.append({"confidence >=": f"{lo:.0%}", "games": int(m.sum()),
                     "model right": f"{model_right[m].mean():.1%}",
                     "market right (same games)": f"{mkt_right[m].mean():.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))


def ablation(played, base, groups):
    """Does each new feature group improve out-of-sample accuracy? Same games in every row."""
    print("\nFeature-group ablation (walk-forward; lower is better):")
    sets = ([("base only", base)]
            + [(f"base + {g}", base + cols) for g, cols in groups.items()]
            + [("all features", base + [c for cols in groups.values() for c in cols])])
    rows, wf = [], None
    for name, cols in sets:
        wf = walk_forward(played, cols)
        rows.append({"feature set": name, "n_features": len(cols),
                     "MAE": mean_absolute_error(wf.result, wf.pred_margin),
                     "logloss": log_loss(wf.home_win, wf.p_home_model)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"(market MAE on the same games: {mean_absolute_error(wf.result, wf.spread_line):.4f}. "
          "Differences under ~0.05 MAE are within noise; keep a group only if it helps consistently.)")


# ---------- Over/under model: predicts each game's total points ----------
def make_total_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=100))


def walk_forward_total(played_t, T_cols):
    """Same idea as the spread backtest: train only on games before each week, predict that week."""
    out = []
    for _, wk in played_t[played_t.season >= TEST_FROM].groupby(["season", "week"]):
        train = played_t[played_t.gameday < wk.gameday.min()]
        if len(train) < MIN_TRAIN:
            continue
        m = make_total_model()
        m.fit(train[T_cols], train.total_actual)
        w = wk.copy()
        w["pred_total"] = m.predict(wk[T_cols])
        out.append(w)
    return pd.concat(out)


def total_backtest_report(wf):
    err = wf.pred_total - wf.total_actual
    print(f"\nOver/under backtest: {len(wf)} games, seasons {wf.season.min()}-{wf.season.max()}")
    print(f"Total MAE  model {mean_absolute_error(wf.total_actual, wf.pred_total):.2f} | "
          f"posted line {mean_absolute_error(wf.total_actual, wf.total_line):.2f} | "
          f"model bias (predicted minus actual) {err.mean():+.2f}")
    print("\nOver/under bets at -110 (bet when |model total - line| >= threshold):")
    edge = wf.pred_total - wf.total_line
    cm = wf.total_actual - wf.total_line   # > 0 means the game went over
    for t in (0, 1, 2, 3):
        m = (edge.abs() >= t) & (cm != 0) & (edge != 0)
        if not m.any():
            print(f"  >= {t} pts:    0 bets")
            continue
        win = np.where(edge[m] > 0, cm[m] > 0, cm[m] < 0)
        profit = np.where(win, 100 / 110, -1.0)
        print(f"  >= {t} pts: {int(m.sum()):4d} bets | win {win.mean():.1%} | ROI {profit.mean():+.1%}")


def total_ablation(played_t, base, groups):
    """Does each input group help predict totals? In particular: does weather?"""
    print("\nOver/under feature-group ablation (walk-forward; lower is better):")
    sets = ([("base only", base)]
            + [(f"base + {g}", base + cols) for g, cols in groups.items()]
            + [("all features", base + [c for cols in groups.values() for c in cols])])
    rows, wf = [], None
    for name, cols in sets:
        wf = walk_forward_total(played_t, cols)
        rows.append({"feature set": name, "n_features": len(cols),
                     "MAE": mean_absolute_error(wf.total_actual, wf.pred_total)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"(posted-total MAE on the same games: {mean_absolute_error(wf.total_actual, wf.total_line):.4f}. "
          "Differences under ~0.05 are within noise.)")


def backtest_total_summary(wf):
    out = {"games": int(len(wf)), "seasons": f"{wf.season.min()}-{wf.season.max()}",
           "avg_total_error": float(mean_absolute_error(wf.total_actual, wf.pred_total)),
           "bias": float((wf.pred_total - wf.total_actual).mean())}
    out["market"] = {"line_mae": float(mean_absolute_error(wf.total_actual, wf.total_line))}
    return out


def totals_table(upcoming, pt, margin, unc_t):
    away, home = upcoming.away_team.tolist(), upcoming.home_team.tolist()
    t = upcoming[["gameday", "away_team", "home_team"]].copy()
    t["projected_score"] = [f"{a} {(p - m) / 2:.0f} - {h} {(p + m) / 2:.0f}"
                            for a, h, p, m in zip(away, home, pt, margin)]
    t["pred_total"] = np.round(pt, 1)
    t["likely_range"] = [f"{p + unc_t['q10']:.0f} to {p + unc_t['q90']:.0f}" for p in pt]
    t["total_line"] = upcoming.total_line.to_numpy()
    t["total_edge"] = (t.pred_total - t.total_line).round(1)   # + = model expects more points than the line
    t["lean"] = np.where(t.total_edge.isna(), "", np.where(t.total_edge > 0, "Over", "Under"))
    return t


def top_totals(t, n=TOP_N):
    e = t[t.total_edge.notna()].assign(abs_edge=lambda d: d.total_edge.abs())
    e = e.sort_values("abs_edge", ascending=False).head(n)
    print(f"\nTOP {len(e)} BIGGEST DISAGREEMENTS WITH THE POSTED TOTAL (model total vs line):")
    print(e[["away_team", "home_team", "pred_total", "total_line", "total_edge", "lean"]].to_string(index=False))
    print("  Caution: check the over/under bet ROI in the backtest before treating these as bets.")


# ---------- Most-confident picks ----------
def top_picks(out, n=TOP_N):
    d = out.copy()
    home = d.p_home_model >= 0.5
    d["game"] = d.away_team + " @ " + d.home_team
    d["pick"] = np.where(home, d.home_team, d.away_team)
    d["win_prob"] = np.where(home, d.p_home_model, 1 - d.p_home_model)
    d["model_margin"] = np.where(home, d.pred_margin, -d.pred_margin)   # pick wins by this much
    d["line"] = np.where(home, d.spread_line, -d.spread_line)           # + = pick favoured by this much
    d["mkt_prob"] = np.where(home, d.p_home_market, 1 - d.p_home_market)
    d["agree"] = (d.pred_margin > 0) == home   # margin model and win-prob model name the same winner

    best = d[d.agree].sort_values("win_prob", ascending=False).head(n)
    print(f"\nTOP {len(best)} MOST CONFIDENT PICKS (highest model win probability; both models agree):")
    print(best[["game", "pick", "win_prob", "model_margin", "line", "mkt_prob"]]
          .round(3).to_string(index=False))
    print("  win_prob = model chance the pick wins | model_margin = predicted winning margin |")
    print("  line = how much the market favours the pick (negative = underdog) | mkt_prob = market win chance")
    print("  See 'Straight-up accuracy by model confidence' in the backtest for how this tier has performed.")

    e = d.assign(abs_edge=d.spread_edge.abs(),
                 lean=np.where(d.spread_edge > 0, d.home_team, d.away_team))
    e = e.sort_values("abs_edge", ascending=False).head(n)
    print(f"\nTOP {len(e)} BIGGEST DISAGREEMENTS WITH THE POINT SPREAD (model margin vs line):")
    print(e[["game", "lean", "pred_margin", "spread_line", "spread_edge"]].to_string(index=False))
    print("  Caution: check the spread-bet ROI in the backtest before treating these as bets.")


# ---------- Weather forecast: US National Weather Service (free, no API key) ----------
NWS_BASE = "https://api.weather.gov"


def _nws_get(url, retries=2, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": NWS_USER_AGENT,
                                               "Accept": "application/geo+json"})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise last


def _kickoff_utc(gameday, gametime):
    """Schedule kickoff times are Eastern; return a UTC timestamp."""
    hhmm = str(gametime) if pd.notna(gametime) and ":" in str(gametime) else "13:00"
    local = pd.Timestamp(f"{pd.Timestamp(gameday):%Y-%m-%d} {hhmm}")
    return local.tz_localize("America/New_York").tz_convert("UTC")


def _wind_mph(text):
    if isinstance(text, dict):
        text = text.get("value", "")
    nums = [int(n) for n in re.findall(r"\d+", str(text))]
    return float(max(nums)) if nums else 0.0   # "10 to 20 mph" -> 20, "Calm" -> 0


def _forecast_for_kickoff(periods, kickoff_utc, hours=3):
    """Summarize the hourly forecast over the ~3 hours of the game."""
    end = kickoff_utc + pd.Timedelta(hours=hours)
    rows = []
    for p in periods:
        s = pd.Timestamp(p["startTime"]).tz_convert("UTC")
        e = pd.Timestamp(p["endTime"]).tz_convert("UTC")
        if s < end and e > kickoff_utc:
            t = p.get("temperature")
            if isinstance(t, dict):
                t = t.get("value")
            if t is not None and str(p.get("temperatureUnit", "F")).upper() == "C":
                t = t * 9 / 5 + 32
            pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
            rows.append((t, _wind_mph(p.get("windSpeed")), pop, p.get("shortForecast")))
    if not rows:
        return None   # kickoff is outside the forecast window (about 6.5 days out)
    temps = [r[0] for r in rows if r[0] is not None]
    pops = [r[2] for r in rows if r[2] is not None]
    return {"temp": float(np.mean(temps)) if temps else None,
            "wind": float(max(r[1] for r in rows)),
            "precip_pct": float(max(pops)) if pops else 0.0,
            "summary": rows[0][3]}


def fetch_forecasts(games):
    """NWS hourly forecast at each outdoor game's home city. Returns {game_id: info}.
    Fails soft: any problem just means that game is treated as neutral weather."""
    out = {}
    if not USE_NWS_FORECAST or games.empty:
        return out
    if "your-email@example.com" in NWS_USER_AGENT:
        print("Note: set NWS_USER_AGENT to include your contact email (the NWS asks for this).")
    cache, outdoor, failures, warned = {}, 0, 0, False
    for g in games.itertuples():
        roof = str(g.roof).lower() if pd.notna(g.roof) else "outdoors"
        if roof in ("dome", "closed") or g.location == "Neutral" or g.home_team not in TEAM_GEO:
            continue
        outdoor += 1
        if failures >= 3:
            continue   # NWS looks unreachable; stop trying
        try:
            if g.home_team not in cache:
                lat, lon, _ = TEAM_GEO[g.home_team]
                pts = _nws_get(f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}")
                cache[g.home_team] = _nws_get(pts["properties"]["forecastHourly"])
                time.sleep(0.3)   # be polite to a free public API
            props = cache[g.home_team]["properties"]
            f = _forecast_for_kickoff(props["periods"], _kickoff_utc(g.gameday, g.gametime))
            failures = 0
            if f and f["temp"] is not None:
                f["source"] = "nws_forecast"
                f["forecast_updated"] = props.get("generatedAt") or props.get("updateTime")
                out[g.game_id] = f
        except Exception as e:
            failures += 1
            if not warned:
                print(f"NWS forecast lookup failed for {g.home_team}: {e}")
                warned = True
    print(f"NWS forecast loaded for {len(out)} of {outdoor} outdoor games "
          "(the rest are treated as neutral weather).")
    return out


def _fmt_weather(w):
    if not w:
        return "no forecast (treated as neutral)"
    parts = []
    if w.get("temp") is not None:
        parts.append(f"{w['temp']:.0f}F")
    if w.get("wind") is not None:
        parts.append(f"wind {w['wind']:.0f} mph")
    if w.get("precip_pct") is not None:
        parts.append(f"precip {w['precip_pct']:.0f}%")
    if w.get("summary"):
        parts.append(str(w["summary"]))
    if w.get("source") == "manual_override":
        parts.append("[manual override]")
    return ", ".join(parts)


def weather_report(upcoming, weather):
    print("\nGame-day weather used by the model (NWS forecast unless noted):")
    for r in upcoming.itertuples():
        line = "indoors" if r.dome == 1 else _fmt_weather(weather.get(r.game_id))
        print(f"  {r.away_team} @ {r.home_team}: {line}")


# ---------- Injury report (informational, not a model feature) ----------
INJ_STATUSES = ["Out", "Doubtful", "Questionable"]


def get_injuries(upcoming):
    try:
        season, week = int(upcoming.season.iloc[0]), int(upcoming.week.iloc[0])
        teams = set(upcoming.home_team) | set(upcoming.away_team)
        inj = nfl.load_injuries([season]).to_pandas()
        inj = inj[(inj.week == week) & inj.team.isin(teams)]
        practice = inj["practice_status"].fillna("").astype(str) if "practice_status" in inj.columns else pd.Series("", index=inj.index)
        # players with a game status, plus anyone who practiced less than fully
        inj = inj[inj.report_status.isin(INJ_STATUSES) | ((practice != "") & ~practice.str.contains("Full", case=False))]
        return inj, None
    except Exception as e:
        return None, str(e)


def injury_report(upcoming, inj, note):
    if inj is None:
        print(f"\n(Injury data unavailable: {note})")
        return
    week = int(upcoming.week.iloc[0])
    inj = inj[inj.report_status.isin(["Out", "Doubtful"])]
    print(f"\nInjury report, Out/Doubtful, week {week}:")
    if inj.empty:
        print("  none listed yet (reports usually post midweek)")
    for team, g in inj.groupby("team"):
        names = ", ".join(f"{r.full_name} ({r.position})" for r in g.itertuples())
        print(f"  {team}: {names}")
        if (g.position == "QB").any():
            print(f"  ** {team} has a QB listed on the injury report (AUTO_QB steps past Out/Doubtful QBs) **")


# ---------- "Why": split each predicted margin into per-feature contributions ----------
STAT_LABELS = {   # order matters: pass/rush names must be checked before the plain epa_ names
    "epa_off_pass": ("Passing offense EPA/play", "Offense"),
    "epa_def_pass": ("Pass defense EPA allowed/play", "Defense"),
    "epa_off_rush": ("Rushing offense EPA/play", "Offense"),
    "epa_def_rush": ("Rush defense EPA allowed/play", "Defense"),
    "epa_off": ("Offense EPA/play", "Offense"),
    "epa_def": ("Defense EPA allowed/play", "Defense"),
    "margin": ("Scoring margin", "Team strength"),
}
WINDOW_LABELS = {"r8": "last 8 games", "ewm": "recent-weighted", "blend": "season blend"}
FIXED_LABELS = {
    "elo_diff": ("Elo rating", "Team strength"),
    "qb_rating_diff": ("Quarterback rating", "Quarterback"),
    "h_qb_chg": ("Home team changed QB", "Quarterback"),
    "a_qb_chg": ("Away team changed QB", "Quarterback"),
    "rest_diff": ("Rest days (home minus away)", "Rest & schedule"),
    "home_bye": ("Home team off a bye", "Rest & schedule"),
    "away_bye": ("Away team off a bye", "Rest & schedule"),
    "home_short": ("Home team on a short week", "Rest & schedule"),
    "away_short": ("Away team on a short week", "Rest & schedule"),
    "neutral": ("Neutral-site game", "Venue & situation"),
    "div_game": ("Divisional game", "Venue & situation"),
    "dome": ("Indoor stadium", "Venue & situation"),
    "grass": ("Natural grass field", "Venue & situation"),
    "travel_k": ("Away team travel distance (thousand miles)", "Travel"),
    "tz_shift": ("Time zones the away team crossed eastward", "Travel"),
    "west_early": ("West-coast team, early body-clock kickoff", "Travel"),
    "wind_mph": ("Wind speed", "Weather"),
    "cold": ("Cold (below 45F)", "Weather"),
    "wind_x_qb": ("Wind x QB-rating gap", "Weather"),
    "cold_x_qb": ("Cold x QB-rating gap", "Weather"),
    "spread_line": ("Market spread (model input)", "Market"),
}


TOTAL_LABELS = {
    "plays_sum_ewm": ("Combined plays per game (pace)", "Pace"),
    "pass_rate_avg_ewm": ("Average pass rate", "Pace"),
    "qb_rating_sum": ("Combined quarterback ratings", "Quarterback"),
    "short_any": ("Teams on a short week", "Rest & schedule"),
    "wind_x_pass": ("Wind x pass rate", "Weather"),
    "cold_x_pass": ("Cold x pass rate", "Weather"),
    "total_line": ("Posted total (model input)", "Market"),
    "epa_off_pass_sum_ewm": ("Combined passing offense EPA/play", "Offense"),
    "epa_def_pass_sum_ewm": ("Combined pass defense EPA allowed/play", "Defense"),
    "epa_off_rush_sum_ewm": ("Combined rushing offense EPA/play", "Offense"),
    "epa_def_rush_sum_ewm": ("Combined rush defense EPA allowed/play", "Defense"),
}
for _v in VARIANTS:
    TOTAL_LABELS[f"exp_total_{_v}"] = (f"Expected points from both teams' scoring ({WINDOW_LABELS[_v]})", "Scoring form")
    TOTAL_LABELS[f"epa_off_sum_{_v}"] = (f"Combined offense EPA/play ({WINDOW_LABELS[_v]})", "Offense")
    TOTAL_LABELS[f"epa_def_sum_{_v}"] = (f"Combined defense EPA allowed/play ({WINDOW_LABELS[_v]})", "Defense")


def feature_meta(name):
    if name in FIXED_LABELS:
        return FIXED_LABELS[name]
    if name in TOTAL_LABELS:
        return TOTAL_LABELS[name]
    base = name[:-5] if name.endswith("_diff") else name
    for stat, (label, group) in STAT_LABELS.items():
        if base.startswith(stat + "_"):
            return f"{label} ({WINDOW_LABELS.get(base[len(stat) + 1:], base[len(stat) + 1:])})", group
    return name, "Other"


def explain_games(upcoming, X_cols, reg):
    """Ridge on standardized inputs is additive: margin = intercept + sum(coef * z-score).
    Each term is that feature's contribution in points (positive = favors the home team)."""
    scaler, ridge = reg[0], reg[-1]
    X = upcoming[X_cols].to_numpy(dtype=float)
    contrib = (X - scaler.mean_) / scaler.scale_ * ridge.coef_
    intercept = float(ridge.intercept_)
    return X, contrib, intercept


# ---------- Game cards (one JSON file per matchup, for the app) ----------
INJ_POS_PRIORITY = {"QB": 0, "T": 1, "OT": 1, "G": 1, "C": 1, "OL": 1, "DE": 2, "DT": 2, "EDGE": 2,
                    "OLB": 2, "LB": 2, "CB": 2, "S": 2, "DB": 2, "WR": 3, "RB": 3, "TE": 3}
INJ_STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2}
TEAM_STATS = [   # (snapshot column, label, higher is better)
    ("elo", "Elo rating", True),
    ("margin_ewm", "Scoring margin (recent form)", True),
    ("epa_off_ewm", "Offense EPA/play", True),
    ("epa_def_ewm", "Defense EPA allowed/play (lower is better)", False),
    ("epa_off_pass_ewm", "Passing offense EPA/play", True),
    ("epa_def_pass_ewm", "Pass defense EPA allowed/play (lower is better)", False),
    ("epa_off_rush_ewm", "Rushing offense EPA/play", True),
    ("epa_def_rush_ewm", "Rush defense EPA allowed/play (lower is better)", False),
    ("qb_rating", "Quarterback rating (EPA/dropback)", True),
]
BLIND_SPOTS = [
    "Injuries and inactives (shown for context only; they do not change the prediction)",
    "Confirmed starting quarterbacks (the model assumes each team starts its last game's starter)",
    "Depth charts, snap counts and coaching changes",
    "Betting market movement and public betting trends",
]


def jsonable(o):
    """Convert numpy/pandas values to plain JSON types; NaN/inf become null."""
    if o is None or o is pd.NaT:
        return None
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 3)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (int, str)):
        return o
    return str(o)


# ---------- Team trends: record, streaks, scoring, and results against the spread and total ----------
def _ats(margin, team_spread):
    """'W' covered, 'L' failed to cover, 'P' push, '' no line. team_spread is negative for a favorite."""
    if pd.isna(margin) or pd.isna(team_spread):
        return ""
    c = margin + team_spread
    return "W" if c > 0 else "L" if c < 0 else "P"


def _ou(total, line):
    if pd.isna(total) or pd.isna(line):
        return ""
    return "O" if total > line else "U" if total < line else "P"


def _fmt_rec(w, l, p=0):
    return f"{w}-{l}" + (f"-{p}" if p else "")


def _streak(vals):
    """vals are most recent first. A push or tie ends a streak; blanks (no line) are skipped."""
    vals = [x for x in vals if x]
    if not vals or vals[0] in ("P", "T"):
        return None
    n = 1
    while n < len(vals) and vals[n] == vals[0]:
        n += 1
    return {"type": vals[0], "length": n, "display": f"{vals[0]}{n}"}


def team_game_log(hist):
    """One row per team per played game, with its result against the spread and the total."""
    h = hist[hist.result.notna()]
    parts = []
    for is_home in (True, False):
        sgn = 1 if is_home else -1
        parts.append(pd.DataFrame({
            "game_id": h.game_id, "season": h.season, "week": h.week, "gameday": h.gameday,
            "team": h.home_team if is_home else h.away_team, "opp": h.away_team if is_home else h.home_team,
            "is_home": is_home, "pf": h.home_score if is_home else h.away_score,
            "pa": h.away_score if is_home else h.home_score, "margin": sgn * h.result,
            "team_spread": -sgn * h.spread_line,   # negative = this team was the favorite
            "total_line": h.total_line, "total_pts": h.total_actual}))
    tg = pd.concat(parts, ignore_index=True)
    cover = tg.margin + tg.team_spread
    tg["cover"] = cover
    tg["ats"] = np.where(tg.team_spread.isna(), "", np.where(cover > 0, "W", np.where(cover < 0, "L", "P")))
    tg["fav"] = np.where(tg.team_spread < 0, 1.0, np.where(tg.team_spread > 0, 0.0, np.nan))
    ok = tg.total_line.notna() & tg.total_pts.notna()
    tg["ou"] = np.where(~ok, "", np.where(tg.total_pts > tg.total_line, "O",
                                         np.where(tg.total_pts < tg.total_line, "U", "P")))
    tg["win"] = np.where(tg.margin > 0, "W", np.where(tg.margin < 0, "L", "T"))
    return tg


def _season_games(tg, team, season, before):
    return tg[(tg.team == team) & (tg.season == season) & (tg.gameday < before)].sort_values("gameday")


def team_form(tg, team, season, before, is_home):
    """Record, streak and scoring this season. No betting lines involved."""
    s = _season_games(tg, team, season, before)
    out = {"games_played": int(len(s))}
    if s.empty:
        return out
    res = s.win.tolist()[::-1]   # most recent first
    l5 = s.tail(5)
    loc = s[s.is_home == is_home]
    lres = loc.win.tolist()
    out.update({
        "record": _fmt_rec(res.count("W"), res.count("L"), res.count("T")),
        "streak": _streak(res),
        "points_per_game": {"scored": s.pf.mean(), "allowed": s.pa.mean(),
                            "scored_last5": l5.pf.mean(), "allowed_last5": l5.pa.mean(),
                            "avg_margin": s.margin.mean()},
        "at_this_venue": {"venue": "home" if is_home else "away", "games": int(len(loc)),
                          "record": _fmt_rec(lres.count("W"), lres.count("L"), lres.count("T")),
                          "scored": loc.pf.mean() if len(loc) else None,
                          "allowed": loc.pa.mean() if len(loc) else None}})
    return out


def _ats_block(d):
    v = [x for x in d.ats.tolist() if x]
    w, l, p = v.count("W"), v.count("L"), v.count("P")
    return {"record": _fmt_rec(w, l, p), "covered": w, "failed": l, "pushed": p, "games": len(v),
            "cover_pct": w / (w + l) if (w + l) else None}


def _ou_block(d):
    v = [x for x in d.ou.tolist() if x]
    o, u, p = v.count("O"), v.count("U"), v.count("P")
    return {"record": _fmt_rec(o, u, p), "overs": o, "unders": u, "pushes": p, "games": len(v),
            "over_pct": o / (o + u) if (o + u) else None}


def team_betting(tg, team, season, before, is_home, is_fav):
    """Against-the-spread and over/under history this season. is_fav = this team's role today
    (None for a pick'em or when no line is posted)."""
    s = _season_games(tg, team, season, before)
    a, o = s[s.ats != ""], s[s.ou != ""]
    venue = "home" if is_home else "away"
    ats = {"season": _ats_block(a), "last5": _ats_block(a.tail(5)),
           "venue": {"venue": venue, **_ats_block(a[a.is_home == is_home])},
           "streak": _streak(a.ats.tolist()[::-1]),
           "avg_cover_margin": float(a.cover.mean()) if len(a) else None}
    if is_fav is not None:
        ats["role"] = {"role": "favorite" if is_fav else "underdog",
                       **_ats_block(a[a.fav == (1.0 if is_fav else 0.0)])}
    ou = {"season": _ou_block(o), "last5": _ou_block(o.tail(5)),
          "venue": {"venue": venue, **_ou_block(o[o.is_home == is_home])},
          "streak": _streak(o.ou.tolist()[::-1]),
          "avg_game_total": float(o.total_pts.mean()) if len(o) else None,
          "avg_posted_total": float(o.total_line.mean()) if len(o) else None}
    return {"ats": ats, "over_under": ou}


def trend_angles(team, b, is_home):
    """Short plain-English notes for the top of a matchup. Small samples, so they are trivia, not prediction."""
    out = []
    where = "at home" if is_home else "on the road"

    def lopsided(w, l, n=3, t=0.75):
        d = w + l
        return d >= n and (w / d >= t or l / d >= t)

    def add(games, text):
        out.append({"team": team, "text": text, "games": games})

    v = b["ats"]["venue"]
    if lopsided(v["covered"], v["failed"]):
        add(v["games"], f"{team} is {v['covered']}-{v['failed']} against the spread {where} this season")
    r = b["ats"].get("role")
    if r and lopsided(r["covered"], r["failed"]):
        who = "a favorite" if r["role"] == "favorite" else "an underdog"
        add(r["games"], f"{team} is {r['covered']}-{r['failed']} against the spread as {who} this season")
    l5 = b["ats"]["last5"]
    if l5["games"] >= 5 and lopsided(l5["covered"], l5["failed"], n=5, t=0.8):
        add(5, f"{team} has covered {l5['covered']} of its last 5 games" if l5["covered"] > l5["failed"]
            else f"{team} has failed to cover {l5['failed']} of its last 5 games")
    st = b["ats"]["streak"]
    if st and st["length"] >= 3:
        add(st["length"], f"{team} has {'covered' if st['type'] == 'W' else 'failed to cover'} {st['length']} games in a row")
    ov = b["over_under"]["venue"]
    if lopsided(ov["overs"], ov["unders"]):
        wh = "home" if is_home else "road"
        add(ov["games"], f"The over is {ov['overs']}-{ov['unders']} in {team}'s {wh} games" if ov["overs"] > ov["unders"]
            else f"The under is {ov['unders']}-{ov['overs']} in {team}'s {wh} games")
    ost = b["over_under"]["streak"]
    if ost and ost["length"] >= 3:
        add(ost["length"], f"The {'over' if ost['type'] == 'O' else 'under'} has hit in {team}'s last {ost['length']} games")
    return out


def recent_games(hist, team, before, epa_by_game, n=5, market=False):
    m = ((hist.home_team == team) | (hist.away_team == team)) & hist.result.notna() & (hist.gameday < before)
    out = []
    for r in hist[m].sort_values("gameday", ascending=False).head(n).itertuples():
        home = r.home_team == team
        pf, pa = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
        e = epa_by_game.get((r.game_id, team), {})
        entry = {"date": f"{r.gameday:%Y-%m-%d}", "opponent": r.away_team if home else r.home_team,
                 "home_away": "home" if home else "away", "points_for": pf, "points_against": pa,
                 "result": "W" if pf > pa else "L" if pf < pa else "T",
                 "epa_off": e.get("epa_off"), "epa_def": e.get("epa_def")}
        if market:   # how the game went against the spread and total
            ts = -r.spread_line if home else r.spread_line
            entry.update({"team_spread": ts, "ats": _ats(pf - pa, ts) or None,
                          "total_line": r.total_line, "total_points": pf + pa,
                          "ou": _ou(pf + pa, r.total_line) or None})
        out.append(entry)
    return out


def head_to_head(hist, home, away, before, n=5, market=False):
    m = (((hist.home_team == home) & (hist.away_team == away))
         | ((hist.home_team == away) & (hist.away_team == home))) & hist.result.notna() & (hist.gameday < before)
    out = []
    for r in hist[m].sort_values("gameday", ascending=False).head(n).itertuples():
        entry = {"date": f"{r.gameday:%Y-%m-%d}", "home": r.home_team, "away": r.away_team,
                 "home_score": r.home_score, "away_score": r.away_score,
                 "total_points": r.home_score + r.away_score,
                 "winner": r.home_team if r.result > 0 else r.away_team if r.result < 0 else "tie"}
        if market:
            sl = r.spread_line
            fav = None if pd.isna(sl) or sl == 0 else (r.home_team if sl > 0 else r.away_team)
            entry.update({"favorite_team": fav, "favorite_spread": None if pd.isna(sl) else abs(sl),
                          "spread_home": r.spread_line, "home_ats": _ats(r.result, -r.spread_line) or None,
                          "total_line": r.total_line, "ou": _ou(r.home_score + r.away_score, r.total_line) or None})
        out.append(entry)
    return out


def _injury_entries(inj, team):
    if inj is None or inj.empty:
        return []
    rows = [{"name": r.get("full_name"), "position": r.get("position"), "status": r.get("report_status"),
             "injury": r.get("report_primary_injury") or r.get("practice_primary_injury"),
             "practice": r.get("practice_status"),
             "updated": (str(r.get("date_modified"))[:16] if r.get("date_modified") is not None
                         and pd.notna(r.get("date_modified")) else None)}
            for r in inj[inj.team == team].to_dict("records")]
    rows.sort(key=lambda x: (INJ_POS_PRIORITY.get(x["position"], 4), INJ_STATUS_ORDER.get(x["status"], 3)))
    return rows


def _badges(r, w, qb_inj):
    b = []
    if r["h_qb_chg"] or r["a_qb_chg"]:
        b.append("qb_change")
    if qb_inj:
        b.append("qb_injury")
    if r["dome"] == 0 and w:
        text = str(w.get("summary") or "").lower()
        if (w.get("wind") or 0) >= 15:
            b.append("windy")
        if w.get("temp") is not None and w["temp"] <= 32:
            b.append("freezing")
        if (w.get("precip_pct") or 0) >= 50 or any(k in text for k in ("rain", "snow", "shower", "storm")):
            b.append("precipitation")
    if r["home_short"] or r["away_short"]:
        b.append("short_week")
    if r["home_bye"] or r["away_bye"]:
        b.append("off_bye")
    if r["div_game"]:
        b.append("divisional")
    if r["west_early"]:
        b.append("west_coast_early")
    if r["travel_k"] >= 1.5:
        b.append("long_travel")
    if r["neutral"]:
        b.append("neutral_site")
    return b


# ---------- Matchup sections: the five groups of facts the app shows for each game ----------
CTX_TAG = "c1"
CTX_YEARS = 3                  # seasons of play-by-play counting stats loaded (referee penalty rates use all three)
PLAYOFF_PUSH_FROM_WEEK = 10    # playoff-picture status is shown from this week on
COLD_F, WINDY_MPH = 40, 15     # a game counts as "cold" at or below this, "windy" at or above this
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SF", "SEA"],
}
CTX_WANT = ["game_id", "season_type", "posteam", "drive", "fixed_drive_result", "drive_inside20", "yardline_100",
            "qb_dropback", "sack", "interception", "fumble_lost", "third_down_converted", "third_down_failed",
            "penalty", "penalty_team"]
CTX_SUM = ["sacks_taken", "dropbacks", "giveaways", "third_conv", "third_att", "rz_trips", "rz_td", "penalties"]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)


def context_from_pbp(pbp):
    """One row per team per game: sacks taken, dropbacks, giveaways, 3rd downs, red-zone drives, penalties."""
    p = pbp[pbp.season_type == "REG"] if "season_type" in pbp.columns else pbp
    o = p[p.posteam.notna()]
    g = pd.DataFrame({"game_id": o.game_id, "team": o.posteam, "sacks_taken": _num(o, "sack"),
                      "dropbacks": _num(o, "qb_dropback"),
                      "giveaways": _num(o, "interception") + _num(o, "fumble_lost"),
                      "third_conv": _num(o, "third_down_converted"), "third_fail": _num(o, "third_down_failed")})
    agg = g.groupby(["game_id", "team"])[["sacks_taken", "dropbacks", "giveaways", "third_conv", "third_fail"]].sum()
    agg["third_att"] = agg.third_conv + agg.third_fail
    agg = agg.drop(columns="third_fail")
    agg["rz_trips"], agg["rz_td"] = 0.0, 0.0
    if "drive" in o.columns and "fixed_drive_result" in o.columns:
        if "drive_inside20" in o.columns:
            inside = _num(o, "drive_inside20") == 1
        else:
            inside = pd.to_numeric(o.get("yardline_100"), errors="coerce") <= 20
        d = pd.DataFrame({"game_id": o.game_id, "team": o.posteam, "drive": o.drive, "res": o.fixed_drive_result})
        d = d[inside & d.drive.notna()].drop_duplicates(["game_id", "team", "drive"])
        agg["rz_trips"] = d.groupby(["game_id", "team"]).size().reindex(agg.index).fillna(0)
        agg["rz_td"] = d[d.res == "Touchdown"].groupby(["game_id", "team"]).size().reindex(agg.index).fillna(0)
    agg["penalties"] = 0.0
    if "penalty_team" in p.columns:
        pen = p[(_num(p, "penalty") == 1) & p.penalty_team.notna()]
        pc = pen.groupby(["game_id", "penalty_team"]).size()
        pc.index = pc.index.set_names(["game_id", "team"])
        agg["penalties"] = pc.reindex(agg.index).fillna(0)
    return agg.reset_index()


def load_context(seasons):
    """Counting stats from play-by-play (separate cache from the EPA tables)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    for s in seasons:
        path = f"{CACHE_DIR}/ctx_{CTX_TAG}_{s}.parquet"
        if s < CURRENT_SEASON and os.path.exists(path):
            frames.append(pd.read_parquet(path))
            continue
        try:
            raw = nfl.load_pbp([s])
            keep = [c for c in CTX_WANT if c in raw.columns]
            ctx = context_from_pbp(raw.select(keep).to_pandas())
        except Exception as e:
            print(f"No team counting stats for {s}: {e}")
            continue
        ctx["season"] = s
        if s < CURRENT_SEASON:
            ctx.to_parquet(path)
        frames.append(ctx)
    return pd.concat(frames, ignore_index=True) if frames else None


def season_stat_table(ctx, tg, season):
    """Each team's season totals: turnover margin, sack rates, red zone, 3rd down (offense and defense)."""
    if ctx is None or ctx.empty:
        return None
    base = ctx.drop(columns=["season"], errors="ignore")
    c = tg[tg.season == season][["game_id", "team", "opp"]].merge(base, on=["game_id", "team"], how="inner")
    if c.empty:
        return None
    oc = base.rename(columns=lambda k: k if k == "game_id" else ("opp" if k == "team" else "o_" + k))
    c = c.merge(oc, on=["game_id", "opp"], how="left")
    sm = c.groupby("team")[CTX_SUM + ["o_" + k for k in CTX_SUM]].sum()
    sm["games"] = c.groupby("team").game_id.nunique()

    def ratio(a, b):
        return (a / b).where(b > 0)

    t = pd.DataFrame(index=sm.index)
    t["games"] = sm.games
    t["turnover_margin"] = sm.o_giveaways - sm.giveaways
    t["giveaways"], t["takeaways"] = sm.giveaways, sm.o_giveaways
    t["sack_rate_off"] = ratio(sm.sacks_taken, sm.dropbacks)       # share of dropbacks ending in a sack (lower is better)
    t["sack_rate_def"] = ratio(sm.o_sacks_taken, sm.o_dropbacks)   # share of opposing dropbacks ending in a sack
    t["rz_off"], t["rz_off_trips"] = ratio(sm.rz_td, sm.rz_trips), sm.rz_trips
    t["rz_def"], t["rz_def_trips"] = ratio(sm.o_rz_td, sm.o_rz_trips), sm.o_rz_trips
    t["third_off"], t["third_def"] = ratio(sm.third_conv, sm.third_att), ratio(sm.o_third_conv, sm.o_third_att)
    t["penalties_pg"] = sm.penalties / sm.games
    return t


STAT_HIGHER_IS_BETTER = {"turnover_margin": True, "sack_rate_off": False, "sack_rate_def": True,
                         "rz_off": True, "rz_def": False, "third_off": True, "third_def": False}


def team_stat_block(stat_tbl, team):
    if stat_tbl is None or team not in stat_tbl.index:
        return None
    out = {"games": int(stat_tbl.at[team, "games"])}
    for k in ("giveaways", "takeaways", "rz_off_trips", "rz_def_trips", "penalties_pg"):
        out[k] = stat_tbl.at[team, k]
    for k, hi in STAT_HIGHER_IS_BETTER.items():
        col = stat_tbl[k]
        v = col.get(team)
        rk = col.rank(ascending=not hi, method="min").get(team)
        out[k] = {"value": v, "rank": int(rk) if pd.notna(rk) else None, "of": int(col.notna().sum())}
    return out


def category_ranks(tables, team):
    out = {}
    for key in ("passing_offense", "passing_defense", "rushing_offense", "rushing_defense", "offense", "defense"):
        t = (tables or {}).get(key)
        if t is not None and team in t.index and pd.notna(t.at[team, "rank"]):
            out[key] = {"rank": int(t.at[team, "rank"]), "of": int(t["rank"].notna().sum()),
                        "value": t.at[team, "value"], "last8_rank": _i(t.at[team, "last8_rank"])}
    return out


def standings_table(tg, season, before):
    """Playoff picture from results so far: seed, games behind the last wild card. Simple tiebreak: win%, then point margin."""
    s = tg[(tg.season == season) & (tg.gameday < before)]
    if s.empty:
        return {}
    rec = s.assign(w=(s.win == "W").astype(int), l=(s.win == "L").astype(int), t=(s.win == "T").astype(int)) \
           .groupby("team").agg(w=("w", "sum"), l=("l", "sum"), t=("t", "sum"), diff=("margin", "sum"))
    out = {}
    for conf in ("AFC", "NFC"):
        teams = [t for d, ts in DIVISIONS.items() if d.startswith(conf) for t in ts]
        df = rec.reindex(teams).fillna(0)
        df["pct"] = (df.w + 0.5 * df.t) / (df.w + df.l + df.t).clip(lower=1)
        order = ["pct", "diff"]
        leaders = [df.loc[ts].sort_values(order, ascending=False).index[0]
                   for d, ts in DIVISIONS.items() if d.startswith(conf)]
        lead = df.loc[leaders].sort_values(order, ascending=False)
        rest = df.drop(index=leaders).sort_values(order, ascending=False)
        seeded = list(lead.index) + list(rest.index[:3])
        last = df.loc[seeded[-1]]
        for t in teams:
            r = df.loc[t]
            seed = seeded.index(t) + 1 if t in seeded else None
            gb = 0.0 if seed else max(((last.w - last.l) - (r.w - r.l)) / 2, 0.0)
            out[t] = {"seed": seed, "division_leader": t in leaders, "games_back": gb,
                      "record": _fmt_rec(int(r.w), int(r.l), int(r.t))}
    return out


def playoff_status(e, week):
    if week < PLAYOFF_PUSH_FROM_WEEK or not e:
        return {"available": False, "note": f"Playoff picture is shown from Week {PLAYOFF_PUSH_FROM_WEEK}."}
    if e["seed"]:
        label = (f"Leading its division (No. {e['seed']} seed)" if e["division_leader"]
                 else f"Holding a wild-card spot (No. {e['seed']} seed)")
        push = True
    else:
        gb = float(e["games_back"])
        back = f"{gb:g} game{'' if gb == 1 else 's'} back"
        if gb <= 1.5:
            label, push = f"In the hunt ({back} of the last spot)", True
        elif gb <= 3:
            label, push = f"On the fringe ({back})", True
        else:
            label, push = f"Long shot ({back})", False
    return {"available": True, "label": label, "seed": e["seed"], "games_back": e["games_back"],
            "division_leader": e["division_leader"], "in_push": push}


def rest_label(days, bye, short):
    if pd.isna(days):
        return None
    d = int(days)
    if bye:
        return f"Off a bye ({d} days)"
    if short:
        return f"Short week ({d} days)"
    if d >= 9:
        return f"Extra rest ({d} days)"
    return f"Normal week ({d} days)"


def condition_frame(tg, hist):
    """tg plus the weather and roof of each past game, to build records in cold, wind and indoors."""
    c = hist[["game_id", "temp", "wind", "roof"]].copy()
    roof = c.roof.fillna("outdoors").astype(str).str.lower()
    temp, wind = pd.to_numeric(c.temp, errors="coerce"), pd.to_numeric(c.wind, errors="coerce")
    outdoors = ~roof.isin(["dome", "closed"])
    c["is_indoor"] = ~outdoors
    c["is_cold"] = outdoors & (temp <= COLD_F)
    c["is_windy"] = outdoors & (wind >= WINDY_MPH)
    return tg.merge(c[["game_id", "is_indoor", "is_cold", "is_windy"]], on="game_id", how="left")


def condition_records(tgc, team, season, before, market):
    s = tgc[(tgc.team == team) & (tgc.season >= season - 2) & (tgc.gameday < before)]
    out = {}
    for key, col, label in (("indoors", "is_indoor", "In domes / closed roofs"),
                            ("cold", "is_cold", f"In the cold ({COLD_F}°F or colder)"),
                            ("windy", "is_windy", f"In the wind ({WINDY_MPH}+ mph)")):
        d = s[s[col].fillna(False).astype(bool)]
        w = (d.win == "W").sum()
        l = (d.win == "L").sum()
        e = {"label": label, "games": int(len(d)), "record": _fmt_rec(int(w), int(l), int((d.win == "T").sum()))}
        if market:
            e["ats"] = _ats_block(d[d.ats != ""])["record"]
        out[key] = e
    return out


def last_games_detail(tg, team, before, market, n=5):
    s = tg[(tg.team == team) & (tg.gameday < before)].sort_values("gameday").tail(n).iloc[::-1]
    out = []
    for r in s.itertuples():
        e = {"date": f"{r.gameday:%Y-%m-%d}", "season": int(r.season), "week": int(r.week), "opponent": r.opp,
             "home_away": "home" if r.is_home else "away", "points_for": r.pf, "points_against": r.pa,
             "result": r.win, "team_spread": None, "ats": None, "cover_margin": None,
             "total_line": None, "total_points": r.pf + r.pa, "ou": None}
        if market:
            e.update({"team_spread": r.team_spread, "ats": r.ats or None,
                      "cover_margin": r.cover if r.ats else None,
                      "total_line": r.total_line, "ou": r.ou or None})
        out.append(e)
    return out


def scoring_block(tg, epa_by_game, team, season, before, ranks):
    s = _season_games(tg, team, season, before)

    def one(d):
        if d.empty:
            return None
        plays = pd.Series([epa_by_game.get((g, team), {}).get("plays_off") for g in d.game_id], dtype=float)
        prate = pd.Series([epa_by_game.get((g, team), {}).get("pass_rate_off") for g in d.game_id], dtype=float)
        ok = plays.notna() & prate.notna()
        return {"games": int(len(d)), "ppg_for": d.pf.mean(), "ppg_against": d.pa.mean(),
                "avg_total": (d.pf + d.pa).mean(),
                "plays_pg": plays.mean() if plays.notna().any() else None,
                "pass_rate": float((prate[ok] * plays[ok]).sum() / plays[ok].sum()) if ok.any() and plays[ok].sum() else None}

    out = {"season": one(s), "last5": one(s.tail(5))}
    if ranks is not None and team in ranks.index:
        out["ppg_for_rank"] = int(ranks.at[team, "for_rank"])
        out["ppg_against_rank"] = int(ranks.at[team, "against_rank"])   # 1 = fewest points allowed
        out["of"] = int(len(ranks))
    return out


def scoring_ranks(tg, season, before):
    s = tg[(tg.season == season) & (tg.gameday < before)]
    if s.empty:
        return None
    g = s.groupby("team").agg(pf=("pf", "mean"), pa=("pa", "mean"))
    g["for_rank"] = g.pf.rank(ascending=False, method="min")
    g["against_rank"] = g.pa.rank(ascending=True, method="min")
    return g


def referee_profile(hist, ctx, ref, season, before, market):
    if not ref or pd.isna(ref) or "referee" not in hist.columns:
        return None
    h = hist[(hist.referee == ref) & hist.result.notna() & (hist.gameday < before) & (hist.season >= season - CTX_YEARS + 1)]
    out = {"name": str(ref), "games": int(len(h)), "penalties_per_game": None, "league_penalties_per_game": None,
           "over_record": None, "over_pct": None}
    if ctx is not None and not ctx.empty:
        per_game = ctx.groupby("game_id").penalties.sum()
        mine = per_game.reindex(h.game_id).dropna()
        league_ids = hist[hist.result.notna() & (hist.gameday < before) & (hist.season >= season - CTX_YEARS + 1)].game_id
        lg = per_game.reindex(league_ids).dropna()
        if len(mine):
            out["penalties_per_game"] = float(mine.mean())
        if len(lg):
            out["league_penalties_per_game"] = float(lg.mean())
    if market and len(h):
        ok = h.total_line.notna() & h.total_actual.notna()
        o = int((h[ok].total_actual > h[ok].total_line).sum())
        u = int((h[ok].total_actual < h[ok].total_line).sum())
        out["over_record"] = f"{o}-{u}"
        out["over_pct"] = o / (o + u) if (o + u) else None
    out["note"] = "Last three seasons. Penalties are flags on the play-by-play record, both teams combined."
    return out


def build_sections(r, S, wx, bh, ba, injuries_by_side, market):
    """The five groups of facts for one game. S holds data shared by every game this week."""
    home, away, tg, hist = r["home_team"], r["away_team"], S["tg"], S["hist"]
    sea, kd, week = int(r["season"]), r["gameday"], int(r["week"])
    sides = (("home", home, True, bh), ("away", away, False, ba))
    sc_ranks = scoring_ranks(tg, sea, kd)

    # 1) betting
    betting = {"available": market,
               "note": None if market else "Set INCLUDE_MARKET_LINES = True in nfl_model.py to include spread and total records."}
    for side, team, is_home, b in sides:
        e = {"last5": last_games_detail(tg, team, kd, market)}
        if market and b:
            e.update({"ats": b["ats"], "over_under": b["over_under"]})
        betting[side] = e

    # 2) scoring and pace
    scoring = {}
    for side, team, is_home, b in sides:
        e = scoring_block(tg, S["epa_by_game"], team, sea, kd, sc_ranks)
        if market and b:
            e["over_share"] = {"record": b["over_under"]["season"]["record"], "over_pct": b["over_under"]["season"]["over_pct"],
                               "games": b["over_under"]["season"]["games"],
                               "avg_game_total": b["over_under"]["avg_game_total"],
                               "avg_posted_total": b["over_under"]["avg_posted_total"]}
        scoring[side] = e

    # 3) rankings and stats
    rankings = {"counting_stats_available": S["stat_tbl"] is not None}
    for side, team, is_home, b in sides:
        rankings[side] = {"categories": category_ranks(S["tables"], team), "stats": team_stat_block(S["stat_tbl"], team)}

    # 4) situation
    tzc = r["tz_shift"]
    prior = hist[(((hist.home_team == home) & (hist.away_team == away)) | ((hist.home_team == away) & (hist.away_team == home)))
                 & (hist.season == sea) & (hist.gameday < kd) & hist.result.notna()].sort_values("gameday")
    rematch = None
    if len(prior):
        p = prior.iloc[-1]
        rematch = {"date": f"{p.gameday:%Y-%m-%d}", "home": p.home_team, "away": p.away_team,
                   "home_score": p.home_score, "away_score": p.away_score,
                   "winner": p.home_team if p.result > 0 else p.away_team if p.result < 0 else "tie"}
    temp_now, wind_now = wx.get("temp_f"), wx.get("wind_mph")
    today = {"indoors": bool(wx.get("indoors")),
             "cold": bool(not wx.get("indoors") and temp_now is not None and temp_now <= COLD_F),
             "windy": bool(not wx.get("indoors") and wind_now is not None and wind_now >= WINDY_MPH)}
    pp = {side: playoff_status(S["stand"].get(team), week) for side, team, _, _ in sides}
    situation = {"game": {"divisional": bool(r["div_game"]), "rematch": rematch, "conditions_today": today,
                          "neutral_site": bool(r["neutral"]),
                          "playoff_implications": any(v.get("in_push") for v in pp.values()),
                          "playoff_note": f"Playoff picture uses a simple tiebreak (win %, then point margin); real tiebreakers are more involved."},
                 "away_travel_miles": r["travel_k"] * 1000, "time_zones_crossed_east": tzc,
                 "west_coast_early_body_clock": bool(r["west_early"])}
    for side, team, is_home, b in sides:
        days = r["home_rest"] if is_home else r["away_rest"]
        bye = bool(r["home_bye"] if is_home else r["away_bye"])
        short = bool(r["home_short"] if is_home else r["away_short"])
        situation[side] = {"rest_days": days, "off_bye": bye, "short_week": short, "rest_label": rest_label(days, bye, short),
                           "records": condition_records(S["tgc"], team, sea, kd, market), "playoff": pp[side]}

    # 5) people and history
    people = {}
    for side, team, is_home, b in sides:
        pre = "h_" if is_home else "a_"
        alts = qb_alternatives(S["qb_dir"], S["snap"], team, sea, S["weeks_played"], n=1)
        backup = next(({"name": n, "rating": rt} for qid, n, rt in alts if qid is not None), None)
        people[side] = {"starter": {"name": r[pre + "qb_name"], "rating": r[pre + "qb_rating"],
                                    "changed_from_last_game": bool(r[pre + "qb_chg"]), "assumed": True},
                        "backup": backup, "injuries": injuries_by_side[side]}
    people["meetings"] = head_to_head(hist, home, away, kd, n=3, market=market)
    people["meetings_note"] = "Regular-season games only, newest first."
    people["referee"] = referee_profile(hist, S["ctx"], r.get("referee"), sea, kd, market)
    people["referee_note"] = None if people["referee"] else "Referee crew not assigned yet (usually posted midweek)."
    return {"betting": betting, "scoring": scoring, "rankings": rankings, "situation": situation, "people": people}


def build_cards(upcoming, X_cols, reg, clf, extras, inj, inj_note, hist, unc, tables=None, tot=None, ctx=None):
    snap, weather, epa_by_game = extras["snapshot"], extras["weather"], extras["epa_by_game"]
    X, contrib, intercept = explain_games(upcoming, X_cols, reg)
    pred = reg.predict(upcoming[X_cols])
    if not np.allclose(intercept + contrib.sum(axis=1), pred, atol=1e-6):
        print("Warning: per-feature explanation does not add up to the prediction")
    p_home = clf.predict_proba(upcoming[X_cols])[:, 1]
    if tot:   # over/under model: same additive explanation as the spread model
        Xt, contrib_t, intercept_t = explain_games(upcoming, tot["cols"], tot["model"])
        pred_t = tot["model"].predict(upcoming[tot["cols"]])
        unc_t = tot["unc"]
        if not np.allclose(intercept_t + contrib_t.sum(axis=1), pred_t, atol=1e-6):
            print("Warning: over/under explanation does not add up to the prediction")
    ranks = {c: snap[c].rank(ascending=not hi, method="min") for c, _, hi in TEAM_STATS if c in snap.columns}
    n_ranked = {c: int(snap[c].notna().sum()) for c in ranks}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    qb_dir = extras["qb_dir"]
    tg = team_game_log(hist)   # every team's results against the spread and total
    s0 = int(upcoming.season.iloc[0])
    weeks_played = int(hist[(hist.season == s0) & hist.result.notna()].week.nunique())
    shared = {"tg": tg, "tgc": condition_frame(tg, hist), "hist": hist, "epa_by_game": epa_by_game, "ctx": ctx,
              "stat_tbl": season_stat_table(ctx, tg, s0), "tables": tables, "snap": snap, "qb_dir": qb_dir,
              "weeks_played": weeks_played,
              "stand": standings_table(tg, s0, upcoming.gameday.min() + pd.Timedelta(days=1))}

    def profile(team):
        if team not in snap.index:
            return []
        out = []
        for c, label, hi in TEAM_STATS:
            if c in snap.columns:
                rk = ranks[c].get(team)
                out.append({"stat": c, "label": label, "value": snap.at[team, c],
                            "rank": int(rk) if pd.notna(rk) else None, "of": n_ranked[c],
                            "higher_is_better": hi})
        return out

    cards = []
    for i, (_, r) in enumerate(upcoming.iterrows()):
        home, away = r["home_team"], r["away_team"]
        margin, ph = float(pred[i]), float(p_home[i])
        fav = home if margin >= 0 else away
        kick = _kickoff_utc(r["gameday"], r.get("gametime"))
        hhmm = str(r.get("gametime")) if pd.notna(r.get("gametime")) else "13:00"

        feats = []
        for j, name in enumerate(X_cols):
            label, group = feature_meta(name)
            e = {"name": name, "label": label, "group": group, "value": X[i, j],
                 "points": contrib[i, j], "favors": home if contrib[i, j] >= 0 else away}
            if name.endswith("_diff") and f"h_{name[:-5]}" in upcoming.columns:
                e["home_value"], e["away_value"] = r[f"h_{name[:-5]}"], r[f"a_{name[:-5]}"]
            feats.append(e)
        feats_sorted = sorted(feats, key=lambda e: -abs(e["points"]))
        group_pts = {}
        for e in feats:
            group_pts[e["group"]] = group_pts.get(e["group"], 0.0) + e["points"]
        groups_out = sorted(({"group": g, "points": p, "favors": home if p >= 0 else away}
                             for g, p in group_pts.items()), key=lambda x: -abs(x["points"]))
        drivers = [{"label": e["label"], "group": e["group"], "favors": e["favors"],
                    "points": abs(e["points"])} for e in feats_sorted[:6]]
        top3 = ", ".join(f"{d['label']} ({d['favors']} +{d['points']:.1f})" for d in drivers[:3])
        summary = (f"Model favors {fav} by {abs(margin):.1f} points ({max(ph, 1 - ph):.0%} win chance). "
                   f"Biggest factors: {top3}. "
                   f"The baseline of {intercept:+.1f} covers typical home-field advantage.")

        total_block = None
        if tot:
            pt = float(pred_t[i])
            tf = [{"name": name, "label": feature_meta(name)[0], "group": feature_meta(name)[1],
                   "value": Xt[i, j], "points": contrib_t[i, j],
                   "effect": "raises total" if contrib_t[i, j] >= 0 else "lowers total"}
                  for j, name in enumerate(tot["cols"])]
            tf_sorted = sorted(tf, key=lambda e: -abs(e["points"]))
            gp = {}
            for e in tf:
                gp[e["group"]] = gp.get(e["group"], 0.0) + e["points"]
            t_groups = sorted(({"group": g, "points": p, "effect": "raises total" if p >= 0 else "lowers total"}
                               for g, p in gp.items()), key=lambda x: -abs(x["points"]))
            t_drivers = [{"label": e["label"], "group": e["group"], "effect": e["effect"],
                          "points": abs(e["points"])} for e in tf_sorted[:6]]
            top3t = ", ".join(f"{d['label']} ({d['effect']} {d['points']:.1f})" for d in t_drivers[:3])
            hp, ap = (pt + margin) / 2, (pt - margin) / 2
            total_block = {
                "predicted_total": pt, "projected_score": {"home": hp, "away": ap},
                "range_50": [pt + unc_t["q25"], pt + unc_t["q75"]],
                "range_80": [pt + unc_t["q10"], pt + unc_t["q90"]],
                "typical_error_points": unc_t["sd"], "uncertainty_basis": unc_t["basis"],
                "baseline_points": intercept_t, "groups": t_groups, "top_drivers": t_drivers,
                "features": tf_sorted,
                "summary": (f"Model projects {pt:.1f} total points ({home} {hp:.0f}, {away} {ap:.0f}). "
                            f"Biggest factors: {top3t}. "
                            f"The baseline of {intercept_t:.1f} is the typical NFL total in the training data.")}

        w = weather.get(r["game_id"])
        indoors = bool(r["dome"] == 1)
        wx = {"stadium": r.get("stadium"), "roof": r.get("roof"), "surface": r.get("surface"),
              "indoors": indoors, "source": "indoors" if indoors else (w or {}).get("source", "unavailable"),
              "temp_f": None if indoors or not w else w.get("temp"),
              "wind_mph": None if indoors or not w else w.get("wind"),
              "precip_chance_pct": None if indoors or not w else w.get("precip_pct"),
              "summary": None if indoors or not w else w.get("summary"),
              "forecast_updated": None if not w else w.get("forecast_updated"),
              "used_by_total_model": {"wind_mph": r["wind_mph"], "cold": r["cold"]},
              "used_by_spread_model": False,
              "note": "Weather is shown for context and feeds only the over/under model, not the spread model."}

        qb_inj = [x for x in _injury_entries(inj, home) + _injury_entries(inj, away)
                  if x["position"] == "QB" and x["status"] in INJ_STATUSES]
        inj_status = "unavailable" if inj is None else ("none_listed" if inj.empty else "listed")
        notes = ["Starting quarterbacks are assumed to be each team's last starter."]
        if wx["source"] == "unavailable" and not indoors:
            notes.append("No weather forecast was available; the model treated the game as neutral weather.")
        if inj_status == "none_listed":
            notes.append("No injuries listed yet; reports usually post midweek.")
        if inj_status == "unavailable":
            notes.append(f"Injury data unavailable: {inj_note}")

        whatif = qb_what_if(r, X_cols, reg, clf, snap, qb_dir, weeks_played, margin, ph)

        sl = r.get("spread_line")
        home_fav = None if pd.isna(sl) or sl == 0 else bool(sl > 0)
        away_fav = None if home_fav is None else (not home_fav)
        sea, kd = int(r["season"]), r["gameday"]
        form_h, form_a = team_form(tg, home, sea, kd, True), team_form(tg, away, sea, kd, False)
        trends, bh, ba = None, None, None
        if INCLUDE_MARKET_LINES:
            bh = team_betting(tg, home, sea, kd, True, home_fav)
            ba = team_betting(tg, away, sea, kd, False, away_fav)
            angles = sorted(trend_angles(home, bh, True) + trend_angles(away, ba, False),
                            key=lambda x: -x["games"])[:8]
            trends = {"home": bh, "away": ba, "angles": angles,
                      "note": ("Historical records over small samples. They describe what has happened, "
                               "not what will happen.")}

        sections = build_sections(r, shared, wx, bh if INCLUDE_MARKET_LINES else None, ba if INCLUDE_MARKET_LINES else None,
                                  {"home": _injury_entries(inj, home), "away": _injury_entries(inj, away)},
                                  INCLUDE_MARKET_LINES)
        card = {
            "card_version": 2, "model_version": MODEL_VERSION, "generated_at": now,
            "disclaimer": "For information and entertainment only. Not betting advice.",
            "game": {"game_id": r["game_id"], "season": r["season"], "week": r["week"],
                     "kickoff_et": f"{pd.Timestamp(r['gameday']):%Y-%m-%d} {hhmm} ET",
                     "kickoff_utc": kick.isoformat(), "away": away, "home": home,
                     "stadium": r.get("stadium"), "neutral_site": bool(r["neutral"]),
                     "away_name": TEAM_NAMES.get(away), "home_name": TEAM_NAMES.get(home),
                     "city": None if r["neutral"] else TEAM_CITIES.get(home),
                     "divisional": bool(r["div_game"])},
            "model": {"predicted_margin_home": margin, "favored_team": fav,
                      "predicted_winning_margin": abs(margin),
                      "win_prob_home": ph, "win_prob_away": 1 - ph,
                      "margin_range_50": [margin + unc["q25"], margin + unc["q75"]],
                      "margin_range_80": [margin + unc["q10"], margin + unc["q90"]],
                      "typical_error_points": unc["sd"], "uncertainty_basis": unc["basis"],
                      "summary": summary, "baseline_points": intercept,
                      "groups": groups_out, "top_drivers": drivers, "features": feats_sorted},
            "total_model": total_block,
            "quarterbacks": {
                "assumed": True,
                "home": {"name": r["h_qb_name"], "rating": r["h_qb_rating"], "changed_from_last_game": bool(r["h_qb_chg"])},
                "away": {"name": r["a_qb_name"], "rating": r["a_qb_rating"], "changed_from_last_game": bool(r["a_qb_chg"])}},
            "team_form": {"home": form_h, "away": form_a},
            "qb_what_if": whatif,
            "team_comparison": {"home": profile(home), "away": profile(away)},
            "matchup_edges": matchup_edges(tables, home, away) if tables else [],
            "injuries": {"status": inj_status, "source": "nflverse injury reports",
                         "home": _injury_entries(inj, home), "away": _injury_entries(inj, away),
                         "qb_listed": bool(qb_inj)},
            "weather": wx,
            "schedule_spots": {
                "home": {"rest_days": r["home_rest"], "off_bye": bool(r["home_bye"]), "short_week": bool(r["home_short"])},
                "away": {"rest_days": r["away_rest"], "off_bye": bool(r["away_bye"]), "short_week": bool(r["away_short"])},
                "away_travel_miles": r["travel_k"] * 1000, "time_zones_crossed_east": r["tz_shift"],
                "west_coast_early_body_clock": bool(r["west_early"])},
            "history": {"home_recent": recent_games(hist, home, r["gameday"], epa_by_game, market=INCLUDE_MARKET_LINES),
                        "away_recent": recent_games(hist, away, r["gameday"], epa_by_game, market=INCLUDE_MARKET_LINES),
                        "head_to_head": head_to_head(hist, home, away, r["gameday"], market=INCLUDE_MARKET_LINES)},
            "sections": sections,
            "model_blind_spots": BLIND_SPOTS,
            "notes": notes,
        }
        if INCLUDE_MARKET_LINES:
            ph_m, pa_m = implied(r["home_moneyline"]), implied(r["away_moneyline"])
            card["market"] = {"spread_home": r["spread_line"], "moneyline_home": r["home_moneyline"],
                              "moneyline_away": r["away_moneyline"], "win_prob_home": ph_m / (ph_m + pa_m),
                              "model_margin_minus_spread": margin - r["spread_line"],
                              "favorite": home if home_fav else away if home_fav is False else None,
                              "favorite_spread": abs(r["spread_line"]) if pd.notna(r["spread_line"]) else None}
            if total_block:
                card["market"].update({"total_line": r["total_line"],
                                       "model_total_minus_line": total_block["predicted_total"] - r["total_line"],
                                       "total_lean": "over" if total_block["predicted_total"] > r["total_line"] else "under"})
        if trends:
            card["betting_trends"] = trends
        else:
            card["notes"].append("Against-the-spread and over/under records are off. "
                                 "Set INCLUDE_MARKET_LINES to True to include them.")
        card["chips"] = build_chips(r, sections, bh, ba, wx, [])
        card["_badges"] = _badges(r, w, qb_inj) + (["playoff_push"] if sections["situation"]["game"]["playoff_implications"] else [])
        cards.append(card)
    return cards


# ---------- App extras: angle chips, results, league trends, weather watch ----------
def _lopsided(w, l, n=3, t=0.75):
    d = w + l
    return d >= n and (w / d >= t or l / d >= t)


def build_chips(r, sections, bh, ba, wx, badges):
    """Short, neutral fact tags for a game ('Wind 18 mph', 'KC 4-1 ATS on the road'). They describe, never recommend."""
    home, away = r["home_team"], r["away_team"]
    sit, ppl = sections["situation"], sections["people"]
    out = []

    def add(kind, text, team=None, w=5):
        out.append({"kind": kind, "text": text, "team": team, "weight": w})

    for side, team in (("away", away), ("home", home)):
        st = ppl[side]["starter"]
        if st["changed_from_last_game"] and st.get("name"):
            add("qb", f"{team} QB change: {st['name']}", team, 1)
    if wx.get("indoors"):
        add("weather", "Indoors", None, 9)
    else:
        if (wx.get("wind_mph") or 0) >= WINDY_MPH:
            add("weather", f"Wind {wx['wind_mph']:.0f} mph", None, 2)
        if wx.get("temp_f") is not None and wx["temp_f"] <= 32:
            add("weather", f"{wx['temp_f']:.0f}°F", None, 2)
        text = str(wx.get("summary") or "").lower()
        if (wx.get("precip_chance_pct") or 0) >= 50 or any(k in text for k in ("rain", "snow", "storm")):
            add("weather", "Rain or snow possible", None, 2)
    for side, team in (("away", away), ("home", home)):
        s = sit[side]
        if s["short_week"]:
            add("situation", f"{team} short week", team, 3)
        if s["off_bye"]:
            add("situation", f"{team} off a bye", team, 3)
    if sit["time_zones_crossed_east"] >= 2:
        add("situation", f"{away} crosses {int(sit['time_zones_crossed_east'])} time zones", away, 4)
    elif sit["away_travel_miles"] >= 1500:
        add("situation", f"{away} travels {sit['away_travel_miles']:,.0f} mi", away, 5)
    if sit["game"]["divisional"]:
        add("situation", "Divisional", None, 4)
    if sit["game"]["rematch"]:
        add("situation", "Rematch this season", None, 4)
    if sit["game"]["playoff_implications"]:
        add("situation", "Playoff push", None, 3)
    for team, b, is_home in ((away, ba, False), (home, bh, True)):
        if not b:
            continue
        where = "at home" if is_home else "on the road"
        v = b["ats"]["venue"]
        if _lopsided(v["covered"], v["failed"]):
            add("ats", f"{team} {v['covered']}-{v['failed']} ATS {where}", team, 6)
        l5 = b["ats"]["last5"]
        if l5["games"] >= 5 and _lopsided(l5["covered"], l5["failed"], n=5, t=0.8):
            add("ats", f"{team} {l5['covered']}-{l5['failed']} ATS last 5", team, 6)
        st = b["ats"]["streak"]
        if st and st["length"] >= 3:
            add("ats", f"{team} {'covered' if st['type'] == 'W' else 'failed to cover'} {st['length']} straight", team, 6)
        ov = b["over_under"]["venue"]
        if _lopsided(ov["overs"], ov["unders"]):
            side = "Over" if ov["overs"] > ov["unders"] else "Under"
            a, c = (ov["overs"], ov["unders"]) if side == "Over" else (ov["unders"], ov["overs"])
            add("ou", f"{side} {a}-{c} in {team} {'home' if is_home else 'road'} games", team, 7)
        ost = b["over_under"]["streak"]
        if ost and ost["length"] >= 3:
            add("ou", f"{'Over' if ost['type'] == 'O' else 'Under'} hit {ost['length']} straight ({team})", team, 7)
    out.sort(key=lambda x: x["weight"])
    return out[:10]


def build_results(hist, season, market):
    """Final scores for every game played this season, so the app can grade a user's own picks."""
    h = hist[(hist.season == season) & hist.result.notna()].sort_values("gameday")
    games = []
    for r in h.itertuples():
        g = {"game_id": r.game_id, "week": int(r.week), "date": f"{r.gameday:%Y-%m-%d}",
             "kickoff_utc": _kickoff_utc(r.gameday, r.gametime).isoformat() if hasattr(r, "gametime") else None,
             "away": r.away_team,
             "home": r.home_team, "away_score": r.away_score, "home_score": r.home_score}
        if market:
            g.update({"spread_home": r.spread_line, "total_line": r.total_line})
        games.append(g)
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
            "season": int(season), "games": games}


def build_trends(tg, cards, season, before, market):
    """League and team trend tables for the Trends tab. Betting parts need market lines."""
    s = tg[(tg.season == season) & (tg.gameday < before)]
    teams = []
    for team, d in s.groupby("team"):
        d = d.sort_values("gameday")
        w, l, t = (d.win == "W").sum(), (d.win == "L").sum(), (d.win == "T").sum()
        e = {"team": team, "name": TEAM_NAMES.get(team), "games": int(len(d)), "record": _fmt_rec(int(w), int(l), int(t)),
             "ppg_for": d.pf.mean(), "ppg_against": d.pa.mean(), "streak": _streak(d.win.tolist()[::-1])}
        if market:
            a, o = d[d.ats != ""], d[d.ou != ""]
            ab, ob = _ats_block(a), _ou_block(o)
            e.update({"ats": ab["record"], "ats_pct": ab["cover_pct"], "ats_streak": _streak(a.ats.tolist()[::-1]),
                      "avg_cover_margin": float(a.cover.mean()) if len(a) else None,
                      "ou": ob["record"], "over_pct": ob["over_pct"], "ou_streak": _streak(o.ou.tolist()[::-1])})
        teams.append(e)
    league = None
    if market and len(s):
        a, o = s[s.ats != ""], s[s.ou != ""]
        fav, home = a[a.fav == 1.0], a[a.is_home]
        league = {"games": int(len(s) // 2),
                  "favorite_cover_pct": float((fav.ats == "W").sum() / max(((fav.ats == "W") | (fav.ats == "L")).sum(), 1)),
                  "home_cover_pct": float((home.ats == "W").sum() / max(((home.ats == "W") | (home.ats == "L")).sum(), 1)),
                  "over_pct": float((o.ou == "O").sum() / max(((o.ou == "O") | (o.ou == "U")).sum(), 1)),
                  "avg_total_points": float(o.total_pts.mean()) if len(o) else None,
                  "avg_posted_total": float(o.total_line.mean()) if len(o) else None}
    week_chips = [{"game_id": c["game"]["game_id"], "away": c["game"]["away"], "home": c["game"]["home"],
                   "kickoff_utc": c["game"]["kickoff_utc"], "chips": c.get("chips", [])}
                  for c in sorted(cards, key=lambda c: c["game"]["kickoff_utc"])]
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
            "season": int(season), "week": int(cards[0]["game"]["week"]), "market_lines": bool(market),
            "disclaimer": "Records describe what has already happened over small samples. They do not predict results.",
            "teams": teams, "league": league, "week_chips": week_chips}


def weather_alerts(wx):
    """(level, label, detail) for each weather concern. Level 2 = alert, 1 = watch."""
    if wx.get("indoors") or wx.get("temp_f") is None:
        return []
    out = []
    wind, temp, p = wx.get("wind_mph") or 0, wx["temp_f"], wx.get("precip_chance_pct") or 0
    text = str(wx.get("summary") or "").lower()
    if wind >= 20:
        out.append((2, "Strong wind", f"{wind:.0f} mph"))
    elif wind >= WINDY_MPH:
        out.append((1, "Windy", f"{wind:.0f} mph"))
    if temp <= 25:
        out.append((2, "Very cold", f"{temp:.0f}°F"))
    elif temp <= 32:
        out.append((1, "Freezing", f"{temp:.0f}°F"))
    elif temp >= 90:
        out.append((1, "Hot", f"{temp:.0f}°F"))
    if "snow" in text or "storm" in text or "thunder" in text:
        out.append((2, "Snow or storms", f"{p:.0f}% chance" if p else (wx.get("summary") or "")))
    elif p >= 50 or "rain" in text or "shower" in text:
        out.append((1, "Rain likely", f"{p:.0f}% chance" if p else (wx.get("summary") or "")))
    return out


def build_weather_watch(cards):
    """Games with notable weather, plus how the forecast moved since the last run."""
    path = os.path.join(CARD_DIR, "weather_prev.json")
    try:
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        prev = {}
    watch, calm, indoors, now_snap = [], [], [], {}
    for c in sorted(cards, key=lambda c: c["game"]["kickoff_utc"]):
        g, wx = c["game"], c["weather"]
        gid = g["game_id"]
        base = {"game_id": gid, "away": g["away"], "home": g["home"], "kickoff_utc": g["kickoff_utc"],
                "stadium": g.get("stadium"), "city": g.get("city"), "card": f"{gid}.json"}
        if wx.get("indoors"):
            indoors.append(base)
            continue
        snap = {"temp_f": wx.get("temp_f"), "wind_mph": wx.get("wind_mph"), "precip_chance_pct": wx.get("precip_chance_pct")}
        now_snap[gid] = snap
        alerts = weather_alerts(wx)
        e = {**base, **snap, "summary": wx.get("summary"), "forecast_updated": wx.get("forecast_updated"),
             "alerts": [{"level": a, "label": b, "detail": d} for a, b, d in alerts],
             "level": max([a for a, _, _ in alerts], default=0), "change": None}
        p = prev.get(gid)
        if p and snap["temp_f"] is not None and p.get("temp_f") is not None:
            dt = snap["temp_f"] - p["temp_f"]
            dw = (snap["wind_mph"] or 0) - (p.get("wind_mph") or 0)
            if abs(dt) >= 5 or abs(dw) >= 5:
                bits = []
                if abs(dt) >= 5:
                    bits.append(f"temperature {'up' if dt > 0 else 'down'} {abs(dt):.0f}°F")
                if abs(dw) >= 5:
                    bits.append(f"wind {'up' if dw > 0 else 'down'} {abs(dw):.0f} mph")
                e["change"] = "Since the last update: " + ", ".join(bits)
        (watch if e["level"] else calm).append(e)
    watch.sort(key=lambda x: (-x["level"], x["kickoff_utc"]))
    try:
        os.makedirs(CARD_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jsonable(now_snap), f)
    except Exception:
        pass
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
            "season": int(cards[0]["game"]["season"]), "week": int(cards[0]["game"]["week"]),
            "source": "National Weather Service hourly forecast near kickoff",
            "note": "Strong wind and cold tend to hurt passing and kicking. Forecasts change; check again closer to kickoff.",
            "watch": watch, "calm": calm, "indoors": indoors}


def write_extras(results, trends, weather):
    for name, payload in (("results.json", results), ("trends.json", trends), ("weather_watch.json", weather)):
        with open(os.path.join(CARD_DIR, name), "w", encoding="utf-8") as f:
            json.dump(jsonable(payload), f, indent=2)


# ---------- Team pages and standings (for the app's Rankings tab) ----------
TEAM_DIVISION = {t: d for d, ts in DIVISIONS.items() for t in ts}


def build_standings(tg, season, before, standings=None):
    """Division standings from results so far: record, points, streak, playoff seed."""
    s = tg[(tg.season == season) & (tg.gameday < before)]
    seeds = standings or standings_table(tg, season, before)
    out = []
    for div, teams in DIVISIONS.items():
        rows = []
        for t in teams:
            d = s[s.team == t].sort_values("gameday")
            w, l, ti = (d.win == "W").sum(), (d.win == "L").sum(), (d.win == "T").sum()
            n = int(w + l + ti)
            rows.append({"team": t, "name": TEAM_NAMES.get(t), "record": _fmt_rec(int(w), int(l), int(ti)),
                         "pct": (w + 0.5 * ti) / n if n else 0.0, "pf": int(d.pf.sum()), "pa": int(d.pa.sum()),
                         "diff": int(d.margin.sum()), "streak": _streak(d.win.tolist()[::-1]),
                         "seed": (seeds.get(t) or {}).get("seed"), "games_back": (seeds.get(t) or {}).get("games_back")})
        rows.sort(key=lambda r: (-r["pct"], -r["diff"]))
        out.append({"division": div, "conference": div[:3], "teams": rows})
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"), "season": int(season),
            "note": "Ordered by win percentage, then point margin. Real NFL tiebreakers are more involved.",
            "divisions": out}


def build_team_pages(schedule, tg, rankings, standings, season, before, cards, market, odds=None):
    """One small JSON file per team: record, rankings, QB and the full season schedule."""
    card_by_game = {c["game"]["game_id"]: f"{c['game']['game_id']}.json" for c in cards}
    folder = os.path.join(CARD_DIR, "teams")
    os.makedirs(folder, exist_ok=True)
    sch = schedule[schedule.season == season].sort_values("gameday")
    rk = {e["team"]: e for e in rankings["teams"]}
    stand = {r["team"]: r for d in standings["divisions"] for r in d["teams"]}
    odds_by = {e["team"]: e for e in (odds or {}).get("teams", [])}
    pages = []
    for team in sorted(TEAM_NAMES):
        mine = sch[(sch.home_team == team) | (sch.away_team == team)]
        games, weeks = [], set()
        for r in mine.itertuples():
            home = r.home_team == team
            weeks.add(int(r.week))
            played = pd.notna(r.result)
            e = {"week": int(r.week), "game_id": r.game_id, "date": f"{r.gameday:%Y-%m-%d}",
                 "kickoff_et": f"{r.gameday:%Y-%m-%d} {r.gametime if pd.notna(r.gametime) else '13:00'} ET",
                 "kickoff_utc": _kickoff_utc(r.gameday, r.gametime).isoformat(),
                 "opponent": r.away_team if home else r.home_team, "home_away": "home" if home else "away",
                 "status": "final" if played else "upcoming", "card": card_by_game.get(r.game_id)}
            if played:
                pf, pa = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
                e.update({"points_for": pf, "points_against": pa, "result": "W" if pf > pa else "L" if pf < pa else "T"})
                if market:
                    ts = -r.spread_line if home else r.spread_line
                    e.update({"team_spread": ts, "ats": _ats(pf - pa, ts) or None,
                              "total_line": r.total_line, "ou": _ou(pf + pa, r.total_line) or None})
            games.append(e)
        t = tg[(tg.team == team) & (tg.season == season) & (tg.gameday < before)].sort_values("gameday")
        last5 = [{"opponent": r.opp, "home_away": "home" if r.is_home else "away", "points_for": r.pf,
                  "points_against": r.pa, "result": r.win} for r in t.tail(5).iloc[::-1].itertuples()]
        e = rk.get(team, {})
        st = stand.get(team, {})
        page = {"model_version": MODEL_VERSION, "team": team, "name": TEAM_NAMES.get(team), "city": TEAM_CITIES.get(team),
                "division": TEAM_DIVISION.get(team), "season": int(season),
                "record": st.get("record"), "streak": st.get("streak"), "seed": st.get("seed"),
                "ppg_for": t.pf.mean() if len(t) else None, "ppg_against": t.pa.mean() if len(t) else None,
                "power": {"rank": e.get("rank"), "points": e.get("power_points"), "rank_change": e.get("rank_change"),
                          "offense_rank": e.get("offense_rank"), "defense_rank": e.get("defense_rank")},
                "categories": e.get("categories", {}), "qb": e.get("qb"), "odds": odds_by.get(team),
                "next_game": ({k: e["next_game"].get(k) for k in ("opponent", "home_away", "date", "game_id")}
                              if e.get("next_game") else None),
                "last5": last5, "schedule": games, "bye_week": next((w for w in range(1, 19) if w not in weeks), None),
                "disclaimer": "For information and entertainment only."}
        with open(os.path.join(folder, f"{team}.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(page), f, indent=2)
        pages.append(team)
    return pages


def write_standings(standings):
    with open(os.path.join(CARD_DIR, "standings.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(standings), f, indent=2)


def write_cards(cards):
    season, week = cards[0]["game"]["season"], cards[0]["game"]["week"]
    folder = os.path.join(CARD_DIR, f"{int(season)}_week{int(week):02d}")
    os.makedirs(folder, exist_ok=True)
    index = []
    for c in sorted(cards, key=lambda c: c["game"]["kickoff_utc"]):
        badges = c.pop("_badges")
        gid = c["game"]["game_id"]
        with open(os.path.join(folder, f"{gid}.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(c), f, indent=2)
        m = c["model"]
        index.append({"game_id": gid, "kickoff_et": c["game"]["kickoff_et"], "kickoff_utc": c["game"]["kickoff_utc"],
                      "away": c["game"]["away"], "home": c["game"]["home"],
                      "predicted_margin_home": m["predicted_margin_home"], "favored_team": m["favored_team"],
                      "win_prob_favored": max(m["win_prob_home"], m["win_prob_away"]),
                      "margin_range_80": m["margin_range_80"],
                      "predicted_total": (c.get("total_model") or {}).get("predicted_total"),
                      "projected_score": (c.get("total_model") or {}).get("projected_score"),
                      "total_range_80": (c.get("total_model") or {}).get("range_80"),
                      "away_name": c["game"].get("away_name"), "home_name": c["game"].get("home_name"),
                      "stadium": c["game"].get("stadium"), "city": c["game"].get("city"),
                      "weather": {k: c["weather"].get(k) for k in
                                  ("indoors", "source", "temp_f", "wind_mph", "precip_chance_pct", "summary")},
                      "records": {"home": c["team_form"]["home"].get("record"),
                                  "away": c["team_form"]["away"].get("record")},
                      "market": ({k: c["market"].get(k) for k in ("spread_home", "favorite", "favorite_spread", "total_line")}
                                 if "market" in c else None),
                      "chips": [{k: ch[k] for k in ("kind", "text", "team")} for ch in c.get("chips", [])[:4]],
                      "badges": badges, "card": f"{gid}.json"})
    with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable({"card_version": 1, "model_version": MODEL_VERSION,
                            "generated_at": cards[0]["generated_at"], "season": season, "week": week,
                            "disclaimer": cards[0]["disclaimer"], "games": index}), f, indent=2)
    with open(os.path.join(CARD_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump({"folder": os.path.basename(folder), "season": int(season), "week": int(week)}, f, indent=2)
    return folder


# ---------- Uncertainty: how far results land from the predicted margin ----------
def residual_uncertainty(wf, actual="result", pred="pred_margin", default_sd=13.0):
    """Measured out of sample in the walk-forward backtest, so it reflects real misses."""
    r = (wf[actual] - wf[pred]).dropna()
    if len(r) < 100:   # too little data: fall back to a typical NFL spread of outcomes
        z10, z25 = 1.2816, 0.6745
        return {"n": int(len(r)), "sd": default_sd, "q10": -z10 * default_sd, "q25": -z25 * default_sd,
                "q75": z25 * default_sd, "q90": z10 * default_sd,
                "basis": "typical NFL result spread (backtest too small to measure)"}
    q = r.quantile([0.10, 0.25, 0.75, 0.90])
    return {"n": int(len(r)), "sd": float(r.std()), "q10": float(q[0.10]), "q25": float(q[0.25]),
            "q75": float(q[0.75]), "q90": float(q[0.90]),
            "basis": f"walk-forward backtest, {len(r)} games"}


# ---------- QB what-if: what changes if a different quarterback starts ----------
def build_qb_directory(qb_all, df):
    g = qb_all.merge(df[["game_id", "season", "gameday"]], on="game_id", how="left")
    tot = g.groupby("qb_id")[["n_db", "sum_epa"]].sum()
    rating = (tot.sum_epa + QB_SHRINK_K * QB_PRIOR) / (tot.n_db + QB_SHRINK_K)
    last = g.sort_values("gameday").drop_duplicates("qb_id", keep="last").set_index("qb_id")
    vets = rating[tot.n_db >= 300]
    replacement = float(np.percentile(vets, 10)) if len(vets) >= 10 else QB_PRIOR - 0.1
    return {"recent": g, "rating": rating, "name": last.qb_name, "last_team": last.team,
            "replacement": replacement}


def qb_alternatives(qb_dir, snap, team, season, weeks_played, n=3):
    """Other QBs who have thrown for this team recently, plus a replacement-level option."""
    if team not in snap.index:
        return []
    assumed = snap.at[team, "qb_id"]
    g = qb_dir["recent"]
    cut = season if weeks_played >= 3 else season - 1   # early season: also count last year
    cand = (g[(g.team == team) & (g.season >= cut) & (g.qb_id != assumed)]
            .groupby("qb_id").n_db.sum().sort_values(ascending=False))
    out = []
    for qid in cand.index:
        if qb_dir["last_team"].get(qid) != team:   # has since moved to another team
            continue
        rt = qb_dir["rating"].get(qid)
        if pd.notna(rt):
            out.append((qid, str(qb_dir["name"].get(qid, qid)), float(rt)))
        if len(out) >= n:
            break
    out.append((None, "Replacement-level QB", qb_dir["replacement"]))
    return out


def qb_what_if(r, X_cols, reg, clf, snap, qb_dir, weeks_played, margin, ph):
    """Swap one team's QB rating (and the QB-change flag) and re-predict. Team form stays
    as is, so this is an approximation of the quarterback effect only."""
    home, away = r["home_team"], r["away_team"]
    base_vec = r[X_cols].astype(float)
    scen = []
    for side, team in (("home", home), ("away", away)):
        cur = float(r["h_qb_rating"] if side == "home" else r["a_qb_rating"])
        last_id = snap.at[team, "last_qb_id"] if team in snap.index else None
        for qid, name, rt in qb_alternatives(qb_dir, snap, team, int(r["season"]), weeks_played):
            v = base_vec.copy()
            delta = rt - cur
            v["qb_rating_diff"] += delta if side == "home" else -delta
            v["h_qb_chg" if side == "home" else "a_qb_chg"] = float(qid is None or qid != last_id)
            if "wind_x_qb" in v.index:
                v["wind_x_qb"] = r["wind_mph"] / 10 * v["qb_rating_diff"]
            if "cold_x_qb" in v.index:
                v["cold_x_qb"] = r["cold"] * v["qb_rating_diff"]
            scen.append({"side": side, "team": team, "qb": name, "rating": rt, "delta": delta,
                         "replacement": qid is None, "vec": v})
    out = {"note": ("Approximate: swaps only the quarterback rating and the model's QB-change flag (a new starter "
                    "counts as a change). Team offense and defense form stay as they are. "
                    "A positive margin change favors the home team."),
           "base": {"predicted_margin_home": margin, "win_prob_home": ph}, "home": [], "away": []}
    if scen:
        Xs = pd.DataFrame([s["vec"] for s in scen])[X_cols]
        ms, ps = reg.predict(Xs), clf.predict_proba(Xs)[:, 1]
        for s, m, p in zip(scen, ms, ps):
            out[s["side"]].append({"team": s["team"], "qb": s["qb"], "rating": s["rating"],
                                   "rating_change": s["delta"], "is_replacement_level": s["replacement"],
                                   "predicted_margin_home": m, "margin_change": m - margin,
                                   "win_prob_home": p})
    return out


# ---------- Prediction log: save before kickoff, grade after the game ----------
LOG_COLS = ["game_id", "season", "week", "kickoff_utc", "away", "home", "favored_team",
            "pred_margin_home", "p_home", "range80_lo", "range80_hi",
            "spread_line", "home_moneyline", "away_moneyline", "p_home_market",
            "pred_total", "total_line", "total_lo80", "total_hi80",
            "home_qb", "away_qb", "model_version", "logged_at",
            "home_score", "away_score", "actual_margin", "actual_total", "graded_at"]


def _now():
    return pd.Timestamp.now(tz="UTC")


def _v(x):
    """Plain value for CSV: blank for missing, rounded floats."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def load_log():
    if not os.path.exists(LOG_PATH):
        return {}
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return {r["game_id"]: r for r in csv.DictReader(f)}


def save_log(log):
    rows = sorted(log.values(), key=lambda r: str(r.get("kickoff_utc", "")))
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LOG_COLS})


def log_predictions(log, upcoming, pred, p_home, unc, pt, unc_t):
    """Record (or refresh) each game's prediction up until kickoff. After kickoff a row is
    locked, so the track record only ever contains predictions made before the game started."""
    now = _now()
    stamp = now.isoformat(timespec="seconds")
    new = updated = locked = late = 0
    for i, (_, r) in enumerate(upcoming.iterrows()):
        gid = r["game_id"]
        kick = _kickoff_utc(r["gameday"], r.get("gametime"))
        if kick <= now:
            if gid in log:
                locked += 1
            else:
                late += 1   # never logged before kickoff, so it can't count toward the record
            continue
        margin, ph = float(pred[i]), float(p_home[i])
        mh, ma = r.get("home_moneyline"), r.get("away_moneyline")
        pm = ""
        if pd.notna(mh) and pd.notna(ma):
            a, b = float(implied(mh)), float(implied(ma))
            pm = a / (a + b)
        row = {"game_id": gid, "season": r["season"], "week": r["week"], "kickoff_utc": kick.isoformat(),
               "away": r["away_team"], "home": r["home_team"],
               "favored_team": r["home_team"] if margin >= 0 else r["away_team"],
               "pred_margin_home": margin, "p_home": ph,
               "range80_lo": margin + unc["q10"], "range80_hi": margin + unc["q90"],
               "spread_line": r.get("spread_line"), "home_moneyline": mh, "away_moneyline": ma,
               "p_home_market": pm, "pred_total": float(pt[i]), "total_line": r.get("total_line"),
               "total_lo80": float(pt[i]) + unc_t["q10"], "total_hi80": float(pt[i]) + unc_t["q90"],
               "home_qb": r.get("h_qb_name"), "away_qb": r.get("a_qb_name"),
               "model_version": MODEL_VERSION, "logged_at": stamp}
        row = {k: _v(v) for k, v in row.items()}
        if gid in log:
            log[gid].update(row)
            updated += 1
        else:
            log[gid] = row
            new += 1
    print(f"\nPrediction log: {new} new, {updated} refreshed, {locked} locked (kickoff passed)"
          + (f", {late} not logged (kickoff passed before first run)" if late else "")
          + f" -> {LOG_PATH}")


def grade_log(log, results):
    stamp = _now().isoformat(timespec="seconds")
    n = 0
    for gid, r in log.items():
        if r.get("actual_margin") not in ("", None) and r.get("actual_total") not in ("", None):
            continue
        res = results.get(gid)
        if res is None:
            continue
        if r.get("actual_margin") in ("", None):
            n += 1
            r["graded_at"] = stamp
        r["home_score"], r["away_score"] = _v(res["home_score"]), _v(res["away_score"])
        r["actual_margin"] = _v(res["result"])
        r["actual_total"] = _v(res["home_score"] + res["away_score"])
    return n


def _calibration(p, y):
    d = pd.DataFrame({"b": pd.cut(p, [0, .3, .4, .5, .6, .7, 1]), "p": p, "y": y})
    g = d.groupby("b", observed=True).agg(games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return [{"bucket": str(k), "games": int(v.games), "predicted": float(v.predicted), "actual": float(v.actual)}
            for k, v in g.iterrows()]


def _tiers(p_home, home_win, p_mkt=None):
    p_pick = np.maximum(p_home, 1 - p_home)
    right = (p_home > 0.5) == (home_win == 1)
    rows = []
    for lo in (0.5, 0.6, 0.7, 0.8):
        m = p_pick >= lo
        row = {"confidence_at_least": lo, "games": int(m.sum()),
               "model_right_pct": float(right[m].mean()) if m.any() else None}
        if p_mkt is not None:
            row["market_right_pct"] = float(((p_mkt > 0.5) == (home_win == 1))[m].mean()) if m.any() else None
        rows.append(row)
    return rows


def live_summary(log):
    """Track record from logged (pre-kickoff) predictions that have since been graded."""
    out = {"games_logged": len(log), "games_graded": 0, "games_pending": len(log)}
    if not log:
        return out
    d = pd.DataFrame(list(log.values()), columns=LOG_COLS)
    for c in ["pred_margin_home", "p_home", "range80_lo", "range80_hi", "spread_line",
              "p_home_market", "actual_margin", "season", "week",
              "pred_total", "total_line", "total_lo80", "total_hi80", "actual_total"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d[d.actual_margin.notna()].copy()
    out["games_graded"], out["games_pending"] = len(g), len(d) - len(g)
    if g.empty:
        return out
    g["err"] = (g.actual_margin - g.pred_margin_home).abs()
    dec = g[g.actual_margin != 0]   # ties don't count toward win/loss accuracy
    p, y = dec.p_home.to_numpy(), (dec.actual_margin > 0).to_numpy().astype(int)
    right = (p > 0.5) == (y == 1)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    rng_ok = g.range80_lo.notna() & g.range80_hi.notna()
    out.update({
        "straight_up_pct": float(right.mean()) if len(dec) else None,
        "avg_margin_error": float(g.err.mean()),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))) if len(dec) else None,
        "range80_coverage": float(((g.actual_margin >= g.range80_lo) & (g.actual_margin <= g.range80_hi))[rng_ok].mean())
                            if rng_ok.any() else None,
        "confidence_tiers": _tiers(p, y) if len(dec) else [],
        "calibration": _calibration(p, y) if len(dec) else [],
    })
    weekly = []
    for (s, w), gg in g.groupby(["season", "week"]):
        dd = gg[gg.actual_margin != 0]
        weekly.append({"season": int(s), "week": int(w), "games": len(gg),
                       "straight_up_correct": int(((dd.p_home > 0.5) == (dd.actual_margin > 0)).sum()),
                       "avg_margin_error": float(gg.err.mean())})
    out["by_week"] = weekly
    mk = g[g.spread_line.notna()]
    if len(mk):
        mm = mk[mk.p_home_market.notna() & (mk.actual_margin != 0)]
        out["market"] = {"games": len(mk), "model_mae": float(mk.err.mean()),
                         "line_mae": float((mk.actual_margin - mk.spread_line).abs().mean()),
                         "market_straight_up_pct": float(((mm.p_home_market > 0.5) == (mm.actual_margin > 0)).mean())
                                                   if len(mm) else None}
    tt = d[d.actual_total.notna() & d.pred_total.notna()].copy()
    if len(tt):
        tt["err"] = (tt.actual_total - tt.pred_total).abs()
        okr = tt.total_lo80.notna() & tt.total_hi80.notna()
        out["totals"] = {"games": len(tt), "avg_total_error": float(tt.err.mean()),
                         "bias": float((tt.pred_total - tt.actual_total).mean()),
                         "range80_coverage": float(((tt.actual_total >= tt.total_lo80)
                                                    & (tt.actual_total <= tt.total_hi80))[okr].mean()) if okr.any() else None}
        ml = tt[tt.total_line.notna()]
        if len(ml):
            out["totals"]["market"] = {"games": len(ml), "model_mae": float(ml.err.mean()),
                                       "line_mae": float((ml.actual_total - ml.total_line).abs().mean())}
    return out


def print_live_record(s):
    print("\nLive track record (predictions saved before kickoff, graded after the game):")
    print(f"  logged {s['games_logged']} | graded {s['games_graded']} | waiting on results {s['games_pending']}")
    if not s["games_graded"]:
        print("  nothing graded yet; results appear here once logged games finish")
        return
    print(f"  picked the winner {s['straight_up_pct']:.1%} | avg margin error {s['avg_margin_error']:.2f} pts"
          f" | log loss {s['log_loss']:.4f}")
    if s.get("range80_coverage") is not None:
        print(f"  results inside the 80% range: {s['range80_coverage']:.0%} (should be near 80%)")
    if "market" in s:
        m = s["market"]
        print(f"  vs market line ({m['games']} games): model MAE {m['model_mae']:.2f} | line MAE {m['line_mae']:.2f}")
    if "totals" in s:
        t = s["totals"]
        extra = f" | inside 80% range {t['range80_coverage']:.0%}" if t.get("range80_coverage") is not None else ""
        print(f"  totals ({t['games']} games): avg miss {t['avg_total_error']:.2f} pts | bias {t['bias']:+.2f}{extra}")
        if "market" in t:
            print(f"    vs posted total: model MAE {t['market']['model_mae']:.2f} | line MAE {t['market']['line_mae']:.2f}")


def finish_log(log, results):
    n = grade_log(log, results)
    save_log(log)
    if n:
        print(f"\nGraded {n} previously logged games.")
    live = live_summary(log)
    print_live_record(live)
    return live


def backtest_summary(wf):
    """Backtest numbers for the app's accuracy screen (labeled as backtest, not live)."""
    p, y = wf.p_home_model.to_numpy(), wf.home_win.to_numpy()
    out = {"games": int(len(wf)), "seasons": f"{wf.season.min()}-{wf.season.max()}",
           "avg_margin_error": float(mean_absolute_error(wf.result, wf.pred_margin)),
           "straight_up_pct": float(((p > 0.5) == (y == 1)).mean()),
           "log_loss": float(log_loss(y, p)),
           "confidence_tiers": _tiers(p, y, wf.p_mkt.to_numpy() if "p_mkt" in wf else None),
           "calibration": _calibration(p, y)}
    if "p_mkt" in wf:
        out["market"] = {"line_mae": float(mean_absolute_error(wf.result, wf.spread_line)),
                         "log_loss": float(log_loss(y, wf.p_mkt)),
                         "straight_up_pct": float(((wf.p_mkt.to_numpy() > 0.5) == (y == 1)).mean())}
    return out


def write_accuracy(live, bt, bt_t=None):
    live, bt, bt_t = dict(live), dict(bt), dict(bt_t or {})
    if not INCLUDE_MARKET_LINES:   # keep market comparisons out of app data unless you opt in
        live.pop("market", None)
        bt.pop("market", None)
        bt_t.pop("market", None)
        if "totals" in live:
            live["totals"] = {k: v for k, v in live["totals"].items() if k != "market"}
        for t in bt.get("confidence_tiers", []):
            t.pop("market_right_pct", None)
    os.makedirs(CARD_DIR, exist_ok=True)
    payload = {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
               "disclaimer": "For information and entertainment only. Not betting advice.",
               "live": live, "backtest": bt, "backtest_totals": bt_t,
               "notes": ["'live' only counts predictions saved before kickoff.",
                         "'backtest' re-predicts past weeks using only data available before each week."]}
    with open(os.path.join(CARD_DIR, "accuracy.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2)


# ---------- Power rankings ----------
def team_record(hist, season):
    rec = {}
    for r in hist[(hist.season == season) & hist.result.notna()].itertuples():
        for t, won in ((r.home_team, r.result > 0), (r.away_team, r.result < 0)):
            w, l, tt = rec.get(t, (0, 0, 0))
            if r.result == 0:
                tt += 1
            elif won:
                w += 1
            else:
                l += 1
            rec[t] = (w, l, tt)
    return {t: f"{w}-{l}" + (f"-{tt}" if tt else "") for t, (w, l, tt) in rec.items()}


def power_ratings(snap, reg, X_cols):
    """Points better than an average team, using the model's own team-strength terms
    (Elo, form, EPA, QB). Same coefficients the game predictions use, so the two always agree."""
    scaler, ridge = reg[0], reg[-1]
    pts = pd.Series(0.0, index=snap.index)
    for j, c in enumerate(X_cols):
        f = c[:-5] if c.endswith("_diff") else None
        if f and f in snap.columns:
            col = snap[f].astype(float)
            pts += ridge.coef_[j] * (col.fillna(col.mean()) - col.mean()) / scaler.scale_[j]
    return pts


def build_power_rankings(extras, reg, X_cols, upcoming, pred, p_home, hist, tables=None):
    snap, prev = extras["snapshot"], extras["snapshot_prev"]
    now = power_ratings(snap, reg, X_cols)
    rank = now.rank(ascending=False, method="first").astype(int)
    prev_pts = prev_rank = None
    if prev is not None and len(prev):
        prev_pts = power_ratings(prev, reg, X_cols)
        prev_rank = prev_pts.rank(ascending=False, method="first").astype(int)
    off_rank = snap["epa_off_ewm"].rank(ascending=False, method="min") if "epa_off_ewm" in snap else None
    def_rank = snap["epa_def_ewm"].rank(ascending=True, method="min") if "epa_def_ewm" in snap else None
    season, week = int(upcoming.season.iloc[0]), int(upcoming.week.iloc[0])
    record = team_record(hist, season)

    nxt = {}
    for i, (_, r) in enumerate(upcoming.iterrows()):
        m, p = float(pred[i]), float(p_home[i])
        kick = f"{pd.Timestamp(r['gameday']):%Y-%m-%d}"
        nxt[r["home_team"]] = {"opponent": r["away_team"], "home_away": "home", "date": kick,
                               "predicted_margin": m, "win_prob": p, "game_id": r["game_id"]}
        nxt[r["away_team"]] = {"opponent": r["home_team"], "home_away": "away", "date": kick,
                               "predicted_margin": -m, "win_prob": 1 - p, "game_id": r["game_id"]}

    teams = []
    for t in now.sort_values(ascending=False).index:
        e = {"rank": int(rank[t]), "team": t, "power_points": now[t], "record": record.get(t),
             "prev_rank": None, "rank_change": None, "points_change": None,
             "elo": snap.at[t, "elo"],
             "offense_rank": int(off_rank[t]) if off_rank is not None and pd.notna(off_rank[t]) else None,
             "defense_rank": int(def_rank[t]) if def_rank is not None and pd.notna(def_rank[t]) else None,
             "qb": {"name": snap.at[t, "qb_name"], "rating": snap.at[t, "qb_rating"]},
             "next_game": nxt.get(t),   # None = bye this week
             "categories": {k: {"rank": _i(tb.at[t, "rank"]), "value": tb.at[t, "value"],
                                "rank_change": _i(tb.at[t, "rank_change"])}
                            for k, tb in (tables or {}).items() if t in tb.index}}
        if prev_rank is not None and t in prev_rank.index:
            e["prev_rank"] = int(prev_rank[t])
            e["rank_change"] = int(prev_rank[t] - rank[t])   # positive = moved up
            e["points_change"] = now[t] - prev_pts[t]
        teams.append(e)
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
            "season": season, "week": week,
            "disclaimer": "For information and entertainment only. Not betting advice.",
            "method": ("power_points = how many points better than an average team, on a neutral field, "
                       "from the model's team-strength inputs (Elo, recent form, EPA, quarterback). "
                       "rank_change compares with the start of last week; positive means moved up."),
            "teams": teams}


def write_rankings(rankings, folder):
    for path in (os.path.join(CARD_DIR, "power_rankings.json"), os.path.join(folder, "power_rankings.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jsonable(rankings), f, indent=2)


# ---------- Category rankings: passing/rushing offense and defense, QB ----------
CATEGORIES = [   # key, label, snapshot stat, higher is better, unit
    ("offense", "Offense", "epa_off", True, "EPA per play"),
    ("defense", "Defense", "epa_def", False, "EPA allowed per play"),
    ("passing_offense", "Passing offense", "epa_off_pass", True, "EPA per pass play"),
    ("passing_defense", "Passing defense", "epa_def_pass", False, "EPA allowed per pass play"),
    ("rushing_offense", "Rushing offense", "epa_off_rush", True, "EPA per rush"),
    ("rushing_defense", "Rushing defense", "epa_def_rush", False, "EPA allowed per rush"),
    ("quarterback", "Quarterback", "qb_rating", True, "career EPA per dropback, shrunk toward average"),
]


def _i(x):
    return int(x) if pd.notna(x) else None


def _rank(series, higher_is_better):
    return series.rank(ascending=not higher_is_better, method="min")


def compute_categories(snap, prev):
    """One table per category, indexed by team. Ranked on the same recent-weighted form the model
    uses; 'last 8 games' is shown alongside, and movement compares with the start of last week."""
    tables = {}
    for key, label, stat, hi, unit in CATEGORIES:
        col = "qb_rating" if stat == "qb_rating" else f"{stat}_ewm"
        if col not in snap.columns:
            continue
        v = snap[col].astype(float)
        n = int(v.notna().sum())
        t = pd.DataFrame({"value": v, "rank": _rank(v, hi)})
        t["percentile"] = 100 * (n - t["rank"]) / max(n - 1, 1)   # 100 = best in the league
        t["vs_average"] = (v - v.mean()) if hi else (v.mean() - v)   # positive = better than average
        t["prev_rank"] = np.nan
        t["rank_change"] = np.nan
        if prev is not None and col in prev.columns:
            t["prev_rank"] = _rank(prev[col].astype(float), hi).reindex(t.index)
            t["rank_change"] = t["prev_rank"] - t["rank"]   # positive = moved up
        c8 = f"{stat}_r8"
        t["last8_value"] = snap[c8].astype(float) if c8 in snap.columns else np.nan
        t["last8_rank"] = _rank(t["last8_value"], hi) if t["last8_value"].notna().any() else np.nan
        tables[key] = t.sort_values("rank")
    return tables


def category_payload(tables, rankings):
    cats = []
    for key, label, stat, hi, unit in CATEGORIES:
        t = tables.get(key)
        if t is None:
            continue
        cats.append({"key": key, "label": label, "unit": unit, "higher_is_better": hi,
                     "league_average": float(t["value"].mean()),
                     "teams": [{"rank": _i(r["rank"]), "team": team, "value": r["value"],
                                "vs_average": r["vs_average"], "percentile": r["percentile"],
                                "prev_rank": _i(r["prev_rank"]), "rank_change": _i(r["rank_change"]),
                                "last_8_value": r["last8_value"], "last_8_rank": _i(r["last8_rank"])}
                               for team, r in t.iterrows() if pd.notna(r["rank"])]})
    return {"model_version": MODEL_VERSION, "generated_at": rankings["generated_at"],
            "season": rankings["season"], "week": rankings["week"],
            "disclaimer": "For information and entertainment only. Not betting advice.",
            "method": ("Teams are ranked on recent-weighted efficiency (EPA per play, with garbage time removed), "
                       "the same inputs the model uses. For defense, lower EPA allowed is better. "
                       "vs_average is positive when a team is better than the league average. "
                       "rank_change compares with the start of last week; positive means moved up. "
                       "last_8 is the same stat over each team's last 8 games."),
            "categories": cats}


def write_category_rankings(payload, folder):
    for path in (os.path.join(CARD_DIR, "category_rankings.json"),
                 os.path.join(folder, "category_rankings.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jsonable(payload), f, indent=2)


EDGE_PAIRS = [   # (offense side, offense category, defense category, label)
    ("home", "passing_offense", "passing_defense", "passing"),
    ("away", "passing_offense", "passing_defense", "passing"),
    ("home", "rushing_offense", "rushing_defense", "rushing"),
    ("away", "rushing_offense", "rushing_defense", "rushing"),
]


def matchup_edges(tables, home, away, gap_needed=8):
    """Each offense against the opposing defense in the same phase. A gap of 8+ ranks is flagged."""
    out = []
    for side, off_key, def_key, kind in EDGE_PAIRS:
        o, d = (home, away) if side == "home" else (away, home)
        to, td = tables.get(off_key), tables.get(def_key)
        if to is None or td is None or o not in to.index or d not in td.index:
            continue
        ro, rd = to.at[o, "rank"], td.at[d, "rank"]
        if pd.isna(ro) or pd.isna(rd):
            continue
        gap = int(rd - ro)   # positive = the offense ranks higher than the defense it faces
        adv = "offense" if gap >= gap_needed else "defense" if gap <= -gap_needed else "even"
        out.append({"matchup": f"{o} {kind} offense vs {d} {kind} defense", "kind": kind,
                    "offense_team": o, "defense_team": d, "offense_rank": int(ro), "defense_rank": int(rd),
                    "rank_gap": gap, "advantage": adv,
                    "advantage_team": o if adv == "offense" else d if adv == "defense" else None})
    return out


def print_category_leaders(tables, n=3):
    print("\nCategory leaders (recent-weighted EPA; best first | worst last):")
    for key, label, stat, hi, unit in CATEGORIES:
        t = tables.get(key)
        if t is None:
            continue
        t = t[t["rank"].notna()]
        best = ", ".join(f"{tm} {r['value']:+.2f}" for tm, r in t.head(n).iterrows())
        worst = ", ".join(f"{tm} {r['value']:+.2f}" for tm, r in t.tail(n).iloc[::-1].iterrows())
        print(f"  {label:<16} {best}  |  {worst}")


# ---------- v14: playoff odds, game of the week, post-game recaps ----------
SIM_COUNT = 10000        # simulated seasons for the playoff odds
SIM_HFA = 2.0            # home-field advantage, in points
SIM_GAME_SD = 13.0       # typical miss on one game's margin, in points
SIM_RATING_SD = 2.0      # how unsure we are of a team's true strength (shared across its remaining games)
ALL_TEAMS = [t for ts in DIVISIONS.values() for t in ts]   # 32 teams; AFC divisions first, 4 teams per division


def _write_json(name, payload, folder=None):
    os.makedirs(CARD_DIR, exist_ok=True)
    paths = [os.path.join(CARD_DIR, name)] + ([os.path.join(folder, name)] if folder else [])
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jsonable(payload), f, indent=2)


def simulate_season(schedule, tg, power, season, n=SIM_COUNT, seed=14):
    """Plays out the rest of the regular season n times and counts who makes the playoffs.
    Each game is a coin weighted by the power ratings (plus home field). Ties in the standings are
    broken by point margin, which is simpler than the real NFL tiebreakers."""
    idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    wins, diff = np.zeros(32), np.zeros(32)
    losses, ties = np.zeros(32), np.zeros(32)
    s = tg[tg.season == season]
    for t, g in s.groupby("team"):
        if t in idx:
            i = idx[t]
            wins[i] = (g.win == "W").sum() + 0.5 * (g.win == "T").sum()
            losses[i], ties[i] = (g.win == "L").sum(), (g.win == "T").sum()
            diff[i] = g.margin.sum()
    sch = schedule[(schedule.season == season)].sort_values(["gameday", "week"])
    rem = sch[sch.result.isna() & sch.home_team.isin(idx) & sch.away_team.isin(idx)]
    G = len(rem)
    pp = np.array([float(power.get(t, 0.0)) for t in ALL_TEAMS])
    hi = rem.home_team.map(idx).to_numpy().astype(int)
    ai = rem.away_team.map(idx).to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    sd_game = float(np.sqrt(max(SIM_GAME_SD ** 2 - 2 * SIM_RATING_SD ** 2, 25.0)))
    R = rng.normal(0, SIM_RATING_SD, size=(n, 32))
    mu = (pp[hi][None, :] + R[:, hi]) - (pp[ai][None, :] + R[:, ai]) + SIM_HFA
    margin = mu + rng.normal(0, sd_game, size=(n, G))
    hw = margin > 0
    H, A = np.zeros((G, 32), dtype=np.float32), np.zeros((G, 32), dtype=np.float32)
    H[np.arange(G), hi] = 1
    A[np.arange(G), ai] = 1
    hwf = hw.astype(np.float32)
    W = wins[None, :] + hwf @ H + (1 - hwf) @ A
    D = diff[None, :] + margin.astype(np.float32) @ H - margin.astype(np.float32) @ A
    score = W + 5e-4 * D + 1e-6 * rng.random((n, 32))

    seed_arr = np.zeros((n, 32), dtype=np.int8)      # 0 = missed the playoffs
    div_win = np.zeros((n, 32), dtype=bool)
    rows = np.arange(n)
    for c in range(2):
        base = c * 16
        block = score[:, base:base + 16]
        winners = base + np.arange(4)[None, :] * 4 + block.reshape(n, 4, 4).argmax(axis=2)   # (n, 4)
        wscore = np.take_along_axis(score, winners, axis=1)
        ranked = np.take_along_axis(winners, np.argsort(-wscore, axis=1), axis=1)          # seeds 1-4
        rest = block.copy()
        np.put_along_axis(rest, winners - base, -1e9, axis=1)
        wild = np.argsort(-rest, axis=1)[:, :3] + base                                       # seeds 5-7
        order = np.concatenate([ranked, wild], axis=1)
        for k in range(7):
            seed_arr[rows, order[:, k]] = k + 1
        div_win[rows[:, None], winners] = True
    made = seed_arr > 0

    nxt = {}
    for g in range(G):
        for t in (int(hi[g]), int(ai[g])):
            nxt.setdefault(t, g)
    rem_ids = rem.game_id.tolist()
    record = lambda i: _fmt_rec(int(wins[i] - 0.5 * ties[i]), int(losses[i]), int(ties[i]))
    out_teams = []
    for t in ALL_TEAMS:
        i = idx[t]
        wd = np.bincount(np.clip(np.rint(W[:, i]).astype(int), 0, 17), minlength=18)[:18] / n
        e = {"team": t, "record": record(i), "projected_wins": float(W[:, i].mean()),
             "p_playoffs": float(made[:, i].mean()), "p_division": float(div_win[:, i].mean()),
             "p_bye": float((seed_arr[:, i] == 1).mean()),
             "p_seed": [float((seed_arr[:, i] == k).mean()) for k in range(1, 8)],
             "win_dist": [round(float(x), 4) for x in wd], "next_game": None}
        g = nxt.get(i)
        if g is not None:
            is_home = int(hi[g]) == i
            won = hw[:, g] if is_home else ~hw[:, g]
            def cond(arr, mask):
                return float(arr[mask, i].mean()) if mask.any() else None
            e["next_game"] = {"game_id": rem_ids[g], "opponent": ALL_TEAMS[int(ai[g] if is_home else hi[g])],
                              "home_away": "home" if is_home else "away",
                              "p_playoffs_if_win": cond(made, won), "p_playoffs_if_loss": cond(made, ~won),
                              "p_division_if_win": cond(div_win, won), "p_division_if_loss": cond(div_win, ~won)}
        out_teams.append(e)
    out_teams.sort(key=lambda e: -e["p_playoffs"])
    played = int(s.groupby("team").size().max()) if len(s) else 0
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
            "season": int(season), "games_played_per_team": played, "games_remaining": int(G), "simulations": int(n),
            "method": (f"The rest of the season was simulated {n:,} times using the power ratings, home field and "
                       "some uncertainty about each team's true strength. Seeds use the real 7-team format. "
                       "Standings ties are broken by point margin, which is simpler than the NFL's tiebreakers."),
            "disclaimer": "Estimates for fun and information. Not betting advice.",
            "teams": out_teams}


def build_highlights(upcoming, rankings, odds):
    """Which games matter most this week. Describes the game (quality, closeness, playoff stakes) and never names a pick."""
    rk = {e["team"]: e for e in rankings["teams"]}
    od = {e["team"]: e for e in odds["teams"]}
    games = []
    for r in upcoming.itertuples():
        h, a = r.home_team, r.away_team
        if h not in rk or a not in rk:
            continue
        gap = abs(rk[h]["power_points"] - rk[a]["power_points"] + SIM_HFA)
        quality = 1 - ((rk[h]["rank"] + rk[a]["rank"]) / 2 - 1) / 31
        closeness = 1 - min(gap / 10.0, 1.0)
        swings = []
        for t in (h, a):
            ng = (od.get(t) or {}).get("next_game") or {}
            if ng.get("game_id") == r.game_id and ng.get("p_playoffs_if_win") is not None and ng.get("p_playoffs_if_loss") is not None:
                swings.append(abs(ng["p_playoffs_if_win"] - ng["p_playoffs_if_loss"]))
        swing = float(np.mean(swings)) if swings else 0.0
        stakes = min(swing / 0.25, 1.0)
        divisional = bool(getattr(r, "div_game", 0) == 1) or TEAM_DIVISION.get(h) == TEAM_DIVISION.get(a)
        score = 100 * (0.35 * quality + 0.25 * closeness + 0.30 * stakes + 0.10 * (1.0 if divisional else 0.0))
        reasons, tags = [], []
        if rk[h]["rank"] <= 8 and rk[a]["rank"] <= 8:
            reasons.append("Two top-8 teams in the power rankings"); tags.append("top_matchup")
        elif rk[h]["rank"] <= 16 and rk[a]["rank"] <= 16:
            reasons.append("Two top-half teams in the power rankings")
        if gap <= 3.0:
            reasons.append("The ratings have these teams nearly even"); tags.append("toss_up")
        if swing >= 0.08:
            reasons.append(f"Playoff odds swing about {round(swing * 100)} points on the result"); tags.append("playoff_swing")
        if divisional:
            reasons.append("Division rivals"); tags.append("division")
        kick = _kickoff_utc(r.gameday, getattr(r, "gametime", None)).isoformat()
        games.append({"game_id": r.game_id, "card": f"{r.game_id}.json", "away": a, "home": h, "kickoff_utc": kick,
                      "score": round(score, 1), "rating_gap": round(float(gap), 1), "playoff_swing": round(swing, 3),
                      "reasons": reasons, "tags": tags})
    games.sort(key=lambda g: -g["score"])
    toss = sorted([g for g in games if "toss_up" in g["tags"]], key=lambda g: g["rating_gap"])[:3]
    swing = sorted([g for g in games if g["playoff_swing"] >= 0.08], key=lambda g: -g["playoff_swing"])[:3]
    return {"model_version": MODEL_VERSION, "generated_at": _now().isoformat(timespec="seconds"),
            "season": int(upcoming.season.iloc[0]), "week": int(upcoming.week.iloc[0]),
            "note": "Ranks games by team quality, how even the ratings are, playoff stakes and rivalry. It does not pick winners.",
            "game_of_the_week": games[0] if games else None, "games": games,
            "toss_ups": [g["game_id"] for g in toss], "playoff_swing": [g["game_id"] for g in swing]}


RECAP_COLS = ["game_id", "season_type", "qtr", "time", "posteam", "defteam", "desc", "play_type", "epa", "wpa", "home_wp",
              "pass", "rush", "qb_dropback", "qb_kneel", "qb_spike", "yards_gained", "success", "sack", "interception",
              "fumble_lost", "third_down_converted", "third_down_failed", "penalty", "penalty_team", "drive",
              "fixed_drive_result", "drive_inside20", "yardline_100", "total_home_score", "total_away_score"]


def load_recap_pbp(season):
    """This season's play-by-play with just the columns the recaps need. None when it isn't available yet."""
    try:
        raw = nfl.load_pbp([season])
        df = raw.select([c for c in RECAP_COLS if c in raw.columns]).to_pandas()
        return df[df.season_type == "REG"] if "season_type" in df.columns else df
    except Exception as e:
        print(f"No play-by-play for recaps ({e}).")
        return None


def _clean_desc(text, limit=170):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


def _team_game_stats(p, ctx, team):
    o = p[(p.posteam == team) & (((_num(p, "pass") == 1) | (_num(p, "rush") == 1)) & (_num(p, "qb_kneel") == 0) & (_num(p, "qb_spike") == 0))]
    ps, rs = o[_num(o, "pass") == 1], o[_num(o, "rush") == 1]
    epa = lambda d: float(pd.to_numeric(d.epa, errors="coerce").mean()) if len(d) and "epa" in d else None
    c = ctx[ctx.team == team]
    c = c.iloc[0] if len(c) else None
    def cv(col):
        return float(c[col]) if c is not None else None
    return {"plays": int(len(o)), "yards": float(_num(o, "yards_gained").sum()),
            "epa_per_play": epa(o), "pass_epa": epa(ps), "rush_epa": epa(rs),
            "success_rate": float(_num(o, "success").mean()) if len(o) else None,
            "giveaways": cv("giveaways"), "sacks_taken": cv("sacks_taken"), "penalties": cv("penalties"),
            "third_conv": cv("third_conv"), "third_att": cv("third_att"), "rz_td": cv("rz_td"), "rz_trips": cv("rz_trips")}


def recap_for_game(p, home, away, hs, as_, meta):
    """One game's recap from its play-by-play: score by quarter, team stats, biggest swings, win-probability trace."""
    p = p.reset_index(drop=True)
    ctx = context_from_pbp(p.assign(season_type="REG"))
    stats = {home: _team_game_stats(p, ctx, home), away: _team_game_stats(p, ctx, away)}
    # score by quarter
    quarters, prev_h, prev_a = [], 0, 0
    if "qtr" in p.columns and "total_home_score" in p.columns:
        last = p.dropna(subset=["qtr"]).groupby("qtr").last()
        for q, row in last.iterrows():
            th, ta = int(row.get("total_home_score") or 0), int(row.get("total_away_score") or 0)
            quarters.append({"qtr": int(q), "home": th - prev_h, "away": ta - prev_a})
            prev_h, prev_a = th, ta
    # win probability trace (home team), about 48 points
    trace = []
    if "home_wp" in p.columns:
        w = pd.to_numeric(p.home_wp, errors="coerce").dropna().to_numpy()
        if len(w) >= 2:
            pick = np.linspace(0, len(w) - 1, 48).astype(int)
            trace = [round(float(w[i]), 3) for i in pick]
    # biggest swings
    swings = []
    if "wpa" in p.columns and "desc" in p.columns:
        q = p[pd.to_numeric(p.wpa, errors="coerce").notna() & p.posteam.notna()].copy()
        q["wpa"] = pd.to_numeric(q.wpa)
        q["home_wpa"] = np.where(q.posteam == home, q.wpa, -q.wpa)
        q = q[~q.get("play_type", pd.Series("", index=q.index)).isin(["kickoff", "no_play"])]
        for r in q.reindex(q.home_wpa.abs().sort_values(ascending=False).index).head(3).itertuples():
            swings.append({"qtr": int(r.qtr) if pd.notna(getattr(r, "qtr", np.nan)) else None, "clock": str(getattr(r, "time", "") or ""),
                           "team": r.posteam, "toward": home if r.home_wpa > 0 else away,
                           "swing_pts": round(abs(float(r.home_wpa)) * 100, 1), "text": _clean_desc(r.desc)})
    # biggest deficit overcome by the winner
    deficit = 0
    if "total_home_score" in p.columns and hs != as_:
        d = (pd.to_numeric(p.total_home_score, errors="coerce") - pd.to_numeric(p.total_away_score, errors="coerce")).dropna()
        if len(d):
            deficit = int(max(0, -d.min())) if hs > as_ else int(max(0, d.max()))
    winner, loser = (home, away) if hs > as_ else (away, home) if as_ > hs else (None, None)
    ws, ls = (hs, as_) if hs > as_ else (as_, hs)
    ot = any(q["qtr"] >= 5 for q in quarters)
    headline = (f"{winner} {int(ws)}, {loser} {int(ls)}" + (" (OT)" if ot else "")) if winner else f"{away} {int(as_)}, {home} {int(hs)} (tie)"
    story = []
    ea, eh = stats[away]["epa_per_play"], stats[home]["epa_per_play"]
    if ea is not None and eh is not None and winner:
        better = home if eh > ea else away
        if abs(eh - ea) >= 0.05:
            if better == winner:
                story.append(f"{winner} won the efficiency battle ({stats[winner]['epa_per_play']:+.2f} EPA per play to {stats[loser]['epa_per_play']:+.2f}).")
            else:
                story.append(f"{loser} was more efficient ({stats[loser]['epa_per_play']:+.2f} EPA per play to {stats[winner]['epa_per_play']:+.2f}) but still lost.")
    ga, gh = stats[away]["giveaways"], stats[home]["giveaways"]
    if ga is not None and gh is not None and abs(ga - gh) >= 2:
        fewer, more = (home, away) if gh < ga else (away, home)
        story.append(f"{fewer} won the turnover battle, {int(stats[more]['giveaways'])} giveaways to {int(stats[fewer]['giveaways'])}.")
    if winner and deficit >= 10:
        story.append(f"{winner} came back from {deficit} points down.")
    if winner and (hs - as_ if hs > as_ else as_ - hs) <= 3:
        story.append("Decided by a field goal or less.")
    if winner and (ws - ls) >= 21:
        story.append(f"A {int(ws - ls)}-point rout.")
    sk = [(stats[t]["sacks_taken"], t) for t in (home, away) if stats[t]["sacks_taken"] is not None]
    if sk and max(sk)[0] >= 4:
        story.append(f"{max(sk)[1]}'s offense was sacked {int(max(sk)[0])} times.")
    return {"model_version": MODEL_VERSION, **meta, "home": home, "away": away, "home_score": hs, "away_score": as_,
            "headline": headline, "overtime": ot, "story": story[:4], "quarters": quarters, "stats": stats,
            "biggest_plays": swings, "home_win_prob": trace, "max_deficit_overcome": deficit,
            "note": "Computed from play-by-play. EPA is expected points added; positive is good for the offense."}


def build_recaps(hist, season, pbp=None):
    """A recap file for every finished game this season, plus an index. Returns the number written."""
    done = hist[(hist.season == season) & hist.result.notna()].sort_values("gameday")
    if pbp is None:
        pbp = load_recap_pbp(season)
    if pbp is None or done.empty:
        return 0
    folder = os.path.join(CARD_DIR, "recaps")
    os.makedirs(folder, exist_ok=True)
    index = []
    for r in done.itertuples():
        p = pbp[pbp.game_id == r.game_id]
        if p.empty:
            continue
        meta = {"game_id": r.game_id, "season": int(season), "week": int(r.week), "date": f"{r.gameday:%Y-%m-%d}"}
        try:
            rec = recap_for_game(p, r.home_team, r.away_team, float(r.home_score), float(r.away_score), meta)
        except Exception as e:   # one odd game should never stop the rest
            print(f"Recap skipped for {r.game_id}: {e}")
            continue
        with open(os.path.join(folder, f"{r.game_id}.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(rec), f)
        index.append({"game_id": r.game_id, "week": int(r.week), "date": meta["date"], "away": r.away_team, "home": r.home_team,
                      "away_score": float(r.away_score), "home_score": float(r.home_score), "headline": rec["headline"],
                      "overtime": rec["overtime"], "top_story": (rec["story"] or [None])[0]})
    index.sort(key=lambda g: (g["date"], g["game_id"]), reverse=True)
    with open(os.path.join(folder, "index.json"), "w", encoding="utf-8") as f:
        json.dump(jsonable({"model_version": MODEL_VERSION, "season": int(season), "games": index}), f)
    return len(index)


# ---------- "Where did he go to college?" game (cards/college/*.json) ----------
# Built from the nflverse players table. It does not touch the models; it only gives the app a pool of players.
# Players who attended more than one school are left out so every question has exactly one right answer.
COLLEGE_SUBDIR = "college"
COLLEGE_SINCE = 2014          # players whose last season is this year or later
COLLEGE_HARD_MAX = 3000       # keeps the hard file small
_CONF_SHORT = {
    "Southeastern Conference": "SEC", "Big Ten Conference": "Big Ten", "Atlantic Coast Conference": "ACC",
    "Pacific Twelve Conference": "Pac-12", "Pacific Ten Conference": "Pac-12", "Big Twelve Conference": "Big 12",
    "American Athletic Conference": "American", "Mountain West Conference": "Mountain West",
    "Conference USA": "Conference USA", "Mid-American Conference": "MAC", "Sun Belt Conference": "Sun Belt",
    "Independent": "Independent", "Big East": "Big East",
}
_FAME_POS = {"QB": 4, "RB": 3, "WR": 3, "TE": 2, "DL": 1, "LB": 1, "DB": 1, "OL": 0, "SPEC": -1}
_FAME_ROUND = {1: 5, 2: 4, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1}


def _clean_school(name):
    n = str(name).strip()
    fixed = {"Miami": "Miami (FL)", "Miami (Ohio)": "Miami (OH)"}
    if n in fixed:
        return fixed[n]
    n = re.sub(r",\s*University of$", "", n)
    n = re.sub(r"^Univ(ersity)?\.? of ", "", n)
    n = re.sub(r" University$", "", n)
    return n.strip()


def _num(v, as_int=True):
    """A plain number, or None when the cell is empty."""
    try:
        if v is None or pd.isna(v):
            return None
        return int(v) if as_int else float(v)
    except (TypeError, ValueError):
        return None


def build_college_game(players):
    """players: the nflverse players table (a pandas DataFrame). Returns {file name: payload}."""
    p = players.copy()
    p = p[p.last_season.notna() & p.college_name.notna() & p.position.notna() & p.display_name.notna()]
    p = p[p.last_season >= COLLEGE_SINCE]
    p = p[~p.college_name.astype(str).str.contains(";")]          # transfers: no single right answer
    cur = int(p.last_season.max())
    p["school"] = p.college_name.map(_clean_school)
    p = p[p.school != ""]
    p = p.drop_duplicates(["display_name", "position"], keep=False)  # two players sharing a name and position are ambiguous

    def fame(r):
        s = _FAME_POS.get(r.position_group, 0)
        if pd.notna(r.draft_round):
            s += _FAME_ROUND.get(int(r.draft_round), 0)
        if pd.notna(r.draft_pick) and r.draft_pick <= 10:
            s += 1
        if pd.notna(r.years_of_experience) and r.years_of_experience >= 4:
            s += 1
        return s

    p["fame"] = p.apply(fame, axis=1)
    recent = p.last_season >= cur - 1
    tier = np.where(recent & (p.fame >= 8), "easy", np.where((recent & (p.fame >= 4)) | (p.fame >= 8), "medium", "hard"))
    p["tier"] = tier

    def row(r):
        return {
            "n": r.display_name, "p": r.position, "t": r.latest_team if isinstance(r.latest_team, str) else None,
            "c": r.school, "y": _num(r.rookie_season), "l": _num(r.last_season),
            "dy": _num(r.draft_year), "dr": _num(r.draft_round), "dk": _num(r.draft_pick),
            "h": _num(r.height), "w": _num(r.weight),
        }

    out = {}
    for t in ("easy", "medium", "hard"):
        sub = p[p.tier == t].sort_values(["fame", "last_season"], ascending=False)
        if t == "hard":
            sub = sub.head(COLLEGE_HARD_MAX)
        out[t] = {"tier": t, "season": cur, "players": [row(r) for r in sub.itertuples(index=False)]}

    # Every school that appears, with its conference and how many players it has (the app uses this for wrong
    # answers and for the type-ahead on the hard level).
    used = p.groupby("school").agg(k=("display_name", "size"), cf=("college_conference", lambda s: s.dropna().iloc[0] if s.notna().any() else None))
    schools = [{"n": n, "k": int(r.k), "cf": _CONF_SHORT.get(r.cf, str(r.cf).replace(" Conference", "") if pd.notna(r.cf) else None)}
               for n, r in used.sort_values("k", ascending=False).iterrows()]
    out["schools"] = {"season": cur, "schools": schools}
    return out


def write_college_game(players=None):
    if players is None:
        players = nfl.load_players().to_pandas()
    files = build_college_game(players)
    folder = os.path.join(CARD_DIR, COLLEGE_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    for name, payload in files.items():
        with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    sizes = {k: (len(v["players"]) if "players" in v else len(v["schools"])) for k, v in files.items()}
    print(f"Wrote the college game to {folder}/: {sizes}")


# ---------- "Test your skills" game (cards/skills/*.json) ----------
# One file per season, each holding past regular-season games (week 4 on). Every number for a game uses ONLY games played
# before it, plus that week's injury report and the posted spread, so nothing from the future leaks into the question.
# It does not touch the models.
SKILLS_SUBDIR = "skills"
SKILLS_FIRST_SEASON = 2014
SKILLS_MIN_WEEK = 4
_TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}      # franchises that moved, so every team has one code
_POS_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "OT": "OL",
              "OG": "OL", "OL": "OL", "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB",
              "OLB": "LB", "MLB": "LB", "CB": "DB", "S": "DB", "SS": "DB", "FS": "DB", "DB": "DB"}


def _fix_team(t):
    return _TEAM_FIX.get(t, t)


def _rec(w, l, t):
    return f"{w}-{l}" + (f"-{t}" if t else "")


def _r(x, d=1):
    return None if x is None or pd.isna(x) else round(float(x), d)


def _skills_injuries(inj):
    """{(season, week, team): ({position group: [out or doubtful, questionable]}, worst QB status or None)}"""
    out = {}
    if inj is None or len(inj) == 0:
        return out
    d = inj[inj.game_type == "REG"].copy()
    d["team"] = d.team.map(_fix_team)
    d["grp"] = d.position.map(_POS_GROUP)
    d = d[d.report_status.isin(["Out", "Doubtful", "Questionable"]) & d.grp.notna()]
    for (s, w, t), rows in d.groupby(["season", "week", "team"]):
        groups, qb = {}, None
        for r in rows.itertuples():
            od, q = groups.get(r.grp, [0, 0])
            groups[r.grp] = [od + (r.report_status in ("Out", "Doubtful")), q + (r.report_status == "Questionable")]
            if r.grp == "QB":
                rank = {"Out": 3, "Doubtful": 2, "Questionable": 1}
                if qb is None or rank[r.report_status] > rank[qb]:
                    qb = r.report_status
        out[(int(s), int(w), t)] = (groups, qb)
    return out


def build_skills_game(sched, inj, epa=None, first_season=SKILLS_FIRST_SEASON):
    """sched: nflverse schedules, inj: nflverse injuries, epa: the model's per-team-game EPA table (optional).
    Returns {"2014": payload, ..., "index": payload}."""
    g = sched[(sched.game_type == "REG") & (sched.season >= first_season)].dropna(subset=["home_score", "away_score"]).copy()
    g["home_team"] = g.home_team.map(_fix_team)
    g["away_team"] = g.away_team.map(_fix_team)
    side = lambda pre, home, opp: pd.DataFrame({
        "season": g.season, "week": g.week, "game_id": g.game_id, "team": g[f"{pre}_team"], "pf": g[f"{pre}_score"],
        "pa": g[f"{opp}_score"], "home": home, "qb": g[f"{pre}_qb_name"]})
    tg = pd.concat([side("home", 1, "away"), side("away", 0, "home")], ignore_index=True)
    tg["margin"] = tg.pf - tg.pa
    tg["epa_off"], tg["epa_def"] = np.nan, np.nan
    if epa is not None and len(epa):
        e = epa[["game_id", "team", "epa_off", "epa_def"]].copy()
        e["team"] = e.team.map(_fix_team)
        tg = tg.drop(columns=["epa_off", "epa_def"]).merge(e, on=["game_id", "team"], how="left")
    injmap = _skills_injuries(inj)

    def snapshot(prior):
        snap = {}
        for t, d in prior.groupby("team"):
            d = d.sort_values("week")
            h, a = d[d.home == 1], d[d.home == 0]
            usual = d.qb.dropna().mode()
            snap[t] = {
                "n": len(d), "w": int((d.margin > 0).sum()), "l": int((d.margin < 0).sum()), "t": int((d.margin == 0).sum()),
                "hw": int((h.margin > 0).sum()), "hl": int((h.margin < 0).sum()), "ht": int((h.margin == 0).sum()),
                "aw": int((a.margin > 0).sum()), "al": int((a.margin < 0).sum()), "at": int((a.margin == 0).sum()),
                "ppg": d.pf.mean(), "papg": d.pa.mean(),
                "eo": d.epa_off.mean() if d.epa_off.notna().any() else np.nan,
                "ed": d.epa_def.mean() if d.epa_def.notna().any() else np.nan,
                "l3": [("W" if m > 0 else "L" if m < 0 else "T") + (f"{int(m):+d}" if m else "") for m in d.margin.tail(3)],
                "qb": usual.iloc[0] if len(usual) else None,
            }
        s = pd.DataFrame(snap).T
        for col, asc, name in (("ppg", False, "rpf"), ("papg", True, "rpa"), ("eo", False, "reo"), ("ed", True, "red")):
            vals = pd.to_numeric(s[col], errors="coerce")
            ranks = vals.rank(ascending=asc, method="min")
            for t in snap:
                snap[t][name] = None if pd.isna(ranks.get(t)) else int(ranks[t])
        return snap

    def block(sn, t, home, qb_now, season, week):
        x = sn.get(t)
        if not x:
            return None
        groups, qbs = injmap.get((season, week, t), ({}, None))
        venue = _rec(x["hw"], x["hl"], x["ht"]) if home else _rec(x["aw"], x["al"], x["at"])
        return {
            "rec": _rec(x["w"], x["l"], x["t"]), "ven": venue, "ppg": _r(x["ppg"]), "papg": _r(x["papg"]),
            "rpf": x["rpf"], "rpa": x["rpa"], "eo": _r(x["eo"], 2), "ed": _r(x["ed"], 2), "reo": x["reo"], "red": x["red"],
            "l3": x["l3"],
            "qb": None if not qb_now or not x["qb"] else ("same" if qb_now == x["qb"] else "new"),
            "qbs": qbs, "inj": groups,
        }

    out, seasons = {}, []
    for season, gs in g.groupby("season"):
        tgs = tg[tg.season == season]
        games = []
        for wk in sorted(gs.week.unique()):
            if wk < SKILLS_MIN_WEEK:
                continue
            sn = snapshot(tgs[tgs.week < wk])
            for r in gs[gs.week == wk].itertuples():
                if pd.isna(r.spread_line) or r.spread_line == 0:
                    continue
                margin = r.home_score - r.away_score
                if margin == 0:
                    continue
                fav_home = r.spread_line > 0
                fav_margin = margin if fav_home else -margin
                line = abs(r.spread_line)
                if fav_margin == line:      # a push: nobody covered
                    continue
                a = block(sn, r.away_team, False, r.away_qb_name, int(season), int(wk))
                h = block(sn, r.home_team, True, r.home_qb_name, int(season), int(wk))
                if not a or not h:
                    continue
                games.append({
                    "id": r.game_id, "s": int(season), "wk": int(wk), "wd": str(r.weekday)[:3], "a": r.away_team, "h": r.home_team,
                    "as": int(r.away_score), "hs": int(r.home_score), "sp": float(r.spread_line), "tot": _r(r.total_line),
                    "rest": [_num(r.away_rest), _num(r.home_rest)], "roof": None if pd.isna(r.roof) else r.roof,
                    "surf": None if pd.isna(r.surface) else r.surface, "temp": _num(r.temp), "wind": _num(r.wind),
                    "div": _num(r.div_game) or 0,
                    "w": "H" if margin > 0 else "A", "cv": "F" if fav_margin > line else "D",
                    "A": a, "H": h,
                })
        if games:
            out[str(int(season))] = {"season": int(season), "games": games}
            seasons.append(int(season))
    out["index"] = {"seasons": seasons, "games": sum(len(v["games"]) for k, v in out.items() if k != "index"),
                    "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    return out


def write_skills_game(sched=None, inj=None, epa=None, with_epa=True):
    seasons = list(range(SKILLS_FIRST_SEASON, CURRENT_SEASON + 1))
    if sched is None:
        sched = nfl.load_schedules(seasons).to_pandas()
    if inj is None:
        inj = nfl.load_injuries(seasons).to_pandas()
    if epa is None and with_epa:
        try:
            epa = load_pbp_tables(seasons)[0]     # reads the cache the model already built
        except Exception as e:
            print(f"(No EPA for the skills game this run: {e})")
    files = build_skills_game(sched, inj, epa)
    folder = os.path.join(CARD_DIR, SKILLS_SUBDIR)
    os.makedirs(folder, exist_ok=True)
    for name, payload in files.items():
        with open(os.path.join(folder, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), allow_nan=False)
    print(f"Wrote the skills game to {folder}/: {files['index']['games']} games in {len(files['index']['seasons'])} seasons")


def main():
    if EXPORT_CARDS:
        try:
            write_college_game()
        except Exception as e:   # the game is a bonus; never let it stop the weekly run
            print(f"(Skipped the college game this run: {e})")
    df, base, groups, extras = build_dataset()
    if EXPORT_CARDS:
        try:
            write_skills_game()
        except Exception as e:
            print(f"(Skipped the skills game this run: {e})")
    X_cols = base + [c for g in groups.values() for c in g]
    T_base, T_groups = extras["total_base"], extras["total_groups"]
    T_cols = T_base + [c for g in T_groups.values() for c in g]

    played = df.dropna(subset=["result", "spread_line", "home_moneyline", "away_moneyline"]).copy()
    played = played[played.result != 0]  # drop ties
    played["home_win"] = (played.result > 0).astype(int)
    played_t = df.dropna(subset=["total_actual", "total_line"]).copy()

    upcoming = next_week_games(df).sort_values("gameday")
    results = df[df.result.notna()].set_index("game_id")[["home_score", "away_score", "result"]].to_dict("index")
    log = load_log() if LOG_PREDICTIONS else {}

    # ----- spread model backtest -----
    wf = walk_forward(played, X_cols)
    backtest_report(wf)
    unc = residual_uncertainty(wf)
    print(f"\nTypical miss ({unc['basis']}): standard deviation {unc['sd']:.1f} pts. "
          f"Half of games land within {unc['q25']:+.1f} to {unc['q75']:+.1f} of the predicted margin, "
          f"80% within {unc['q10']:+.1f} to {unc['q90']:+.1f}.")
    if RUN_ABLATION:
        ablation(played, base, groups)
    bt = backtest_summary(wf)

    # ----- over/under model backtest -----
    wf_t = walk_forward_total(played_t, T_cols)
    total_backtest_report(wf_t)
    unc_t = residual_uncertainty(wf_t, "total_actual", "pred_total", default_sd=10.0)
    print(f"\nTypical total miss ({unc_t['basis']}): standard deviation {unc_t['sd']:.1f} pts. "
          f"Half of games land within {unc_t['q25']:+.1f} to {unc_t['q75']:+.1f} of the predicted total, "
          f"80% within {unc_t['q10']:+.1f} to {unc_t['q90']:+.1f}.")
    if RUN_ABLATION:
        total_ablation(played_t, T_base, T_groups)
    bt_t = backtest_total_summary(wf_t)

    # ----- Refit on all played games, predict next week -----
    reg, clf = make_models()
    reg.fit(played[X_cols], played.result)
    clf.fit(played[X_cols], played.home_win)
    tmod = make_total_model()
    tmod.fit(played_t[T_cols], played_t.total_actual)

    coefs = pd.Series(reg[-1].coef_, index=X_cols)
    print("\nStrongest spread-model inputs (standardized Ridge coefficients, points of margin per 1 std):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12).round(2).to_string())
    tco = pd.Series(tmod[-1].coef_, index=T_cols)
    print("\nStrongest over/under inputs (points of total per 1 std):")
    print(tco.reindex(tco.abs().sort_values(ascending=False).index).head(8).round(2).to_string())

    if upcoming.empty:
        print("\nNo upcoming games found (offseason, or schedule not published yet).")
        if LOG_PREDICTIONS:
            live = finish_log(log, results)
            if EXPORT_CARDS:
                write_accuracy(live, bt, bt_t)
        return

    pred_raw = reg.predict(upcoming[X_cols])
    p_raw = clf.predict_proba(upcoming[X_cols])[:, 1]
    pt_raw = tmod.predict(upcoming[T_cols])
    out = upcoming[["gameday", "away_team", "home_team", "spread_line"]].copy()
    out["away_qb"] = upcoming.a_qb_name
    out["home_qb"] = upcoming.h_qb_name
    out["pred_margin"] = pred_raw.round(1)   # + = home wins by this much
    out["likely_range"] = [f"{m + unc['q10']:+.0f} to {m + unc['q90']:+.0f}" for m in pred_raw]
    out["spread_edge"] = (out.pred_margin - out.spread_line).round(1)  # + = model likes home vs line
    out["p_home_model"] = p_raw.round(3)
    ph, pa_ = implied(upcoming.home_moneyline), implied(upcoming.away_moneyline)
    out["p_home_market"] = (ph / (ph + pa_)).round(3)   # NaN where moneylines aren't posted yet
    out["ml_edge"] = (out.p_home_model - out.p_home_market).round(3)

    print(f"\nPredictions: season {upcoming.season.iloc[0]}, week {upcoming.week.iloc[0]}")
    print("(QBs shown are ASSUMED starters (depth chart + injuries). Set QB_OVERRIDE if wrong. likely_range = 80% range "
          "for the home margin.)")
    print(out.to_string(index=False))
    weather_report(upcoming, extras["weather"])
    ttab = totals_table(upcoming, pt_raw, pred_raw, unc_t)
    print("\nOver/under predictions (weather feeds this model only; total_edge = model minus posted line):")
    print(ttab.to_string(index=False))
    top_picks(out)
    top_totals(ttab)
    inj, inj_note = get_injuries(upcoming)
    injury_report(upcoming, inj, inj_note)

    live = {}
    if LOG_PREDICTIONS:
        log_predictions(log, upcoming, pred_raw, p_raw, unc, pt_raw, unc_t)
        live = finish_log(log, results)

    if EXPORT_CARDS:
        tables = compute_categories(extras["snapshot"], extras["snapshot_prev"])
        ctx = load_context(list(range(CURRENT_SEASON - CTX_YEARS + 1, CURRENT_SEASON + 1)))
        cards = build_cards(upcoming, X_cols, reg, clf, extras, inj, inj_note, df, unc, tables,
                            tot={"model": tmod, "cols": T_cols, "unc": unc_t}, ctx=ctx)
        folder = write_cards(cards)
        write_accuracy(live, bt, bt_t)
        rankings = build_power_rankings(extras, reg, X_cols, upcoming, pred_raw, p_raw, df, tables)
        write_rankings(rankings, folder)
        write_category_rankings(category_payload(tables, rankings), folder)
        first_kick = upcoming.gameday.min()
        write_extras(build_results(df, int(upcoming.season.iloc[0]), INCLUDE_MARKET_LINES),
                     build_trends(team_game_log(df), cards, int(upcoming.season.iloc[0]), first_kick, INCLUDE_MARKET_LINES),
                     build_weather_watch(cards))
        tg_all = team_game_log(df)
        stand = build_standings(tg_all, int(upcoming.season.iloc[0]), first_kick)
        write_standings(stand)
        season_now = int(upcoming.season.iloc[0])
        odds = simulate_season(extras["schedule"], tg_all, {e["team"]: e["power_points"] for e in rankings["teams"]}, season_now)
        _write_json("playoff_odds.json", odds)
        _write_json("highlights.json", build_highlights(upcoming, rankings, odds), folder)
        pages = build_team_pages(extras["schedule"], tg_all, rankings, stand, season_now,
                                 first_kick, cards, INCLUDE_MARKET_LINES, odds)
        print(f"Wrote standings.json, playoff_odds.json, highlights.json and {len(pages)} team pages to {CARD_DIR}/")
        n_recaps = build_recaps(df, season_now)
        print(f"Wrote {n_recaps} game recaps to {CARD_DIR}/recaps/")
        print(f"\nWrote {len(cards)} game cards, index.json, accuracy.json, power_rankings.json and "
              f"category_rankings.json to {CARD_DIR}/ (cards in {folder}/)")
        print("\nPower rankings (points better than an average team; arrows = move since last week):")
        for e in rankings["teams"][:10]:
            mv = e["rank_change"]
            arrow = "  new" if mv is None else (f" +{mv}" if mv > 0 else f" {mv}" if mv < 0 else "  --")
            print(f"  {e['rank']:>2}. {e['team']:<4} {e['power_points']:+5.1f}  {arrow}")
        print_category_leaders(tables)


if __name__ == "__main__":
    import sys
    if "--college" in sys.argv:      # python nfl_model.py --college  -> only rebuilds the college game files (a few seconds)
        write_college_game()
    elif "--skills" in sys.argv:     # python nfl_model.py --skills   -> only rebuilds the Test Your Skills files
        write_skills_game()
    else:
        main()
