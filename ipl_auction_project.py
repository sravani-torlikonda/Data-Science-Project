"""
IPL AUCTION PRICE PREDICTION - "Is this player worth his auction money?"
=======================================================================
The whole project in ONE file. Run it top to bottom:

    pip install -r requirements.txt
    python ipl_auction_project.py

or, in VS Code, click "Run Cell" above each  # %%  line to go one step at a time.
Close each plot window to continue. All plots are also saved in outputs/figures/,
and tables (CV results, test results, overpaid/bargain lists) in outputs/.

QUESTION: Can we predict how much an IPL team will pay for a player at the auction, using only
what is known BEFORE the auction (base price, nationality, past IPL performance), and use the
model to spot the most overpaid players and the biggest bargains?

STEPS
  Step 1  Settings and helper functions (cleaning, season stats, name matching, features)
  Step 2  Load the data and find the problems
  Step 3  One row per sold player; split train (2013-2022) / test (2023 + 2025)
  Step 4  Exploratory data analysis (5 plots)
  Step 5  Cleaning report, name matching, features, correlations
  Step 6  2 baselines + 3 models, cross-validation grouped by auction year, tuning
  Step 7  Test results, error analysis, overpaid/bargain players 2025, feature importance
"""

# %% STEP 1 - Settings and helper functions
# ---------------------------------------------------------------------------------------------
# Everything later steps need: settings, cleaning, season stats, name matching, feature building.
import re
import warnings
from difflib import get_close_matches
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42                                     # fixed seed -> same results every run
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MEGA_AUCTIONS = {2014, 2018, 2022, 2025}      # years where teams rebuilt (almost) their whole squad
TEST_YEARS = [2023, 2025]                     # the latest auctions -> final test set

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8986"

TEAM_NAMES = {   # old names / short codes -> one current name per franchise
    "Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "CSK": "Chennai Super Kings", "MI": "Mumbai Indians", "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders", "SRH": "Sunrisers Hyderabad", "RR": "Rajasthan Royals",
    "DC": "Delhi Capitals", "PBKS": "Punjab Kings", "GT": "Gujarat Titans", "LSG": "Lucknow Super Giants",
}


def save_and_show(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.show()


def norm(name):
    """Simple name form for comparing: lower case, no dots/hyphens/extra spaces."""
    name = str(name).lower().replace(".", " ").replace("-", " ").replace("'", "")
    name = name.replace("mohammad ", "mohammed ").replace("md ", "mohammed ")
    return re.sub(r"\s+", " ", name).strip()


# ---- 1a. auction data -------------------------------------------------------------------------
def load_auction_raw():
    old = pd.read_csv(DATA_DIR / "auction_2013_2023.csv")
    new = pd.read_csv(DATA_DIR / "auction_2025.csv")
    return old, new


def load_auction_list_2025():
    """Official 2025 auction list: nationality, role, capped/uncapped. The file has 3 title rows."""
    lst = pd.read_csv(DATA_DIR / "auction_list_2025.csv", header=3)
    lst = lst[pd.to_numeric(lst["List Sr.No."], errors="coerce").notna()]      # drop repeated header rows
    lst["name"] = (lst["First Name"].fillna("").str.strip() + " " + lst["Surname"].fillna("").str.strip()).str.strip()
    lst["Country"] = lst["Country"].str.strip()
    return lst[["name", "Country", "Specialism", "C/U/A"]]


def clean_auction(verbose=False):
    """Both auction files -> one clean table: player, year, team, country, overseas, base_cr, price_cr."""
    old, new = load_auction_raw()
    log = {}

    # 2013-2023 file: prices in LAKHS, sold players only
    log["2013-23 raw rows"] = len(old)
    old = old.drop(columns=["Unnamed: 0"])
    old["Country"] = old["Country"].str.strip()                                  # " India" -> "India"
    n = len(old)
    old = old.drop_duplicates()                                                  # exact duplicate rows
    log["exact duplicates removed"] = n - len(old)
    old["price_cr"] = pd.to_numeric(old["Winning bid"].astype(str).str.replace(",", ""), errors="coerce") / 100
    old["base_cr"] = old["Base price"] / 100                                     # 100 lakh = 1 crore
    old = old.rename(columns={"Player": "player", "Team": "team", "Year": "year", "Country": "country"})
    old = old[["player", "year", "team", "country", "base_cr", "price_cr"]]

    # 2025 file: prices in CRORES, also lists retained / unsold / not-auctioned players
    log["2025 raw rows"] = len(new)
    new = new.rename(columns={"Players": "player", "Team": "team"})
    new["player"] = new["player"].str.strip()
    retained = new["Base"].str.strip() == "-"                                    # kept by their team, no auction
    unsold = new["Sold"].isin(["Unsold", "TBA"])                                 # no price paid
    log["2025 retained (no auction) removed"] = int(retained.sum())
    log["2025 unsold / never auctioned removed"] = int((unsold & ~retained).sum())
    new = new[~retained & ~unsold].copy()
    new["base_cr"] = pd.to_numeric(new["Base"], errors="coerce")
    new["price_cr"] = pd.to_numeric(new["Sold"], errors="coerce")
    new["year"] = 2025
    # nationality is not in this file -> take it from the official 2025 auction list
    lst = load_auction_list_2025()
    by_name = dict(zip(lst["name"].map(norm), lst["Country"]))
    past = dict(zip(old["player"].map(norm), old["country"]))                    # or from earlier auctions

    def country_for(p):
        k = norm(p)
        if k in by_name:
            return by_name[k]
        if k in past:
            return past[k]
        close = get_close_matches(k, list(by_name), n=1, cutoff=0.85)           # small spelling differences
        return by_name[close[0]] if close else np.nan

    new["country"] = new["player"].map(country_for)
    new = new[["player", "year", "team", "country", "base_cr", "price_cr"]]

    auction = pd.concat([old, new], ignore_index=True)
    auction["team"] = auction["team"].str.strip().replace(TEAM_NAMES)
    auction["overseas"] = (auction["country"] != "India").astype(float)
    auction.loc[auction["country"].isna(), "overseas"] = np.nan                 # unknown stays unknown
    auction["mega_auction"] = auction["year"].isin(MEGA_AUCTIONS).astype(int)
    log["clean auction rows"] = len(auction)
    log["unknown nationality"] = int(auction["country"].isna().sum())
    if verbose:
        for k, v in log.items():
            print(f"  {k:<40} {v}")
    return auction.reset_index(drop=True)


# ---- 1b. ball-by-ball -> season stats ---------------------------------------------------------
def load_ball_by_ball():
    bbb = pd.read_csv(DATA_DIR / "ipl_ball_by_ball.csv.gz")
    matches = pd.read_csv(DATA_DIR / "ipl_matches.csv")
    season = pd.to_datetime(matches["match_date"]).dt.year
    bbb["season"] = bbb["ID"].map(dict(zip(matches["match_number"], season)))
    return bbb


def player_season_stats(bbb):
    """One row per (player, season) with batting, bowling and keeping numbers."""
    legal_bat = bbb["ExtraType"].ne("wides")                                    # wides are not balls faced
    legal_bowl = ~bbb["ExtraType"].isin(["wides", "noballs"])                   # wides/no-balls are re-bowled
    bowler_runs = bbb["TotalRun"] - np.where(bbb["ExtraType"].isin(["byes", "legbyes"]), bbb["ExtrasRun"], 0)
    bowler_wkt = (bbb["IsWicketDelivery"] == 1) & ~bbb["Kind"].isin(
        ["run out", "retired hurt", "retired out", "obstructing the field"])

    bat = bbb.assign(balls=legal_bat.astype(int), four=(bbb["BatsmanRun"] == 4).astype(int),
                     six=(bbb["BatsmanRun"] == 6).astype(int),
                     out=(bbb["PlayerOut"] == bbb["Batter"]).astype(int))
    bat = bat.groupby(["Batter", "season"]).agg(runs=("BatsmanRun", "sum"), balls_faced=("balls", "sum"),
                                                fours=("four", "sum"), sixes=("six", "sum"),
                                                outs=("out", "sum"), bat_matches=("ID", "nunique"))
    bowl = bbb.assign(balls=legal_bowl.astype(int), conceded=bowler_runs, wkt=bowler_wkt.astype(int))
    bowl = bowl.groupby(["Bowler", "season"]).agg(balls_bowled=("balls", "sum"), runs_conceded=("conceded", "sum"),
                                                  wickets=("wkt", "sum"), bowl_matches=("ID", "nunique"))
    stump = bbb[bbb["Kind"] == "stumped"].groupby(["FieldersInvolved", "season"]).size().rename("stumpings")

    bat.index.names = bowl.index.names = stump.index.names = ["player", "season"]
    stats = bat.join(bowl, how="outer").join(stump, how="left").fillna(0)
    stats["matches"] = stats[["bat_matches", "bowl_matches"]].max(axis=1)
    return stats.reset_index()


# ---- 1c. name matching ------------------------------------------------------------------------
def match_names(auction_players, stats, verbose=False, auction_years=None):
    """Map each auction name (e.g. 'Ravichandran Ashwin') to a scorecard name (e.g. 'R Ashwin').

    Rules, in order:
      1. exact same name                                   'Rinku Singh' -> 'Rinku Singh'
      2. official full-name list from the 2024 squads      'Ruturaj Gaikwad' -> 'RD Gaikwad'
      3. same surname + same first letter                  'Ravichandran Ashwin' -> 'R Ashwin'
         (if several fit: prefer an exact first name, then the one who actually played in the
          season he was bought for; if still several -> 'ambiguous', left unmatched)
      4. very close spelling (typos)
    Unmatched players are treated as having NO previous IPL record.
    """
    sc_names = sorted(stats["player"].unique())
    sc_norm = {norm(n): n for n in sc_names}
    official = pd.read_csv(DATA_DIR / "player_names_2024.csv")
    official = {norm(l): s for l, s in zip(official["longName"], official["Name"]) if s in set(sc_names)}

    by_surname = {}
    for n in sc_names:
        toks = norm(n).split()
        for k in (1, 2):                                   # surnames like 'de kock', 'du plessis'
            if len(toks) > k:
                by_surname.setdefault(" ".join(toks[-k:]), []).append(n)

    seasons_of = stats.groupby("player")["season"].apply(set).to_dict()
    mapping, how = {}, {}
    for p in auction_players:
        k = norm(p)
        toks = k.split()
        if len(toks) > 2 and all(len(t) == 1 for t in toks[:-1]):          # 'r p singh' -> 'rp singh'
            toks = ["".join(toks[:-1]), toks[-1]]
            k = " ".join(toks)
        if k in sc_norm:
            mapping[p], how[p] = sc_norm[k], "exact"
            continue
        if k in official:
            mapping[p], how[p] = official[k], "official list"
            continue
        cands = set()
        for kk in (1, 2):
            if len(toks) > kk:
                for c in by_surname.get(" ".join(toks[-kk:]), []):
                    first = c.split()[0]
                    if first.isupper() and len(first) <= 3:      # initials like 'R', 'RP', 'MS'
                        if first[0].lower() == toks[0][0]:
                            cands.add(c)
                    elif norm(first) == toks[0]:                   # full first name must match exactly
                        cands.add(c)
        if len(cands) > 1:
            exact_first = {c for c in cands if norm(c).split()[0] == toks[0]}
            cands = exact_first or cands
        if len(cands) > 1 and auction_years is not None:                   # who played after being bought?
            yrs = auction_years.get(p, [])
            active = {c for c in cands if any(y in seasons_of[c] or y + 1 in seasons_of[c] for y in yrs)}
            if len(active) == 1:
                mapping[p], how[p] = active.pop(), "surname + initial (played that season)"
                continue
        if len(cands) == 1:
            mapping[p], how[p] = cands.pop(), "surname + initial"
            continue
        if len(cands) > 1:
            how[p] = "ambiguous"
            continue
        close = get_close_matches(k, list(sc_norm), n=1, cutoff=0.9)
        if close:
            mapping[p], how[p] = sc_norm[close[0]], "close spelling"
        else:
            how[p] = "not found"
    if verbose:
        print(pd.Series(how).value_counts().to_string())
    return mapping, how


# ---- 1d. features and the modelling table -----------------------------------------------------
def career_features(stats, name, year):
    """Stats of one player from ALL seasons BEFORE the auction year (no peeking into the future)."""
    s = stats[(stats["player"] == name) & (stats["season"] < year)]
    if s.empty:
        return {"ipl_seasons": 0}
    last = s[s["season"] == year - 1]
    recent = s[s["season"] >= year - 3]
    f = {
        "ipl_seasons": s["season"].nunique(),
        "career_matches": s["matches"].sum(),
        "career_runs": s["runs"].sum(),
        "career_wickets": s["wickets"].sum(),
        "career_strike_rate": 100 * s["runs"].sum() / s["balls_faced"].sum() if s["balls_faced"].sum() >= 30 else np.nan,
        "career_economy": 6 * s["runs_conceded"].sum() / s["balls_bowled"].sum() if s["balls_bowled"].sum() >= 60 else np.nan,
        "recent_runs": recent["runs"].sum(),               # last 3 seasons
        "recent_wickets": recent["wickets"].sum(),
        "recent_sixes": recent["sixes"].sum(),
        "last_season_matches": last["matches"].sum(),
        "last_season_runs": last["runs"].sum(),
        "last_season_wickets": last["wickets"].sum(),
        "years_since_last_played": year - s["season"].max(),
        "is_keeper": int(s["stumpings"].sum() > 0),
    }
    faced, bowled = s["balls_faced"].sum(), s["balls_bowled"].sum()
    f["batting_share"] = faced / (faced + bowled) if faced + bowled > 0 else np.nan   # 1 = pure batter, 0 = pure bowler
    return f


def build_dataset(auction, stats, mapping, how):
    """Final table: one row per player sold at an auction (2013-2023, 2025)."""
    auction = auction.copy()
    auction["scorecard_name"] = auction["player"].map(mapping)
    auction["match_type"] = auction["player"].map(how)
    rows = [career_features(stats, n, y) if isinstance(n, str) else {"ipl_seasons": 0}
            for n, y in zip(auction["scorecard_name"], auction["year"])]
    df = pd.concat([auction.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df["ipl_seasons"] = df["ipl_seasons"].fillna(0)
    df["has_ipl_record"] = (df["ipl_seasons"] > 0).astype(int)
    count_cols = ["career_matches", "career_runs", "career_wickets", "recent_runs", "recent_wickets",
                  "recent_sixes", "last_season_matches", "last_season_runs", "last_season_wickets", "is_keeper"]
    df[count_cols] = df[count_cols].fillna(0)        # no IPL record -> 0 matches, 0 runs, ...
    df["log_price"] = np.log(df["price_cr"])         # prices are very skewed -> model the log
    df["log_base"] = np.log(df["base_cr"])
    return df


class BasePriceModel(RegressorMixin, BaseEstimator):
    """BASELINE: 'every player sells at his base price'. Predicts log(base price)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["log_base"])


FEATURES = ["log_base", "overseas", "mega_auction", "year", "has_ipl_record", "ipl_seasons",
            "career_matches", "career_runs", "career_wickets", "career_strike_rate", "career_economy",
            "recent_runs", "recent_wickets", "recent_sixes", "last_season_matches", "last_season_runs",
            "last_season_wickets", "years_since_last_played", "is_keeper", "batting_share"]


# %% STEP 2 - Load the data and look at it
# ---------------------------------------------------------------------------------------------
# 3 sources:
#   1. auction_2013_2023.csv   - every player SOLD at IPL auctions 2013-2023 (prices in LAKHS)
#   2. auction_2025.csv        - the 2025 mega auction (prices in CRORES; also retained/unsold players)
#   3. ipl_ball_by_ball.csv.gz - every ball of every IPL match 2008-2026
# Goal: understand each file and FIND the problems (step 5 fixes them).
print("=" * 30, "STEP 2: LOAD THE DATA", "=" * 30)
old, new = load_auction_raw()
print("2013-2023 file:", old.shape)
print(old.head())
print(old.dtypes)
# -> 'Winning bid' is stored as TEXT, not a number. Prices are in LAKHS (100 lakh = 1 crore).

print("\nPlayers per auction year:\n", old["Year"].value_counts().sort_index().to_dict())
print("\nCountry values (look at the spaces!):", old["Country"].value_counts().head(8).to_dict())
print("\nExact duplicate rows:", old.drop(columns="Unnamed: 0").duplicated().sum(), "(ignoring the row-number column)")
print(old[old.duplicated(subset=["Player", "Year"], keep=False)].sort_values("Player"))
# -> ' India' and 'India' are counted separately because of a leading space.
# -> 3 rows in 2023 are exact copies. Two different players called 'Amit Mishra' were bought in 2014!

print("\n2025 file:", new.shape)
print(new.head())
print("Base price values:", new["Base"].value_counts().head(5).to_dict())
print("Sold values      :", new["Sold"].value_counts().head(5).to_dict())
# -> Base '-' means the player was RETAINED by his team (no auction, no market price).
# -> Sold 'TBA' = never came up for bidding, 'Unsold' = nobody bid. Neither has a price.
# -> prices here are in CRORES, not lakhs, and there is no nationality column.

bbb = load_ball_by_ball()
print("\nBall-by-ball:", bbb.shape)
print(bbb.head())
print("Seasons:", bbb["season"].min(), "to", bbb["season"].max(), "| matches:", bbb["ID"].nunique())
print("Example scorecard names:", bbb["Batter"].drop_duplicates().head(8).tolist())
# -> scorecards use short names like 'V Kohli', but the auction files use 'Virat Kohli'.
#    We have to MATCH the two (step 5).


# %% STEP 3 - One row per sold player, and the train / test split
# ---------------------------------------------------------------------------------------------
# ONE ROW = one player sold at one auction (e.g. "Rishabh Pant, 2025, sold for 27 cr").
# INPUTS  = things known BEFORE the auction: base price, overseas or not, IPL record in earlier seasons.
# TARGET  = the price paid (in crore).
# SPLIT   train: auctions 2013-2022  |  test: auctions 2023 and 2025 (used ONCE, in step 7)
# Splitting by TIME tests the model like real life: learn from past auctions, predict the next one.
print("\n" + "=" * 30, "STEP 3: BUILD THE TABLE AND SPLIT", "=" * 30)
print("Cleaning the auction files:")
auction = clean_auction(verbose=True)
stats = player_season_stats(bbb)
years = auction.groupby("player")["year"].apply(list).to_dict()
print("\nName matching (unique auction names):")
mapping, how = match_names(auction["player"].unique(), stats, verbose=True, auction_years=years)
df = build_dataset(auction, stats, mapping, how)

train = df[~df["year"].isin(TEST_YEARS)].copy()
test = df[df["year"].isin(TEST_YEARS)].copy()
print(f"\nAll rows : {len(df)} players sold at auctions {sorted(df['year'].unique())}")
print(f"TRAIN    : {len(train)} rows, auctions {sorted(train['year'].unique())}")
print(f"TEST     : {len(test)} rows, auctions {sorted(test['year'].unique())}")
print("\nOne example row:")
print(df[df["player"] == "Rishabh Pant"].iloc[-1][["player", "year", "team", "base_cr", "price_cr", "overseas",
                                                      "ipl_seasons", "career_runs", "last_season_runs"]])


# %% STEP 4 - Exploratory data analysis (training auctions only)
# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 30, "STEP 4: EDA", "=" * 30)

# 4.1 How are prices spread?
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
a1.hist(train["price_cr"], bins=50, color=BLUE)
a1.set(xlabel="Price (crore)", ylabel="Players", title="Raw prices: most players are cheap")
a2.hist(np.log(train["price_cr"]), bins=40, color=BLUE)
a2.set(xlabel="log(price)", title="After taking the log: much more even")
save_and_show(fig, "01_price_distribution")
print(train["price_cr"].describe().round(2))
# INSIGHT: half the players go for 1 crore or less, a few go for 10+ crore -> very skewed.
#          So we predict log(price); otherwise a few stars would dominate the model.

# 4.2 Prices over the years (mega vs mini auctions)
by_year = df.groupby("year")["price_cr"].agg(["median", "max", "count"])
fig, ax = plt.subplots(figsize=(8, 3.6))
colors = [ORANGE if y in MEGA_AUCTIONS else BLUE for y in by_year.index]
ax.bar(by_year.index.astype(str), by_year["max"], color=colors)
for i, (y, r) in enumerate(by_year.iterrows()):
    ax.text(i, r["max"] + 0.3, f"{r['max']:.1f}", ha="center", fontsize=8)
ax.set(ylabel="Most expensive player (crore)", title="Top price per auction (orange = mega auction)")
save_and_show(fig, "02_top_price_per_year")
print(by_year.round(2))
# INSIGHT: top prices climbed from ~6 cr (2013) to 27 cr (2025). Mega auctions sell 150-200 players,
#          mini auctions far fewer. The 2025 record is far above anything in the training years.

# 4.3 Base price vs final price
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(train["base_cr"], train["price_cr"], s=12, alpha=0.5, color=BLUE)
lims = [0.08, 30]
ax.plot(lims, lims, color=ORANGE, lw=2, label="sold at base price")
ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, xlabel="Base price (crore, log scale)",
       ylabel="Final price (crore, log scale)", title="Base price vs final price")
ax.legend()
save_and_show(fig, "03_base_vs_final_price")
print("Share sold exactly at base price:", (train["price_cr"] == train["base_cr"]).mean().round(2))
# INSIGHT: about half the players are sold exactly at their base price (on the orange line).
#          The rest go higher - sometimes 10-20x - because teams bid against each other.

# 4.4 Overseas vs Indian players
fig, ax = plt.subplots(figsize=(5.5, 3.6))
groups_nat = [train.loc[train["overseas"] == 0, "price_cr"], train.loc[train["overseas"] == 1, "price_cr"]]
ax.boxplot(groups_nat, showfliers=False, widths=0.5)
ax.set_xticks([1, 2]); ax.set_xticklabels(["Indian", "Overseas"])
ax.set(ylabel="Price (crore)", title="Price: Indian vs overseas players")
save_and_show(fig, "04_price_by_nationality")
print(train.groupby("overseas")["price_cr"].median())
# INSIGHT: overseas players have a higher median price - teams have only 4 overseas slots,
#          so they buy fewer but more established foreign players.

# 4.5 Does IPL experience matter?
experience = pd.Series(np.select([train["ipl_seasons"] == 0, train["ipl_seasons"] <= 3],
                                 ["no IPL record", "1-3 seasons"], "4+ seasons"), index=train.index)
order = ["no IPL record", "1-3 seasons", "4+ seasons"]
med = train.groupby(experience)["price_cr"].median().reindex(order)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.bar(order, med.values, color=[GREY, BLUE, BLUE])
for i, v in enumerate(med.values):
    ax.text(i, v + 0.03, f"{v:.2f} cr", ha="center")
ax.set(ylabel="Median price (crore)", title="Median price by IPL experience", ylim=(0, med.max() * 1.15))
save_and_show(fig, "05_price_by_experience")
print(experience.value_counts())
# INSIGHT: players with an IPL record sell for clearly more than newcomers.
#          About 40% of sold players have no IPL record -> for them the model can only
#          use base price and nationality.


# %% STEP 5 - Cleaning report, name matching, features
# ---------------------------------------------------------------------------------------------
# CLEANING (clean_auction, step 1): strip spaces in country names, remove exact duplicates,
#   text prices -> numbers, LAKHS -> CRORE, drop 2025 retained/unsold/TBA rows, add 2025 nationality,
#   old team names -> current names.  (The cleaning counts were printed in step 3.)
# NAME MATCHING (match_names): 'Ravichandran Ashwin' (auction) -> 'R Ashwin' (scorecard)
# FEATURES (career_features) - all from seasons BEFORE the auction:
#   career: seasons, matches, runs, wickets, strike rate, economy
#   recent: runs / wickets / sixes in the last 3 seasons; matches / runs / wickets last season
#   type  : batting_share (1 = pure batter, 0 = pure bowler), is_keeper (has made stumpings)
#   other : log(base price), overseas, mega_auction, year, has_ipl_record, years since last played
print("\n" + "=" * 30, "STEP 5: CLEANING, NAME MATCHING, FEATURES", "=" * 30)

print("Season stats table:", stats.shape, "(one row per player per season)")
print(stats[stats["player"] == "V Kohli"].tail(3)[["player", "season", "matches", "runs", "balls_faced", "wickets"]])

examples = ["Ravichandran Ashwin", "Ruturaj Gaikwad", "Nathan Coulter-Nile", "Faf du Plessis", "R. P. Singh",
            "Dwayne Bravo", "Wanindu Hasaranga"]
print("\nName-matching examples:")
for p in examples:
    if p in how:
        print(f"  {p:<22} -> {mapping.get(p, '-'):<16} ({how[p]})")
# -> about 4 in 5 names are matched. 'not found' is mostly players who never played an IPL match
#    before being bought (newcomers) - correct to give them no record. A few are real misses
#    (e.g. Wanindu Hasaranga is listed as 'PWH de Silva' on scorecards) - a limitation.

counts = pd.Series(how).value_counts()
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(counts.index[::-1], counts.values[::-1],
        color=[GREY if i in ("not found", "ambiguous") else BLUE for i in counts.index[::-1]])
ax.set(xlabel="Number of auction names", title="How auction names were matched to scorecard names")
save_and_show(fig, "06_name_matching")

print("\nFinal table:", df.shape)
print(df[["player", "year", "price_cr", "base_cr", "overseas", "ipl_seasons", "career_runs",
          "career_wickets", "last_season_runs", "batting_share"]].sample(6, random_state=1))
print("\nMissing values in features (filled inside the model pipeline):")
print(df[FEATURES].isna().sum()[lambda s: s > 0])
# -> strike rate / economy / batting_share are missing for players with no (or too little) IPL record.
#    We fill them with the median INSIDE the model pipeline (learned from training folds only)
#    and add a 0/1 'was missing' column so the model knows the value was filled in.

corr = train[FEATURES].corrwith(train["log_price"]).dropna().sort_values()
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.barh(corr.index, corr.values, color=[ORANGE if v < 0 else BLUE for v in corr.values])
ax.axvline(0, color="black", lw=1)
ax.set(xlabel="Correlation with log(price)", title="Which features go together with a high price?")
save_and_show(fig, "07_feature_correlation")
print(corr.round(2).sort_values(ascending=False).head(8))
# INSIGHT: base price is by far the strongest signal; recent form (runs, wickets, matches last
#          season) comes next. Players who haven't played for years sell for less.


# %% STEP 6 - Models, cross-validation and tuning (training auctions only)
# ---------------------------------------------------------------------------------------------
# Models (simplest first):
#   0a. Typical-price baseline : everyone gets the average (log) price         -> the bar to beat
#   0b. Base-price baseline    : "every player sells at his base price"        -> the common-sense guess
#   1.  Linear Regression      : a weighted sum of the features
#   2.  Random Forest          : many decision trees, averaged
#   3.  Gradient Boosting      : trees built one after another, each fixing the last one's errors
# 5-fold GROUP cross-validation: each auction year stays together, so the model is always checked
# on auctions it hasn't seen. GridSearchCV tries a few settings for the tree models ("tuning").
print("\n" + "=" * 30, "STEP 6: MODELS (takes about 1 minute)", "=" * 30)
X, y, groups = train[FEATURES], train["log_price"], train["year"]
print("Training rows:", len(X), "| auctions:", sorted(groups.unique()))

fill = ("fill_missing", SimpleImputer(strategy="median", add_indicator=True))   # median + 'was missing' flag
model_specs = {
    "Typical-price baseline": (DummyRegressor(strategy="mean"), {}),
    "Base-price baseline": (BasePriceModel(), {}),
    "Linear Regression": (Pipeline([fill, ("scale", StandardScaler()), ("model", LinearRegression())]), {}),
    "Random Forest": (
        Pipeline([fill, ("model", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED))]),
        {"model__min_samples_leaf": [3, 10], "model__max_features": [0.5, 1.0]}),
    "Gradient Boosting": (
        Pipeline([fill, ("model", HistGradientBoostingRegressor(random_state=SEED))]),
        {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15]}),
}

mae_crore = make_scorer(lambda yt, yp: mean_absolute_error(np.exp(yt), np.exp(yp)), greater_is_better=False)
scoring = {"mae_log": "neg_mean_absolute_error", "mae_cr": mae_crore, "r2": "r2"}
cv = GroupKFold(n_splits=5)

models, rows = {}, []
for name, (model, settings) in model_specs.items():
    print(f"Training {name} ...")
    if settings:
        search = GridSearchCV(model, settings, scoring=scoring, refit="mae_log", cv=cv)
        search.fit(X, y, groups=groups)
        best, i, res = search.best_estimator_, search.best_index_, search.cv_results_
        score = {m: res[f"mean_test_{m}"][i] for m in scoring}
        spread = res["std_test_mae_log"][i]
        settings_used = search.best_params_
    else:                                                    # nothing to tune: plain cross-validation
        res = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scoring)
        score = {m: res[f"test_{m}"].mean() for m in scoring}
        spread = res["test_mae_log"].std()
        best = model.fit(X, y)
        settings_used = "no settings to tune"
    models[name] = best
    rows.append({"model": name, "MAE_log": -score["mae_log"], "MAE_log_spread": spread,
                 "MAE_crore": -score["mae_cr"], "R2_log": score["r2"], "settings": settings_used})

results = pd.DataFrame(rows).set_index("model")
print("\nCross-validation results (lower MAE = better):")
print(results.drop(columns="settings").round(3).to_string())
print("\nSettings:")
for name, r in results.iterrows():
    print(f"  {name}: {r['settings']}")
results.to_csv(OUT_DIR / "cv_results.csv")
# MAE_log of 0.5 means the prediction is typically off by a factor of about e^0.5 = 1.65 (65%).

# Pick the winner by cross-validation (never by the test set!)
best_name = results.drop(index=["Typical-price baseline", "Base-price baseline"])["MAE_log"].idxmin()
print("\nChosen model:", best_name)

fig, ax = plt.subplots(figsize=(7, 3.4))
names = results.index[::-1]
ax.barh(names, results.loc[names, "MAE_log"], xerr=results.loc[names, "MAE_log_spread"], capsize=4,
        color=[GREY if "baseline" in n else BLUE for n in names])
for i, (v, sd) in enumerate(zip(results.loc[names, "MAE_log"], results.loc[names, "MAE_log_spread"])):
    ax.text(v + sd + 0.02, i, f"{v:.2f}", va="center")
ax.set_xlim(0, results["MAE_log"].max() * 1.25)
ax.set(xlabel="MAE on log(price) - lower is better", title="5-fold cross-validation (grouped by auction year)")
save_and_show(fig, "08_model_comparison_cv")


# %% STEP 7 - Final test on the untouched 2023 + 2025 auctions, and explaining the results
# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 30, "STEP 7: TEST AND EVALUATION", "=" * 30)
print("Test rows:", len(test), "| chosen model:", best_name)


def scores(y_log, p_log):
    real, pred = np.exp(y_log), np.exp(p_log)
    return {"MAE_log": mean_absolute_error(y_log, p_log),
            "MAE_crore": mean_absolute_error(real, pred),
            "median_%_error": 100 * np.median(np.abs(pred - real) / real),
            "R2_log": r2_score(y_log, p_log)}


# 7.1 The final exam: every model on the test auctions, ONCE
rows, preds = [], {}
for name, m in models.items():
    p = m.predict(test[FEATURES])
    preds[name] = p
    rows.append({"model": name, **scores(test["log_price"], p)})
test_results = pd.DataFrame(rows).set_index("model")
print(test_results.round(3).to_string())
test_results.to_csv(OUT_DIR / "test_results.csv")
# Note: the base-price baseline has a tiny MEDIAN error because many players sell exactly at base -
# but it badly underestimates every player with a bidding war (see MAE in crore).

# 7.2 Predicted vs real price (chosen model)
test["pred_cr"] = np.exp(preds[best_name])
test["ratio"] = test["price_cr"] / test["pred_cr"]          # >1 = paid more than the model expected
fig, ax = plt.subplots(figsize=(5.5, 5))
for yr, c in [(2023, BLUE), (2025, ORANGE)]:
    t = test[test["year"] == yr]
    ax.scatter(t["price_cr"], t["pred_cr"], s=14, alpha=0.6, color=c, label=f"{yr} auction")
lims = [0.15, 35]
ax.plot(lims, lims, color="black", lw=1.5, label="perfect prediction")
ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims, xlabel="Real price (crore, log)",
       ylabel="Predicted price (crore, log)", title=f"{best_name} on the test auctions")
ax.legend()
save_and_show(fig, "09_predicted_vs_real")

# 7.3 Error analysis - which players are hard to price?
test["abs_log_error"] = np.abs(np.log(test["ratio"]))
test["group_experience"] = np.where(test["has_ipl_record"] == 1, "has IPL record", "no IPL record")
test["group_nationality"] = test["overseas"].map({0: "Indian", 1: "Overseas"})
test["group_price"] = pd.cut(test["price_cr"], [0, 1, 5, 100], labels=["up to 1 cr", "1-5 cr", "above 5 cr"])
for g in ["year", "group_experience", "group_nationality", "group_price"]:
    t = test.groupby(g, observed=True).agg(players=("player", "size"),
                                           typically_off_by_x=("abs_log_error", lambda e: np.exp(e.mean())))
    t["MAE_crore"] = test.groupby(g, observed=True).apply(lambda d: mean_absolute_error(d["price_cr"], d["pred_cr"]))
    # typically_off_by_x = 1.9 means the prediction is typically 1.9 times too high or too low
    print(f"\nError by {g}:\n", t.round(2))

band = test.groupby("group_price", observed=True).apply(lambda d: (d["pred_cr"] - d["price_cr"]).mean())
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(band.index.astype(str), band.values, color=[BLUE if v > 0 else ORANGE for v in band.values])
ax.axhline(0, color="black", lw=1)
ax.set(xlabel="Real price band", ylabel="Average (predicted - real), crore",
       title="Cheap players over-estimated, stars under-estimated")
save_and_show(fig, "10_error_by_price_band")
# -> tree models can't predict above the highest price they have seen in training (16.25 cr),
#    so 2025's 20-27 cr stars are always under-predicted.

# 7.4 The fun part: overpaid and underpaid in the 2025 mega auction
t25 = test[test["year"] == 2025]
cols = ["player", "team", "base_cr", "price_cr", "pred_cr", "ratio"]
over = t25.sort_values("ratio", ascending=False)[cols].head(10)
under = t25[t25["has_ipl_record"] == 1].sort_values("ratio")[cols].head(10)
print("\nMOST 'OVERPAID' in 2025 (paid far more than the model expected):")
print(over.round(2).to_string(index=False))
print("\nBIGGEST 'BARGAINS' in 2025 (paid far less than the model expected; players with an IPL record):")
print(under.round(2).to_string(index=False))
over.to_csv(OUT_DIR / "overpaid_2025.csv", index=False)
under.to_csv(OUT_DIR / "bargains_2025.csv", index=False)

# 7.5 How much does the base price carry? Same model, trained WITHOUT base price
no_base = [f for f in FEATURES if f != "log_base"]
m2 = clone(models[best_name])
m2.fit(train[no_base], train["log_price"])
s2 = scores(test["log_price"], m2.predict(test[no_base]))
print("\nWithout base price:", {k: round(float(v), 3) for k, v in s2.items()})
print("With base price   :", {k: round(float(v), 3) for k, v in test_results.loc[best_name].items()})

# 7.6 Which features matter most? (permutation importance on the test auctions)
imp = permutation_importance(models[best_name], test[FEATURES], test["log_price"], n_repeats=10,
                             random_state=SEED, scoring="neg_mean_absolute_error")
importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values()
print("\nIncrease in MAE (log) when each feature is shuffled:")
print(importance.sort_values(ascending=False).round(3).head(10))
top = importance.tail(10)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(top.index, top.values, color=GREEN)
ax.set(xlabel="Increase in error when shuffled", title="Top 10 features the model relies on")
save_and_show(fig, "11_feature_importance")

print("\nDone. Figures are in outputs/figures/, tables in outputs/.")
