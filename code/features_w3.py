"""
Wave3: Player-Form Granular Strength — Feature Engineering
Build time-varying player-level team strength features for WC 2026 prediction.
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

DATA_DIR = "/home/user/research/fifa_data"
EAFC_PATH = "/home/user/research/data/eafc25/fifaRatings.csv"
INTL_PATH = "/home/user/research/data/intl-results/results.csv"

WC_START = "2026-06-11"  # tournament start — all features must be pre-this date
PEAK_MIN, PEAK_MAX = 23, 29  # age window for peak performance


def load_base_data():
    matches = pd.read_csv(f"{DATA_DIR}/matches_detailed.csv")
    teams = pd.read_csv(f"{DATA_DIR}/teams.csv")
    squads = pd.read_csv(f"{DATA_DIR}/squads_and_players.csv")
    return matches, teams, squads


def normalize_name(name):
    """Lowercase, strip accents, remove non-alpha chars."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^a-z ]", "", ascii_name.lower()).strip()


def fuzzy_match_name(target, candidates):
    """Return (best_score, best_idx) using character bigram TF-IDF cosine sim."""
    all_names = [target] + list(candidates)
    try:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        mat = vec.fit_transform(all_names)
        sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
        best_idx = int(np.argmax(sims))
        return sims[best_idx], best_idx
    except Exception:
        return 0.0, 0


def match_players_to_ratings(squads_df, ratings_df):
    """Match WC squad players to EAFC25 ratings using TF-IDF cosine similarity on name chars."""
    ratings_df = ratings_df.copy()
    rating_names_raw = ratings_df["PlayerName"].tolist()
    rating_names_norm = [normalize_name(n) for n in rating_names_raw]

    # Build a fast exact lookup on normalized names
    norm_to_idx = {}
    for i, n in enumerate(rating_names_norm):
        norm_to_idx.setdefault(n, i)

    # Vectorize all rating names once
    squad_names_raw = squads_df["player_name"].tolist()
    squad_names_norm = [normalize_name(n) for n in squad_names_raw]

    all_names = squad_names_norm + rating_names_norm
    try:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
        mat = vec.fit_transform(all_names)
        n_squad = len(squad_names_norm)
        squad_mat = mat[:n_squad]
        rating_mat = mat[n_squad:]
        # Batch cosine similarity: (n_squad, n_ratings)
        sim_matrix = cosine_similarity(squad_mat, rating_mat)
    except Exception as e:
        print(f"  TF-IDF vectorization failed: {e}")
        sim_matrix = None

    matched_rows = []
    for i, (_, player) in enumerate(squads_df.iterrows()):
        pnorm = squad_names_norm[i]

        # Exact normalized match first
        if pnorm in norm_to_idx:
            j = norm_to_idx[pnorm]
            r = ratings_df.iloc[j]
            matched_rows.append({
                "player_id": player.player_id,
                "team_id": player.team_id,
                "overall": r.OverallRating,
                "pace": r.PaceRating,
                "shoot": r.ShootRating,
                "pass_": r.PassRating,
                "drib": r.DribRating,
                "defend": r.DefenseRating,
                "physical": r.PhysicalRating,
                "match_quality": 1.0,
            })
            continue

        # TF-IDF cosine similarity
        best_score = 0.0
        best_j = 0
        if sim_matrix is not None:
            best_j = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i, best_j])

        if best_score >= 0.7:
            r = ratings_df.iloc[best_j]
            matched_rows.append({
                "player_id": player.player_id,
                "team_id": player.team_id,
                "overall": r.OverallRating,
                "pace": r.PaceRating,
                "shoot": r.ShootRating,
                "pass_": r.PassRating,
                "drib": r.DribRating,
                "defend": r.DefenseRating,
                "physical": r.PhysicalRating,
                "match_quality": best_score,
            })
        else:
            matched_rows.append({
                "player_id": player.player_id,
                "team_id": player.team_id,
                "overall": np.nan, "pace": np.nan, "shoot": np.nan,
                "pass_": np.nan, "drib": np.nan, "defend": np.nan, "physical": np.nan,
                "match_quality": best_score,
            })

    return pd.DataFrame(matched_rows)


def compute_squad_features(squads_df, match_date_str="2026-06-11"):
    """
    Compute static player-granular team features from squads_and_players.csv.
    All aggregations are per team_id.
    """
    match_date = pd.Timestamp(match_date_str)
    squads = squads_df.copy()

    # Compute age at match date
    squads["dob"] = pd.to_datetime(squads["date_of_birth"], errors="coerce")
    squads["age"] = (match_date - squads["dob"]).dt.days / 365.25

    # Is player in peak age window?
    squads["is_peak_age"] = ((squads["age"] >= PEAK_MIN) & (squads["age"] <= PEAK_MAX)).astype(float)

    # Goal productivity = goals / max(caps, 1)
    squads["goals_per_cap"] = squads["goals"] / squads["caps"].clip(lower=1)

    team_feats = {}
    for team_id, grp in squads.groupby("team_id"):
        # Top 11 by market value
        top11 = grp.nlargest(11, "market_value_eur")
        fwd = grp[grp["position"].isin(["FW", "FWD", "ST", "CF", "LW", "RW", "Attacker"])]
        gk = grp[grp["position"].isin(["GK", "Goalkeeper"])]
        def_ = grp[grp["position"].isin(["DEF", "CB", "LB", "RB", "LWB", "RWB", "Defender"])]
        mid = grp[grp["position"].isin(["MF", "MID", "CM", "CDM", "CAM", "LM", "RM", "Midfielder"])]

        feats = {
            "team_id": team_id,
            # Market value features
            "squad_total_mv": grp["market_value_eur"].sum(),
            "squad_top11_mv": top11["market_value_eur"].sum(),
            "squad_top3_mv": grp.nlargest(3, "market_value_eur")["market_value_eur"].sum(),
            # Age features
            "squad_mean_age": grp["age"].mean(),
            "squad_peak_pct": grp["is_peak_age"].mean(),
            "squad_top11_mean_age": top11["age"].mean(),
            # Experience
            "squad_mean_caps": grp["caps"].mean(),
            "squad_top11_mean_caps": top11["caps"].mean(),
            "squad_veteran_count": (grp["caps"] >= 50).sum(),
            # Goal threat
            "squad_fwd_goals": fwd["goals"].sum() if len(fwd) > 0 else 0.0,
            "squad_fwd_goals_per_cap": fwd["goals_per_cap"].mean() if len(fwd) > 0 else 0.0,
            # Positional market value
            "squad_gk_mv": gk["market_value_eur"].max() if len(gk) > 0 else 0.0,
            "squad_def_mv": def_["market_value_eur"].sum() if len(def_) > 0 else 0.0,
            "squad_mid_mv": mid["market_value_eur"].sum() if len(mid) > 0 else 0.0,
            "squad_fwd_mv": fwd["market_value_eur"].sum() if len(fwd) > 0 else 0.0,
            # Age penalty for top players
            "squad_top11_age_penalty": np.abs(top11["age"] - 26).mean(),  # deviation from peak age 26
            # Depth: 12-23 player value
            "squad_depth_mv": grp.nlargest(23, "market_value_eur").iloc[11:]["market_value_eur"].sum() if len(grp) > 11 else 0.0,
        }
        team_feats[team_id] = feats

    return pd.DataFrame(list(team_feats.values())).set_index("team_id")


def compute_fifa_rating_features(squads_df, ratings_df):
    """
    Add FIFA 24/25 (EAFC25) player ratings matched by name,
    then aggregate to team level.
    """
    print("Matching player names to EAFC25 ratings...")
    match_df = match_players_to_ratings(squads_df, ratings_df)
    merged = squads_df.merge(match_df[["player_id", "overall", "pace", "shoot", "pass_", "drib", "defend", "physical", "match_quality"]], on="player_id", how="left")

    coverage = match_df["overall"].notna().mean()
    print(f"  Player match coverage: {coverage:.1%}")

    team_feats = {}
    for team_id, grp in merged.groupby("team_id"):
        rated = grp[grp["overall"].notna()]
        top11_rated = rated.nlargest(11, "overall") if len(rated) >= 11 else rated

        feats = {
            "team_id": team_id,
            "fifa_mean_overall": rated["overall"].mean() if len(rated) > 0 else np.nan,
            "fifa_top11_mean_overall": top11_rated["overall"].mean() if len(top11_rated) > 0 else np.nan,
            "fifa_top3_overall": rated.nlargest(3, "overall")["overall"].mean() if len(rated) >= 3 else np.nan,
            "fifa_mean_pace": rated["pace"].mean() if len(rated) > 0 else np.nan,
            "fifa_mean_shoot": rated["shoot"].mean() if len(rated) > 0 else np.nan,
            "fifa_mean_pass": rated["pass_"].mean() if len(rated) > 0 else np.nan,
            "fifa_mean_defend": rated["defend"].mean() if len(rated) > 0 else np.nan,
            "fifa_mean_physical": rated["physical"].mean() if len(rated) > 0 else np.nan,
            "fifa_match_coverage": grp["match_quality"].mean(),
        }
        team_feats[team_id] = feats

    return pd.DataFrame(list(team_feats.values())).set_index("team_id")


def compute_recent_form(intl_df, teams_df, lookback_days=550):
    """
    Compute team form from international results.
    lookback_days from WC start = covers ~18 months of recent matches.
    Uses only results before WC_START.
    """
    cutoff = pd.Timestamp(WC_START)
    start = cutoff - pd.Timedelta(days=lookback_days)

    df = intl_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] < cutoff)]

    # Map team names: we need to figure out which intl results name matches which WC team name
    wc_team_names = set(teams_df["team_name"].unique())

    # Compute W/D/L for each team
    records = {}

    def update(team, result):
        if team not in records:
            records[team] = {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "matches": 0}
        records[team]["matches"] += 1
        if result == "W":
            records[team]["W"] += 1
        elif result == "D":
            records[team]["D"] += 1
        else:
            records[team]["L"] += 1

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            continue
        # Home result
        if hs > as_:
            update(h, "W"); update(a, "L")
        elif hs < as_:
            update(h, "L"); update(a, "W")
        else:
            update(h, "D"); update(a, "D")

        # Track goals
        records.setdefault(h, {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "matches": 0})
        records.setdefault(a, {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "matches": 0})
        records[h]["GF"] += hs; records[h]["GA"] += as_
        records[a]["GF"] += as_; records[a]["GA"] += hs

    form_df = pd.DataFrame(records).T.reset_index().rename(columns={"index": "intl_name"})
    form_df["win_rate"] = form_df["W"] / form_df["matches"].clip(lower=1)
    form_df["draw_rate"] = form_df["D"] / form_df["matches"].clip(lower=1)
    form_df["loss_rate"] = form_df["L"] / form_df["matches"].clip(lower=1)
    form_df["gd_per_match"] = (form_df["GF"] - form_df["GA"]) / form_df["matches"].clip(lower=1)
    form_df["goals_per_match"] = form_df["GF"] / form_df["matches"].clip(lower=1)

    # Match intl team names to WC team names via TF-IDF char sim
    intl_names = form_df["intl_name"].tolist()
    wc_names = list(wc_team_names)
    name_map = {}
    try:
        all_names = [normalize_name(n) for n in wc_names + intl_names]
        vec2 = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
        mat2 = vec2.fit_transform(all_names)
        n_wc = len(wc_names)
        sims2 = cosine_similarity(mat2[:n_wc], mat2[n_wc:])
        for wi, wc_name in enumerate(wc_names):
            best_j = int(np.argmax(sims2[wi]))
            best_score = float(sims2[wi, best_j])
            if best_score >= 0.6:
                name_map[wc_name] = intl_names[best_j]
    except Exception:
        pass

    # Attach team_id
    teams_with_form = teams_df.copy()
    teams_with_form["intl_name"] = teams_with_form["team_name"].map(name_map)
    teams_with_form = teams_with_form.merge(
        form_df[["intl_name", "win_rate", "draw_rate", "loss_rate", "gd_per_match", "goals_per_match", "matches"]],
        on="intl_name", how="left"
    )

    form_feats = teams_with_form[["team_id", "win_rate", "draw_rate", "loss_rate", "gd_per_match", "goals_per_match", "matches"]].set_index("team_id")
    form_feats = form_feats.rename(columns=lambda c: f"form_{c}")
    print(f"  Form coverage: {form_feats['form_win_rate'].notna().mean():.1%} of WC teams")
    return form_feats


def build_match_features(matches_df, teams_df, squad_feats, form_feats=None, fifa_feats=None):
    """
    For each match, compute (home - away) differences for all team features.
    Returns feature matrix X and labels y.
    """
    completed = matches_df[matches_df["status"] == "Completed"].copy()
    completed["label"] = np.where(
        completed["home_score"] > completed["away_score"], "H",
        np.where(completed["home_score"] < completed["away_score"], "A", "D")
    )

    # Hosts get home advantage encoding
    hosts = {"Mexico", "United States", "Canada", "MEX", "USA", "CAN"}
    completed["home_is_host"] = completed["home_fifa_code"].isin(hosts).astype(float)
    completed["away_is_host"] = completed["away_fifa_code"].isin(hosts).astype(float)

    # Merge Elo and rank from teams
    teams_idx = teams_df.set_index("team_name")
    teams_code_idx = teams_df.set_index("fifa_code")

    def get_team_stat(code, col):
        if code in teams_code_idx.index:
            return teams_code_idx.loc[code, col]
        return np.nan

    completed["home_elo"] = completed["home_fifa_code"].map(lambda c: get_team_stat(c, "elo_rating"))
    completed["away_elo"] = completed["away_fifa_code"].map(lambda c: get_team_stat(c, "elo_rating"))
    completed["home_rank"] = completed["home_fifa_code"].map(lambda c: get_team_stat(c, "fifa_ranking_pre_tournament"))
    completed["away_rank"] = completed["away_fifa_code"].map(lambda c: get_team_stat(c, "fifa_ranking_pre_tournament"))
    completed["home_team_id"] = completed["home_fifa_code"].map(lambda c: get_team_stat(c, "team_id"))
    completed["away_team_id"] = completed["away_fifa_code"].map(lambda c: get_team_stat(c, "team_id"))

    completed["elo_diff"] = completed["home_elo"] - completed["away_elo"]
    completed["rank_diff"] = -(completed["home_rank"] - completed["away_rank"])  # negative = home is better ranked
    completed["host_diff"] = completed["home_is_host"] - completed["away_is_host"]

    # Merge squad features
    def merge_diff(df, feat_df, col_prefix, home_id_col="home_team_id", away_id_col="away_team_id"):
        if feat_df is None:
            return df
        for col in feat_df.columns:
            df[f"home_{col}"] = df[home_id_col].map(feat_df[col])
            df[f"away_{col}"] = df[away_id_col].map(feat_df[col])
            df[f"diff_{col}"] = df[f"home_{col}"] - df[f"away_{col}"]
        return df

    completed = merge_diff(completed, squad_feats, "squad")
    completed = merge_diff(completed, form_feats, "form")
    completed = merge_diff(completed, fifa_feats, "fifa")

    return completed


if __name__ == "__main__":
    print("Loading base data...")
    matches, teams, squads = load_base_data()

    print("Computing squad features...")
    squad_feats = compute_squad_features(squads)
    print(f"  Squad features shape: {squad_feats.shape}")

    print("Loading EAFC25 ratings...")
    ratings = pd.read_csv(EAFC_PATH)
    print(f"  Ratings shape: {ratings.shape}")

    print("Computing FIFA rating features...")
    fifa_feats = compute_fifa_rating_features(squads, ratings)
    print(f"  FIFA features shape: {fifa_feats.shape}")

    print("Loading international results for form...")
    intl = pd.read_csv(INTL_PATH)
    print(f"  Intl results shape: {intl.shape}")

    print("Computing recent form features...")
    form_feats = compute_recent_form(intl, teams)
    print(f"  Form features shape: {form_feats.shape}")

    print("\nBuilding match feature matrix...")
    completed = build_match_features(matches, teams, squad_feats, form_feats, fifa_feats)
    print(f"  Completed matches: {len(completed)}")
    print(f"  Label distribution: {completed['label'].value_counts().to_dict()}")

    # Show available feature columns
    feat_cols = [c for c in completed.columns if c.startswith("diff_") or c in ["elo_diff", "rank_diff", "host_diff"]]
    print(f"\n  Feature columns ({len(feat_cols)}):")
    for c in feat_cols:
        nan_pct = completed[c].isna().mean()
        print(f"    {c}: nan={nan_pct:.0%}")
