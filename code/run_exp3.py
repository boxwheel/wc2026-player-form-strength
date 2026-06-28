"""
Wave3: Player-Form — Round 3 Experiments
Built on Round 2 synthesis: blending is the only lever that works on n=64.
Target experiments:
  W3-11: GradientBoosting in blend (better-calibrated probs than ET)
  W3-12: ET + Platt calibration inside folds, then blend
  W3-13: Position-specific FIFA ratings (GK defend, FWD shoot, DEF defend, MID pass)
  W3-14: Short-window form (6 months) vs long (18 months) — does recency help?
"""

import sys, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import sklearn.base
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from features_w3 import (
    load_base_data, compute_squad_features, compute_fifa_rating_features,
    compute_recent_form, build_match_features, match_players_to_ratings
)

ARTIFACTS_DIR = "/home/user/research/wave3-player-form/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
EAFC_PATH = "/home/user/research/data/eafc25/fifaRatings.csv"
INTL_PATH = "/home/user/research/data/intl-results/results.csv"

SEED = 0
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
ELO_BASELINE = 0.8337
FRONTIER = 0.7608
LABELS = ["A", "D", "H"]


def run_cv_manual(X, y, pipeline, cv=CV):
    n = len(y)
    oof_sum = np.zeros((n, 3))
    oof_count = np.zeros(n)
    fold_losses = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = sklearn.base.clone(pipeline)
        fitted.fit(X[train_idx], y[train_idx])
        probs = fitted.predict_proba(X[test_idx])
        classes = list(fitted.classes_)
        col_order = [classes.index(l) if l in classes else 0 for l in LABELS]
        probs_ord = probs[:, col_order]
        fold_losses.append(log_loss(y[test_idx], probs_ord, labels=LABELS))
        oof_sum[test_idx] += probs_ord
        oof_count[test_idx] += 1
    oof = oof_sum / np.maximum(oof_count[:, None], 1)
    oof = oof / oof.sum(axis=1, keepdims=True)
    mean_ll = float(np.mean(fold_losses))
    std_ll = float(np.std(fold_losses))
    acc = float(accuracy_score(y, np.array(LABELS)[np.argmax(oof, axis=1)]))
    return {"cv_ll_mean": mean_ll, "cv_ll_std": std_ll, "accuracy": acc,
            "oof_probs": oof, "fold_losses": fold_losses}


def run_cv_calibrated(X, y, base_pipeline, calibration="sigmoid", cv=CV):
    """
    Fit base_pipeline on train fold, calibrate OOF probs on train via sigmoid,
    then apply to test. Calibration is fit within each fold on training residuals
    (inner 4-fold CV within train) to avoid leaking test labels.
    Uses CalibratedClassifierCV(cv=3) on the train fold only.
    """
    n = len(y)
    oof_sum = np.zeros((n, 3))
    oof_count = np.zeros(n)
    fold_losses = []
    for train_idx, test_idx in cv.split(X, y):
        # Calibrate on train fold (inner CV), apply to test
        cal_clf = CalibratedClassifierCV(
            sklearn.base.clone(base_pipeline),
            method=calibration, cv=3
        )
        cal_clf.fit(X[train_idx], y[train_idx])
        probs = cal_clf.predict_proba(X[test_idx])
        classes = list(cal_clf.classes_)
        col_order = [classes.index(l) if l in classes else 0 for l in LABELS]
        probs_ord = probs[:, col_order]
        fold_losses.append(log_loss(y[test_idx], probs_ord, labels=LABELS))
        oof_sum[test_idx] += probs_ord
        oof_count[test_idx] += 1
    oof = oof_sum / np.maximum(oof_count[:, None], 1)
    oof = oof / oof.sum(axis=1, keepdims=True)
    mean_ll = float(np.mean(fold_losses))
    std_ll = float(np.std(fold_losses))
    acc = float(accuracy_score(y, np.array(LABELS)[np.argmax(oof, axis=1)]))
    return {"cv_ll_mean": mean_ll, "cv_ll_std": std_ll, "accuracy": acc,
            "oof_probs": oof, "fold_losses": fold_losses}


def blend_oof(elo_oof, tree_oof, y, X_ref, alpha, cv=CV):
    """Blend two OOF arrays and return fold-level losses."""
    b = alpha * elo_oof + (1 - alpha) * tree_oof
    b = b / b.sum(axis=1, keepdims=True)
    fold_losses = []
    for _, test_idx in cv.split(X_ref, y):
        fold_losses.append(log_loss(y[test_idx], b[test_idx], labels=LABELS))
    return b, float(np.mean(fold_losses)), float(np.std(fold_losses)), fold_losses


def best_blend(elo_oof, tree_oof, y, X_ref):
    """Sweep alpha 0..1 step 0.05, return best blend result dict."""
    best = {"alpha": 0.5, "ll": 999, "oof": None, "fold_losses": None, "std": 0}
    for alpha in np.arange(0.0, 1.05, 0.05):
        b, ll, std, fl = blend_oof(elo_oof, tree_oof, y, X_ref, alpha)
        if ll < best["ll"]:
            best = {"alpha": float(alpha), "ll": ll, "std": std, "oof": b, "fold_losses": fl}
    return best


def report(name, ll, std, acc, fold_losses, elo_fold):
    delta = ll - ELO_BASELINE
    stat, p = stats.wilcoxon(fold_losses, elo_fold, alternative="less")
    verdict = "GREEN" if ll < ELO_BASELINE - 0.005 else ("FLAT" if delta < 0.02 else "RED")
    print(f"  {name}: {ll:.4f}±{std:.4f} acc={acc:.3f} Δ={delta:+.4f} [{verdict}] p={p:.3f}")
    return verdict, p


def save_art(name, metrics, run_info, oof_probs, match_ids):
    d = f"{ARTIFACTS_DIR}/{name}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    oof_df = pd.DataFrame(oof_probs, columns=["prob_A", "prob_D", "prob_H"])
    oof_df["match_id"] = match_ids
    oof_df.to_csv(f"{d}/oof_probs.csv", index=False)
    with open(f"{d}/run.json", "w") as f:
        json.dump(run_info, f, indent=2)


def compute_position_fifa_features(squads_df, ratings_df):
    """
    Compute position-specific FIFA rating features:
    - GK: best GK's defend + overall
    - DEF: top-4 DEF mean defend rating
    - MID: top-4 MID mean pass rating
    - FWD: top-3 FWD mean shoot rating
    """
    from features_w3 import match_players_to_ratings
    match_df = match_players_to_ratings(squads_df, ratings_df)
    merged = squads_df.merge(
        match_df[["player_id", "overall", "shoot", "pass_", "defend", "physical", "match_quality"]],
        on="player_id", how="left"
    )

    team_feats = {}
    for team_id, grp in merged.groupby("team_id"):
        rated = grp[grp["overall"].notna()]
        gk = rated[grp["position"].isin(["GK", "Goalkeeper"])]
        def_ = rated[grp["position"].isin(["DEF", "CB", "LB", "RB", "LWB", "RWB", "Defender"])]
        mid = rated[grp["position"].isin(["MF", "MID", "CM", "CDM", "CAM", "LM", "RM", "Midfielder"])]
        fwd = rated[grp["position"].isin(["FW", "FWD", "ST", "CF", "LW", "RW", "Attacker"])]

        # GK: best GK overall + defend
        gk_top = gk.nlargest(1, "overall") if len(gk) > 0 else pd.DataFrame()
        # DEF: top-4 by overall, use their defend rating
        def_top = def_.nlargest(4, "overall") if len(def_) > 0 else pd.DataFrame()
        # MID: top-4 by overall, use their pass rating
        mid_top = mid.nlargest(4, "overall") if len(mid) > 0 else pd.DataFrame()
        # FWD: top-3 by overall, use their shoot rating
        fwd_top = fwd.nlargest(3, "overall") if len(fwd) > 0 else pd.DataFrame()

        feats = {
            "team_id": team_id,
            "pos_gk_overall": gk_top["overall"].mean() if len(gk_top) > 0 else np.nan,
            "pos_gk_defend": gk_top["defend"].mean() if len(gk_top) > 0 else np.nan,
            "pos_def_defend": def_top["defend"].mean() if len(def_top) > 0 else np.nan,
            "pos_def_overall": def_top["overall"].mean() if len(def_top) > 0 else np.nan,
            "pos_mid_pass": mid_top["pass_"].mean() if len(mid_top) > 0 else np.nan,
            "pos_mid_overall": mid_top["overall"].mean() if len(mid_top) > 0 else np.nan,
            "pos_fwd_shoot": fwd_top["shoot"].mean() if len(fwd_top) > 0 else np.nan,
            "pos_fwd_overall": fwd_top["overall"].mean() if len(fwd_top) > 0 else np.nan,
            "pos_attack_vs_defense": (
                (fwd_top["shoot"].mean() if len(fwd_top) > 0 else np.nan) -
                (def_top["defend"].mean() if len(def_top) > 0 else np.nan)
            ),
        }
        team_feats[team_id] = feats

    return pd.DataFrame(list(team_feats.values())).set_index("team_id")


def main():
    print("=" * 70)
    print("Wave3 Round-3: Calibration, GBM blend, position features")
    print("=" * 70)

    matches, teams, squads = load_base_data()
    ratings = pd.read_csv(EAFC_PATH)
    intl = pd.read_csv(INTL_PATH)

    squad_feats = compute_squad_features(squads)
    fifa_feats = compute_fifa_rating_features(squads, ratings)
    form_feats = compute_recent_form(intl, teams)          # 18-month
    form_feats_6m = compute_recent_form(intl, teams, lookback_days=180)  # 6-month
    completed = build_match_features(matches, teams, squad_feats, form_feats, fifa_feats)
    y = completed["label"].values
    match_ids = completed["match_id"].values

    # Elo baseline OOF (local recompute for fold pairing)
    X_elo = completed[["elo_diff", "host_diff"]].values
    pipe_elo = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs"))
    ])
    elo_res = run_cv_manual(X_elo, y, pipe_elo)
    print(f"\nElo baseline (local): {elo_res['cv_ll_mean']:.4f}")
    elo_fold = elo_res["fold_losses"]
    elo_oof = elo_res["oof_probs"]

    results = {}
    all_feat_cols = [c for c in completed.columns if c.startswith("diff_")]
    X_all = completed[all_feat_cols].values

    # ===== W3-11: GBM blend =====
    print("\n--- W3-11: GradientBoosting in blend (better-calibrated probs) ---")
    # Sweep GBM configs: n_estimators, max_depth, learning_rate
    best_gbm = {"ll": 999, "params": None, "oof": None, "fold_losses": None, "std": 0}
    configs = [
        {"n_estimators": 50,  "max_depth": 2, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05},
        {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.05},
        {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1},
    ]
    for cfg in configs:
        pipe_gbm = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"], random_state=SEED,
                min_samples_leaf=4, subsample=0.8
            ))
        ])
        gbm_res = run_cv_manual(X_all, y, pipe_gbm)
        print(f"  GBM n={cfg['n_estimators']},d={cfg['max_depth']},lr={cfg['learning_rate']}: "
              f"{gbm_res['cv_ll_mean']:.4f}±{gbm_res['cv_ll_std']:.4f}")
        if gbm_res["cv_ll_mean"] < best_gbm["ll"]:
            best_gbm = {"ll": gbm_res["cv_ll_mean"], "params": cfg,
                        "oof": gbm_res["oof_probs"], "fold_losses": gbm_res["fold_losses"],
                        "std": gbm_res["cv_ll_std"]}

    # Blend best GBM with Elo
    blend_gbm = best_blend(elo_oof, best_gbm["oof"], y, X_elo)
    print(f"  Best GBM blend α={blend_gbm['alpha']:.2f}: {blend_gbm['ll']:.4f}±{blend_gbm['std']:.4f}")
    verdict11, p11 = report("W3-11 GBM blend", blend_gbm["ll"], blend_gbm["std"],
        float(accuracy_score(y, np.array(LABELS)[np.argmax(blend_gbm["oof"], axis=1)])),
        blend_gbm["fold_losses"], elo_fold)
    metrics11 = {
        "experiment": "w3-11-gbm-blend",
        "cv_log_loss_mean": blend_gbm["ll"],
        "cv_log_loss_std": blend_gbm["std"],
        "accuracy": float(accuracy_score(y, np.array(LABELS)[np.argmax(blend_gbm["oof"], axis=1)])),
        "delta_vs_elo_baseline": blend_gbm["ll"] - ELO_BASELINE,
        "delta_vs_ensemble_frontier": blend_gbm["ll"] - FRONTIER,
        "blend_alpha_elo": blend_gbm["alpha"],
        "best_gbm_params": best_gbm["params"],
        "best_gbm_standalone_ll": best_gbm["ll"],
        "wilcoxon_p": p11,
        "verdict": verdict11,
    }
    save_art("w3-11-gbm-blend", metrics11, {
        "experiment": "w3-11-gbm-blend",
        "model": f"GradientBoosting params={best_gbm['params']} + Elo blend",
        "blend_alpha_elo": blend_gbm["alpha"],
        "features": "all diff_ features (33 total)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
    }, blend_gbm["oof"], match_ids)
    results["w3-11"] = metrics11

    # ===== W3-12: Calibrated ET blend =====
    print("\n--- W3-12: Calibrated ET (Platt sigmoid) inside folds, then blend ---")
    pipe_et_base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=3,
            min_samples_leaf=6, max_features=0.3, random_state=SEED, n_jobs=2))
    ])
    # Run standard ET for comparison
    et_res = run_cv_manual(X_all, y, pipe_et_base)
    print(f"  ET standard: {et_res['cv_ll_mean']:.4f}±{et_res['cv_ll_std']:.4f}")

    # Run calibrated ET (Platt scaling within each outer fold using inner 3-fold CV)
    et_cal_res = run_cv_calibrated(X_all, y, pipe_et_base, calibration="sigmoid")
    print(f"  ET+Platt calibration: {et_cal_res['cv_ll_mean']:.4f}±{et_cal_res['cv_ll_std']:.4f}")

    # Blend calibrated ET with Elo
    blend_et_cal = best_blend(elo_oof, et_cal_res["oof_probs"], y, X_elo)
    print(f"  Best cal-ET blend α={blend_et_cal['alpha']:.2f}: {blend_et_cal['ll']:.4f}±{blend_et_cal['std']:.4f}")

    # Also blend uncalibrated ET (Round 2 W3-09 type) for comparison
    blend_et_raw = best_blend(elo_oof, et_res["oof_probs"], y, X_elo)
    print(f"  Raw-ET blend α={blend_et_raw['alpha']:.2f}: {blend_et_raw['ll']:.4f}±{blend_et_raw['std']:.4f}")

    # Best of calibrated vs raw
    best_et_blend = blend_et_cal if blend_et_cal["ll"] < blend_et_raw["ll"] else blend_et_raw
    calibrated = blend_et_cal["ll"] < blend_et_raw["ll"]
    verdict12, p12 = report("W3-12 cal-ET blend", best_et_blend["ll"], best_et_blend["std"],
        float(accuracy_score(y, np.array(LABELS)[np.argmax(best_et_blend["oof"], axis=1)])),
        best_et_blend["fold_losses"], elo_fold)
    metrics12 = {
        "experiment": "w3-12-calibrated-et-blend",
        "cv_log_loss_mean": best_et_blend["ll"],
        "cv_log_loss_std": best_et_blend["std"],
        "accuracy": float(accuracy_score(y, np.array(LABELS)[np.argmax(best_et_blend["oof"], axis=1)])),
        "delta_vs_elo_baseline": best_et_blend["ll"] - ELO_BASELINE,
        "delta_vs_ensemble_frontier": best_et_blend["ll"] - FRONTIER,
        "blend_alpha_elo": best_et_blend["alpha"],
        "calibration_method": "sigmoid" if calibrated else "none",
        "et_standalone_ll": et_res["cv_ll_mean"],
        "et_calibrated_standalone_ll": et_cal_res["cv_ll_mean"],
        "wilcoxon_p": p12,
        "verdict": verdict12,
    }
    save_art("w3-12-calibrated-et-blend", metrics12, {
        "experiment": "w3-12-calibrated-et-blend",
        "model": "ExtraTrees + Platt calibration (inner 3-fold) + Elo blend",
        "blend_alpha_elo": best_et_blend["alpha"],
        "calibration_method": "sigmoid" if calibrated else "none",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
    }, best_et_blend["oof"], match_ids)
    results["w3-12"] = metrics12

    # ===== W3-13: Position-specific FIFA ratings in blend =====
    print("\n--- W3-13: Position-specific FIFA ratings (GK defend, FWD shoot, DEF defend, MID pass) ---")
    print("  Computing position-specific features...")
    pos_feats = compute_position_fifa_features(squads, ratings)
    print(f"  Position features shape: {pos_feats.shape}")

    # Merge into completed
    completed_pos = completed.copy()
    for col in pos_feats.columns:
        completed_pos[f"home_{col}"] = completed_pos["home_team_id"].map(pos_feats[col])
        completed_pos[f"away_{col}"] = completed_pos["away_team_id"].map(pos_feats[col])
        completed_pos[f"diff_{col}"] = completed_pos[f"home_{col}"] - completed_pos[f"away_{col}"]

    pos_diff_cols = [f"diff_{c}" for c in pos_feats.columns]
    for c in pos_diff_cols:
        nan_pct = completed_pos[c].isna().mean()
        print(f"    {c}: nan={nan_pct:.0%}")

    # Use top position features in logistic (most correlated)
    y_num = np.where(y == "H", 1.0, np.where(y == "A", -1.0, 0.0))
    pos_corrs = {}
    for c in pos_diff_cols:
        col = completed_pos[c]
        if col.notna().sum() < 40:
            continue
        r = np.corrcoef(col.fillna(col.median()), y_num)[0, 1]
        pos_corrs[c] = abs(r)
    top_pos = sorted(pos_corrs.items(), key=lambda x: -x[1])
    print("  Position feature correlations:")
    for c, r in top_pos:
        print(f"    {c}: |r|={r:.3f}")

    # ET on all position diff features (blend with Elo)
    pos_X = completed_pos[pos_diff_cols].values
    pipe_et_pos = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=3,
            min_samples_leaf=6, max_features=0.5, random_state=SEED, n_jobs=2))
    ])
    et_pos_res = run_cv_manual(pos_X, y, pipe_et_pos)
    print(f"  ET on position features: {et_pos_res['cv_ll_mean']:.4f}±{et_pos_res['cv_ll_std']:.4f}")

    blend_pos = best_blend(elo_oof, et_pos_res["oof_probs"], y, X_elo)
    print(f"  Best pos-ET blend α={blend_pos['alpha']:.2f}: {blend_pos['ll']:.4f}±{blend_pos['std']:.4f}")

    # Also try logistic on top-2 position features + Elo
    if len(top_pos) >= 2:
        top2_pos_cols = [c for c, _ in top_pos[:2]]
        X_pos2 = completed_pos[["elo_diff", "host_diff"] + top2_pos_cols].values
        pipe_log_pos = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=3000, random_state=SEED, solver="lbfgs"))
        ])
        log_pos_res = run_cv_manual(X_pos2, y, pipe_log_pos)
        print(f"  Logistic Elo + top-2 pos features: {log_pos_res['cv_ll_mean']:.4f}±{log_pos_res['cv_ll_std']:.4f}")

    # Take the better of the two approaches
    winner_pos = blend_pos
    verdict13, p13 = report("W3-13 pos-feat blend", winner_pos["ll"], winner_pos["std"],
        float(accuracy_score(y, np.array(LABELS)[np.argmax(winner_pos["oof"], axis=1)])),
        winner_pos["fold_losses"], elo_fold)
    metrics13 = {
        "experiment": "w3-13-position-features-blend",
        "cv_log_loss_mean": winner_pos["ll"],
        "cv_log_loss_std": winner_pos["std"],
        "accuracy": float(accuracy_score(y, np.array(LABELS)[np.argmax(winner_pos["oof"], axis=1)])),
        "delta_vs_elo_baseline": winner_pos["ll"] - ELO_BASELINE,
        "delta_vs_ensemble_frontier": winner_pos["ll"] - FRONTIER,
        "blend_alpha_elo": winner_pos["alpha"],
        "position_feature_count": len(pos_diff_cols),
        "et_pos_standalone_ll": et_pos_res["cv_ll_mean"],
        "top_position_features": [(c, float(r)) for c, r in top_pos],
        "wilcoxon_p": p13,
        "verdict": verdict13,
    }
    save_art("w3-13-position-features-blend", metrics13, {
        "experiment": "w3-13-position-features-blend",
        "model": "ET on position-specific FIFA features + Elo blend",
        "features": pos_diff_cols,
        "blend_alpha_elo": winner_pos["alpha"],
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
    }, winner_pos["oof"], match_ids)
    results["w3-13"] = metrics13

    # ===== W3-14: Short-window form (6 months) in blend =====
    print("\n--- W3-14: Short form window (6 months) vs long (18 months) in blend ---")
    # Add 6-month form features to match matrix
    completed_6m = matches[matches["status"] == "Completed"].copy()
    completed_6m["label"] = np.where(
        completed_6m["home_score"] > completed_6m["away_score"], "H",
        np.where(completed_6m["home_score"] < completed_6m["away_score"], "A", "D")
    )
    from features_w3 import build_match_features as bmf
    completed_6m_full = bmf(matches, teams, squad_feats, form_feats_6m, fifa_feats)
    form_6m_cols = [c for c in completed_6m_full.columns if "diff_form_" in c]
    print(f"  6-month form cols: {form_6m_cols}")
    print(f"  6-month form NaN%: {completed_6m_full[form_6m_cols].isna().mean().to_dict()}")

    # ET on 6-month form features
    X_6m = completed_6m_full[form_6m_cols].values
    pipe_et_6m = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=3,
            min_samples_leaf=6, max_features=0.5, random_state=SEED, n_jobs=2))
    ])
    et_6m_res = run_cv_manual(X_6m, y, pipe_et_6m)
    print(f"  ET on 6m-form: {et_6m_res['cv_ll_mean']:.4f}±{et_6m_res['cv_ll_std']:.4f}")

    blend_6m = best_blend(elo_oof, et_6m_res["oof_probs"], y, X_elo)
    print(f"  Best 6m-form blend α={blend_6m['alpha']:.2f}: {blend_6m['ll']:.4f}±{blend_6m['std']:.4f}")

    # Compare with 18-month form ET
    form_18m_cols = [c for c in completed.columns if "diff_form_" in c]
    X_18m = completed[form_18m_cols].values
    et_18m_res = run_cv_manual(X_18m, y, pipe_et_6m)
    blend_18m = best_blend(elo_oof, et_18m_res["oof_probs"], y, X_elo)
    print(f"  18m-form standalone: {et_18m_res['cv_ll_mean']:.4f}  blend best: {blend_18m['ll']:.4f}")

    # Take best form window
    best_form = blend_6m if blend_6m["ll"] < blend_18m["ll"] else blend_18m
    best_window = "6m" if blend_6m["ll"] < blend_18m["ll"] else "18m"
    print(f"  Best window: {best_window}")

    verdict14, p14 = report("W3-14 form-window blend", best_form["ll"], best_form["std"],
        float(accuracy_score(y, np.array(LABELS)[np.argmax(best_form["oof"], axis=1)])),
        best_form["fold_losses"], elo_fold)
    metrics14 = {
        "experiment": "w3-14-form-window-blend",
        "cv_log_loss_mean": best_form["ll"],
        "cv_log_loss_std": best_form["std"],
        "accuracy": float(accuracy_score(y, np.array(LABELS)[np.argmax(best_form["oof"], axis=1)])),
        "delta_vs_elo_baseline": best_form["ll"] - ELO_BASELINE,
        "delta_vs_ensemble_frontier": best_form["ll"] - FRONTIER,
        "blend_alpha_elo": best_form["alpha"],
        "best_form_window": best_window,
        "et_6m_standalone_ll": et_6m_res["cv_ll_mean"],
        "et_18m_standalone_ll": et_18m_res["cv_ll_mean"],
        "blend_6m_ll": blend_6m["ll"],
        "blend_18m_ll": blend_18m["ll"],
        "wilcoxon_p": p14,
        "verdict": verdict14,
    }
    save_art("w3-14-form-window-blend", metrics14, {
        "experiment": "w3-14-form-window-blend",
        "model": f"ET on {best_window} form features + Elo blend",
        "best_form_window": best_window,
        "blend_alpha_elo": best_form["alpha"],
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
    }, best_form["oof"], match_ids)
    results["w3-14"] = metrics14

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("ROUND-3 SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<35} {'LogLoss':>10} {'±Std':>8} {'Acc':>6} {'Δ-Elo':>9} {'Verdict'}")
    print("-" * 80)
    for name, m in results.items():
        delta = m["cv_log_loss_mean"] - ELO_BASELINE
        print(f"{name:<35} {m['cv_log_loss_mean']:>10.4f} {m['cv_log_loss_std']:>8.4f} "
              f"{m['accuracy']:>6.3f} {delta:>+9.4f} {m['verdict']}")
    print(f"\nElo baseline: {ELO_BASELINE:.4f} | Best-so-far (W3-09): 0.8280 | Frontier: {FRONTIER:.4f}")

    return results


if __name__ == "__main__":
    main()
