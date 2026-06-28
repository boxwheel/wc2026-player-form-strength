"""
Wave3: Player-Form — Round 2 Experiments
Focus: targeted feature selection, better C-sweep, deeper ensemble blending.
Key insight from Exp 01-05: curse of dimensionality. We need ≤3 features beyond Elo
that are actually orthogonal. Target: advance past 0.8337 and challenge 0.7608.
"""

import sys, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.inspection import permutation_importance
import sklearn.base
from scipy import stats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from features_w3 import (
    load_base_data, compute_squad_features, compute_fifa_rating_features,
    compute_recent_form, build_match_features
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


def report(name, res, elo_fold_losses):
    delta = res["cv_ll_mean"] - ELO_BASELINE
    stat, p = stats.wilcoxon(res["fold_losses"], elo_fold_losses, alternative="less")
    verdict = "GREEN" if res["cv_ll_mean"] < ELO_BASELINE - 0.005 else ("FLAT" if delta < 0.02 else "RED")
    print(f"  {name}: {res['cv_ll_mean']:.4f}±{res['cv_ll_std']:.4f} acc={res['accuracy']:.3f} Δ={delta:+.4f} [{verdict}] p={p:.3f}")
    res["verdict"] = verdict
    res["wilcoxon_p"] = p
    return res


def save_art(name, res, run_info, match_ids):
    d = f"{ARTIFACTS_DIR}/{name}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/metrics.json", "w") as f:
        json.dump({k: v for k, v in res.items() if k != "oof_probs" and k != "fold_losses"
                   and not hasattr(v, "__iter__") or k in ["cv_ll_mean","cv_ll_std","accuracy","verdict","wilcoxon_p"]}, f, indent=2)
    oof_df = pd.DataFrame(res["oof_probs"], columns=["prob_A","prob_D","prob_H"])
    oof_df["match_id"] = match_ids
    oof_df.to_csv(f"{d}/oof_probs.csv", index=False)
    with open(f"{d}/run.json", "w") as f:
        json.dump(run_info, f, indent=2)


def main():
    print("=" * 70)
    print("Wave3 Round-2: Targeted feature selection & deep blend experiments")
    print("=" * 70)

    matches, teams, squads = load_base_data()
    ratings = pd.read_csv(EAFC_PATH)
    intl = pd.read_csv(INTL_PATH)

    squad_feats = compute_squad_features(squads)
    fifa_feats = compute_fifa_rating_features(squads, ratings)
    form_feats = compute_recent_form(intl, teams)
    completed = build_match_features(matches, teams, squad_feats, form_feats, fifa_feats)
    y = completed["label"].values
    match_ids = completed["match_id"].values

    # Re-derive Elo OOF for pairing
    X_elo = completed[["elo_diff", "host_diff"]].values
    pipe_elo = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs"))])
    elo_res = run_cv_manual(X_elo, y, pipe_elo)
    print(f"\nElo baseline (local): {elo_res['cv_ll_mean']:.4f}")
    elo_fold = elo_res["fold_losses"]
    elo_oof = elo_res["oof_probs"]

    results = {}

    # ===== A: Correlation-based feature selection =====
    # Identify which INDIVIDUAL player features correlate most with outcome
    print("\n--- Correlation Analysis ---")
    y_numeric = np.where(y == "H", 1.0, np.where(y == "A", -1.0, 0.0))
    feat_cols = [c for c in completed.columns if c.startswith("diff_")]
    corrs = {}
    for c in feat_cols:
        col = completed[c]
        if col.notna().sum() < 50:
            continue
        r = np.corrcoef(col.fillna(col.median()), y_numeric)[0, 1]
        corrs[c] = abs(r)

    top_feats = sorted(corrs.items(), key=lambda x: -x[1])[:10]
    print("Top-10 features by |corr| with numeric outcome (H=+1, D=0, A=-1):")
    for c, r in top_feats:
        print(f"  {c}: |r|={r:.3f}")

    # ===== B: Elo + top-3 player features (small C sweep) =====
    print("\n--- Exp-W3-07: Elo + top-3 most correlated player features ---")
    top3_cols = [c for c, _ in top_feats[:3]]
    best_ll = 999
    best_C = 1.0
    best_res_07 = None
    for C in [0.01, 0.05, 0.1, 0.3, 1.0]:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                         ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED, solver="lbfgs"))])
        X = completed[["elo_diff", "host_diff"] + top3_cols].values
        res = run_cv_manual(X, y, pipe)
        if res["cv_ll_mean"] < best_ll:
            best_ll = res["cv_ll_mean"]; best_C = C; best_res_07 = res
        print(f"  C={C}: {res['cv_ll_mean']:.4f}±{res['cv_ll_std']:.4f}")
    res_07 = report("w3_07", best_res_07, elo_fold)
    results["w3_07"] = res_07
    save_art("w3-07-top3-features", res_07, {
        "experiment": "w3-07-top3-features",
        "features": ["elo_diff", "host_diff"] + top3_cols,
        "best_C": best_C,
        "model": f"LogisticRegression(C={best_C})",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
        "selection": "top-3 features by |corr(feature, H/D/A-numeric)| on full dataset",
        "delta_vs_elo": float(best_res_07["cv_ll_mean"] - ELO_BASELINE),
    }, match_ids)

    # ===== C: Elo + single best feature (ablation) =====
    print("\n--- Exp-W3-08: Elo + single best feature ablation ---")
    best_ll_single = 999
    best_feat = top3_cols[0]
    best_res_single = None
    for feat, r in top_feats[:6]:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, max_iter=3000, random_state=SEED, solver="lbfgs"))])
        X = completed[["elo_diff", "host_diff", feat]].values
        res = run_cv_manual(X, y, pipe)
        print(f"  {feat[:35]:35} {res['cv_ll_mean']:.4f}±{res['cv_ll_std']:.4f}")
        if res["cv_ll_mean"] < best_ll_single:
            best_ll_single = res["cv_ll_mean"]; best_feat = feat; best_res_single = res
    res_08 = report("w3_08", best_res_single, elo_fold)
    results["w3_08"] = res_08
    save_art("w3-08-best-single-feature", res_08, {
        "experiment": "w3-08-best-single-feature",
        "best_feature": best_feat,
        "features": ["elo_diff", "host_diff", best_feat],
        "model": "LogisticRegression(C=0.1)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
        "delta_vs_elo": float(best_res_single["cv_ll_mean"] - ELO_BASELINE),
    }, match_ids)

    # ===== D: Deep Elo + RF blend sweep =====
    print("\n--- Exp-W3-09: Deep Elo + RF OOF blend sweep ---")
    # Use the RF OOF from round 1 results if we have them
    # Re-run RF on best features (all features)
    all_feat_cols = [c for c in completed.columns if c.startswith("diff_")]
    X_rf = completed[all_feat_cols].values
    pipe_rf = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("clf", RandomForestClassifier(n_estimators=500, max_depth=3,
                            min_samples_leaf=6, max_features=0.3, random_state=SEED, n_jobs=2))])
    rf_res = run_cv_manual(X_rf, y, pipe_rf)
    print(f"  RF (n=500, d=3, leaf=6): {rf_res['cv_ll_mean']:.4f}±{rf_res['cv_ll_std']:.4f}")

    # ET with different params
    pipe_et = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("clf", ExtraTreesClassifier(n_estimators=500, max_depth=3,
                            min_samples_leaf=6, max_features=0.3, random_state=SEED, n_jobs=2))])
    et_res = run_cv_manual(X_rf, y, pipe_et)
    print(f"  ET (n=500, d=3, leaf=6): {et_res['cv_ll_mean']:.4f}±{et_res['cv_ll_std']:.4f}")

    best_tree_oof = rf_res["oof_probs"] if rf_res["cv_ll_mean"] < et_res["cv_ll_mean"] else et_res["oof_probs"]
    best_tree_loss = min(rf_res["cv_ll_mean"], et_res["cv_ll_mean"])

    # Blend sweep
    best_blend = {"alpha": 0.5, "ll": 999, "oof": None, "fold_losses": None}
    for alpha in np.arange(0.0, 1.05, 0.05):
        blend_oof = alpha * elo_oof + (1 - alpha) * best_tree_oof
        blend_oof = blend_oof / blend_oof.sum(axis=1, keepdims=True)
        # Recompute fold-level losses for this blend
        fold_blend = []
        for train_idx, test_idx in CV.split(X_elo, y):
            fold_blend.append(log_loss(y[test_idx], blend_oof[test_idx], labels=LABELS))
        mean_blend = float(np.mean(fold_blend))
        if mean_blend < best_blend["ll"]:
            best_blend = {"alpha": float(alpha), "ll": mean_blend, "oof": blend_oof, "fold_losses": fold_blend}

    print(f"  Best blend alpha={best_blend['alpha']:.2f}: ll={best_blend['ll']:.4f}")

    # Compare to alpha=0.7 (from round-1 hint)
    for alpha_try in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        blend_oof = alpha_try * elo_oof + (1 - alpha_try) * best_tree_oof
        blend_oof = blend_oof / blend_oof.sum(axis=1, keepdims=True)
        fold_blend = []
        for train_idx, test_idx in CV.split(X_elo, y):
            fold_blend.append(log_loss(y[test_idx], blend_oof[test_idx], labels=LABELS))
        print(f"  alpha={alpha_try:.2f}: {np.mean(fold_blend):.4f}±{np.std(fold_blend):.4f}")

    res_09 = {
        "cv_ll_mean": best_blend["ll"],
        "cv_ll_std": float(np.std(best_blend["fold_losses"])),
        "accuracy": float(accuracy_score(y, np.array(LABELS)[np.argmax(best_blend["oof"], axis=1)])),
        "oof_probs": best_blend["oof"],
        "fold_losses": best_blend["fold_losses"],
    }
    res_09 = report("w3_09", res_09, elo_fold)
    results["w3_09"] = res_09
    save_art("w3-09-elo-tree-blend", res_09, {
        "experiment": "w3-09-elo-tree-blend",
        "blend_alpha_elo": best_blend["alpha"],
        "base_model": "RandomForest or ExtraTrees (best of two)",
        "model_ll": best_tree_loss,
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64, "seed": 0,
        "delta_vs_elo": float(best_blend["ll"] - ELO_BASELINE),
    }, match_ids)

    # ===== E: 3-model blend: Elo + RF + best logistic =====
    print("\n--- Exp-W3-10: 3-model blend: Elo + RF + best logistic player-form ---")
    best_logistic_oof = results.get("w3_08", results.get("w3_07", None))
    if best_logistic_oof is not None:
        logistic_oof = best_logistic_oof["oof_probs"]
        best_3_blend = {"weights": None, "ll": 999, "oof": None, "fold_losses": None}
        for w_elo in [0.5, 0.6, 0.7, 0.8]:
            for w_rf in [0.1, 0.2, 0.3]:
                w_log = 1.0 - w_elo - w_rf
                if w_log < 0:
                    continue
                blend_3 = w_elo * elo_oof + w_rf * best_tree_oof + w_log * logistic_oof
                blend_3 = blend_3 / blend_3.sum(axis=1, keepdims=True)
                fold_3 = []
                for _, test_idx in CV.split(X_elo, y):
                    fold_3.append(log_loss(y[test_idx], blend_3[test_idx], labels=LABELS))
                mean_3 = float(np.mean(fold_3))
                if mean_3 < best_3_blend["ll"]:
                    best_3_blend = {"weights": (w_elo, w_rf, w_log), "ll": mean_3,
                                    "oof": blend_3, "fold_losses": fold_3}

        print(f"  Best 3-blend weights={best_3_blend['weights']}: ll={best_3_blend['ll']:.4f}")
        res_10 = {
            "cv_ll_mean": best_3_blend["ll"],
            "cv_ll_std": float(np.std(best_3_blend["fold_losses"])),
            "accuracy": float(accuracy_score(y, np.array(LABELS)[np.argmax(best_3_blend["oof"], axis=1)])),
            "oof_probs": best_3_blend["oof"],
            "fold_losses": best_3_blend["fold_losses"],
        }
        res_10 = report("w3_10", res_10, elo_fold)
        results["w3_10"] = res_10
        save_art("w3-10-three-model-blend", res_10, {
            "experiment": "w3-10-three-model-blend",
            "weights_elo_rf_logistic": best_3_blend["weights"],
            "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
            "n_samples": 64, "seed": 0,
            "delta_vs_elo": float(best_3_blend["ll"] - ELO_BASELINE),
        }, match_ids)

    # ===== F: Elo + recent form only (focused form experiment) =====
    print("\n--- Exp-W3-11: Recent form features only (no Elo) ---")
    form_cols = [c for c in completed.columns if c.startswith("diff_form_") and "matches" not in c]
    X_form = completed[form_cols].values
    pipe_form = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                          ("clf", LogisticRegression(C=0.3, max_iter=3000, random_state=SEED, solver="lbfgs"))])
    form_only_res = run_cv_manual(X_form, y, pipe_form)
    report("form_only", form_only_res, elo_fold)

    print("\n--- Exp-W3-12: Elo + win_rate diff only (minimal form) ---")
    if "diff_form_win_rate" in completed.columns:
        X_min = completed[["elo_diff", "host_diff", "diff_form_win_rate", "diff_form_gd_per_match"]].values
        for C_try in [0.05, 0.1, 0.3, 1.0]:
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler()),
                             ("clf", LogisticRegression(C=C_try, max_iter=3000, random_state=SEED, solver="lbfgs"))])
            r = run_cv_manual(X_min, y, pipe)
            print(f"  Elo + form_win_rate + gd_per_match C={C_try}: {r['cv_ll_mean']:.4f}±{r['cv_ll_std']:.4f}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("ROUND-2 SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<25} {'LogLoss':>10} {'±Std':>8} {'Acc':>6} {'Δ-Elo':>8} {'Verdict'}")
    print("-" * 75)
    for name, res in results.items():
        delta = res["cv_ll_mean"] - ELO_BASELINE
        print(f"{name:<25} {res['cv_ll_mean']:>10.4f} {res['cv_ll_std']:>8.4f} {res['accuracy']:>6.3f} {delta:>+8.4f} {res['verdict']}")
    print(f"\nElo baseline: {ELO_BASELINE:.4f} | Ensemble frontier: {FRONTIER:.4f}")

    # Update summary file
    summary = json.load(open(f"{ARTIFACTS_DIR}/summary.json"))
    for name, res in results.items():
        summary["results"][name] = {
            "cv_log_loss_mean": res["cv_ll_mean"],
            "cv_log_loss_std": res["cv_ll_std"],
            "accuracy": res["accuracy"],
            "delta_vs_elo": res["cv_ll_mean"] - ELO_BASELINE,
            "delta_vs_frontier": res["cv_ll_mean"] - FRONTIER,
            "verdict": res["verdict"],
        }
    with open(f"{ARTIFACTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> Summary updated at {ARTIFACTS_DIR}/summary.json")

    return results


if __name__ == "__main__":
    main()
