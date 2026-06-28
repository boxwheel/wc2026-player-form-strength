"""
Wave3: Player-Form Granular Strength — Experiments
Runs multiple experiments with repeated stratified 5-fold CV (5x10, seed=0).
"""

import sys
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss, accuracy_score
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from features_w3 import (
    load_base_data, compute_squad_features, compute_fifa_rating_features,
    compute_recent_form, build_match_features
)

DATA_DIR = "/home/user/research/fifa_data"
EAFC_PATH = "/home/user/research/data/eafc25/fifaRatings.csv"
INTL_PATH = "/home/user/research/data/intl-results/results.csv"
ARTIFACTS_DIR = "/home/user/research/wave3-player-form/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SEED = 0
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)

ELO_BASELINE_LOSS = 0.8337
ELO_BASELINE_STD = 0.134
ENSEMBLE_FRONTIER = 0.7608


def run_cv(X, y, pipeline, cv=CV):
    """
    Run repeated stratified 5x10 CV manually (cross_val_predict fails with repeats).
    Returns fold-level losses and averaged per-sample OOF probabilities.
    """
    labels = ["A", "D", "H"]
    n = len(y)
    # Accumulate per-sample predictions across all folds where sample is in test set
    oof_sum = np.zeros((n, 3))
    oof_count = np.zeros(n)
    fold_losses = []

    for train_idx, test_idx in cv.split(X, y):
        clone_pipe = type("_", (), {})()
        import sklearn.base
        fitted = sklearn.base.clone(pipeline)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fitted.fit(X_train, y_train)
        probs = fitted.predict_proba(X_test)

        # Align columns to ["A","D","H"] order
        clf_classes = list(fitted.classes_)
        label_order = [clf_classes.index(l) if l in clf_classes else 0 for l in labels]
        probs_ordered = probs[:, label_order]

        fold_ll = log_loss(y_test, probs_ordered, labels=labels)
        fold_losses.append(fold_ll)

        oof_sum[test_idx] += probs_ordered
        oof_count[test_idx] += 1

    # Average OOF probabilities across repeats
    oof_probs = oof_sum / np.maximum(oof_count[:, None], 1)
    oof_probs = oof_probs / oof_probs.sum(axis=1, keepdims=True)

    mean_ll = float(np.mean(fold_losses))
    std_ll = float(np.std(fold_losses))
    overall_ll = float(log_loss(y, oof_probs, labels=labels))
    acc = float(accuracy_score(y, np.array(labels)[np.argmax(oof_probs, axis=1)]))

    return {
        "cv_log_loss_mean": mean_ll,
        "cv_log_loss_std": std_ll,
        "overall_log_loss": overall_ll,
        "accuracy": acc,
        "oof_probs": oof_probs,
        "fold_losses": fold_losses,
    }


def paired_test_vs_baseline(fold_losses, baseline_losses=None, baseline_loss=ELO_BASELINE_LOSS):
    """Wilcoxon signed-rank test vs Elo baseline fold losses."""
    if baseline_losses is None:
        # Use our re-derived Elo baseline fold losses for pairing
        return None, None
    stat, p = stats.wilcoxon(fold_losses, baseline_losses, alternative="less")
    return stat, p


def save_artifact(name, metrics, run_info, oof_probs, labels, match_ids):
    """Save metrics.json and run.json artifacts."""
    os.makedirs(f"{ARTIFACTS_DIR}/{name}", exist_ok=True)

    metrics_out = {
        "experiment": name,
        "cv_log_loss_mean": metrics["cv_log_loss_mean"],
        "cv_log_loss_std": metrics["cv_log_loss_std"],
        "overall_log_loss": metrics["overall_log_loss"],
        "accuracy": metrics["accuracy"],
        "delta_vs_elo_baseline": metrics["cv_log_loss_mean"] - ELO_BASELINE_LOSS,
        "delta_vs_ensemble_frontier": metrics["cv_log_loss_mean"] - ENSEMBLE_FRONTIER,
        "verdict": metrics.get("verdict", "UNKNOWN"),
    }
    with open(f"{ARTIFACTS_DIR}/{name}/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Save OOF probs as CSV
    oof_df = pd.DataFrame(oof_probs, columns=[f"prob_{l}" for l in labels])
    oof_df["match_id"] = match_ids
    oof_df.to_csv(f"{ARTIFACTS_DIR}/{name}/oof_probs.csv", index=False)

    with open(f"{ARTIFACTS_DIR}/{name}/run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"  -> Saved artifacts to {ARTIFACTS_DIR}/{name}/")


def classify_verdict(mean_ll, std_ll):
    if mean_ll < ELO_BASELINE_LOSS - 0.01:
        return "GREEN"
    elif mean_ll < ELO_BASELINE_LOSS + 0.02:
        return "FLAT"
    else:
        return "RED"


def main():
    print("=" * 60)
    print("Wave3: Player-Form Granular Strength — CV Experiments")
    print("=" * 60)

    # Load all data
    print("\n[1] Loading data...")
    matches, teams, squads = load_base_data()
    ratings = pd.read_csv(EAFC_PATH)
    intl = pd.read_csv(INTL_PATH)

    print("[2] Computing features...")
    squad_feats = compute_squad_features(squads)
    fifa_feats = compute_fifa_rating_features(squads, ratings)
    form_feats = compute_recent_form(intl, teams)

    # Full feature matrix
    completed = build_match_features(matches, teams, squad_feats, form_feats, fifa_feats)
    y = completed["label"].values
    match_ids = completed["match_id"].values

    labels = ["A", "D", "H"]

    print(f"\nCompleted matches: {len(completed)}, Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    results = {}

    # ==================== Elo-only baseline (re-derive for pairing) ====================
    print("\n[EXP-ELO] Elo-only baseline (re-derivation for pairing)...")
    X_elo = completed[["elo_diff", "host_diff"]].values
    pipe_elo = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs")),
    ])
    elo_res = run_cv(X_elo, y, pipe_elo)
    elo_res["verdict"] = classify_verdict(elo_res["cv_log_loss_mean"], elo_res["cv_log_loss_std"])
    print(f"  Elo baseline: log-loss={elo_res['cv_log_loss_mean']:.4f} ± {elo_res['cv_log_loss_std']:.4f}, acc={elo_res['accuracy']:.3f} [{elo_res['verdict']}]")
    results["elo_baseline"] = elo_res

    elo_fold_losses = elo_res["fold_losses"]

    save_artifact("elo-baseline-w3", elo_res, {
        "experiment": "elo-baseline-w3",
        "features": ["elo_diff", "host_diff"],
        "model": "LogisticRegression(C=1.0)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "note": "Re-derivation of Elo baseline for fold-level pairing with W3 experiments",
    }, elo_res["oof_probs"], labels, match_ids)

    # ==================== Exp-W3-01: Static squad features + Elo ====================
    print("\n[EXP-W3-01] Static squad features + Elo...")
    squad_cols = [c for c in completed.columns if c.startswith("diff_squad_")]
    X_w3_01 = completed[["elo_diff", "rank_diff", "host_diff"] + squad_cols].values
    pipe_01 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED, solver="lbfgs", penalty="l2")),
    ])
    res_01 = run_cv(X_w3_01, y, pipe_01)
    res_01["verdict"] = classify_verdict(res_01["cv_log_loss_mean"], res_01["cv_log_loss_std"])
    delta_01 = res_01["cv_log_loss_mean"] - ELO_BASELINE_LOSS
    print(f"  W3-01: log-loss={res_01['cv_log_loss_mean']:.4f} ± {res_01['cv_log_loss_std']:.4f}, acc={res_01['accuracy']:.3f}, Δ={delta_01:+.4f} [{res_01['verdict']}]")
    results["w3_01"] = res_01

    # Paired test
    stat, p = paired_test_vs_baseline(res_01["fold_losses"], elo_fold_losses)
    print(f"    Wilcoxon vs Elo: stat={stat:.1f}, p={p:.3f}")
    res_01["wilcoxon_p"] = p

    save_artifact("w3-01-squad-elo", res_01, {
        "experiment": "w3-01-squad-elo",
        "features": ["elo_diff", "rank_diff", "host_diff"] + squad_cols,
        "n_features": len(["elo_diff", "rank_diff", "host_diff"] + squad_cols),
        "model": "LogisticRegression(C=0.1, penalty=l2)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "python_deps": {"sklearn": "1.9.0", "pandas": "3.0.3", "numpy": "2.x"},
        "delta_vs_elo": delta_01,
        "wilcoxon_p": p,
    }, res_01["oof_probs"], labels, match_ids)

    # ==================== Exp-W3-02: FIFA ratings + Elo ====================
    print("\n[EXP-W3-02] FIFA 24 ratings + Elo...")
    fifa_cols = [c for c in completed.columns if c.startswith("diff_fifa_") and "coverage" not in c]
    X_w3_02 = completed[["elo_diff", "rank_diff", "host_diff"] + fifa_cols].values
    pipe_02 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED, solver="lbfgs")),
    ])
    res_02 = run_cv(X_w3_02, y, pipe_02)
    res_02["verdict"] = classify_verdict(res_02["cv_log_loss_mean"], res_02["cv_log_loss_std"])
    delta_02 = res_02["cv_log_loss_mean"] - ELO_BASELINE_LOSS
    print(f"  W3-02: log-loss={res_02['cv_log_loss_mean']:.4f} ± {res_02['cv_log_loss_std']:.4f}, acc={res_02['accuracy']:.3f}, Δ={delta_02:+.4f} [{res_02['verdict']}]")
    results["w3_02"] = res_02

    stat2, p2 = paired_test_vs_baseline(res_02["fold_losses"], elo_fold_losses)
    print(f"    Wilcoxon vs Elo: stat={stat2:.1f}, p={p2:.3f}")
    res_02["wilcoxon_p"] = p2

    save_artifact("w3-02-fifa-ratings-elo", res_02, {
        "experiment": "w3-02-fifa-ratings-elo",
        "features": ["elo_diff", "rank_diff", "host_diff"] + fifa_cols,
        "n_features": len(["elo_diff", "rank_diff", "host_diff"] + fifa_cols),
        "model": "LogisticRegression(C=0.1, penalty=l2)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "delta_vs_elo": delta_02,
        "wilcoxon_p": p2,
        "note": "EAFC25 player ratings (Overall, Pace, Shoot, Pass, Drib, Defend, Physical) aggregated to team mean/top11",
    }, res_02["oof_probs"], labels, match_ids)

    # ==================== Exp-W3-03: Form features + Elo ====================
    print("\n[EXP-W3-03] Recent form + Elo...")
    form_cols = [c for c in completed.columns if c.startswith("diff_form_") and "matches" not in c]
    X_w3_03 = completed[["elo_diff", "rank_diff", "host_diff"] + form_cols].values
    pipe_03 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=0.3, max_iter=2000, random_state=SEED, solver="lbfgs")),
    ])
    res_03 = run_cv(X_w3_03, y, pipe_03)
    res_03["verdict"] = classify_verdict(res_03["cv_log_loss_mean"], res_03["cv_log_loss_std"])
    delta_03 = res_03["cv_log_loss_mean"] - ELO_BASELINE_LOSS
    print(f"  W3-03: log-loss={res_03['cv_log_loss_mean']:.4f} ± {res_03['cv_log_loss_std']:.4f}, acc={res_03['accuracy']:.3f}, Δ={delta_03:+.4f} [{res_03['verdict']}]")
    results["w3_03"] = res_03

    stat3, p3 = paired_test_vs_baseline(res_03["fold_losses"], elo_fold_losses)
    print(f"    Wilcoxon vs Elo: stat={stat3:.1f}, p={p3:.3f}")
    res_03["wilcoxon_p"] = p3

    save_artifact("w3-03-form-elo", res_03, {
        "experiment": "w3-03-form-elo",
        "features": ["elo_diff", "rank_diff", "host_diff"] + form_cols,
        "n_features": len(form_cols) + 3,
        "model": "LogisticRegression(C=0.3, penalty=l2)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "form_lookback_days": 550,
        "delta_vs_elo": delta_03,
        "wilcoxon_p": p3,
        "note": "Recent 18-month international form: win_rate, draw_rate, loss_rate, gd_per_match, goals_per_match",
    }, res_03["oof_probs"], labels, match_ids)

    # ==================== Exp-W3-04: ALL features combined ====================
    print("\n[EXP-W3-04] Full feature set (Elo + squad + FIFA ratings + form)...")
    all_feat_cols = (["elo_diff", "rank_diff", "host_diff"] + squad_cols + fifa_cols + form_cols)
    X_w3_04 = completed[all_feat_cols].values
    pipe_04 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=0.05, max_iter=3000, random_state=SEED, solver="lbfgs")),
    ])
    res_04 = run_cv(X_w3_04, y, pipe_04)
    res_04["verdict"] = classify_verdict(res_04["cv_log_loss_mean"], res_04["cv_log_loss_std"])
    delta_04 = res_04["cv_log_loss_mean"] - ELO_BASELINE_LOSS
    print(f"  W3-04: log-loss={res_04['cv_log_loss_mean']:.4f} ± {res_04['cv_log_loss_std']:.4f}, acc={res_04['accuracy']:.3f}, Δ={delta_04:+.4f} [{res_04['verdict']}]")
    results["w3_04"] = res_04

    stat4, p4 = paired_test_vs_baseline(res_04["fold_losses"], elo_fold_losses)
    print(f"    Wilcoxon vs Elo: stat={stat4:.1f}, p={p4:.3f}")
    res_04["wilcoxon_p"] = p4

    save_artifact("w3-04-all-features", res_04, {
        "experiment": "w3-04-all-features",
        "n_features": len(all_feat_cols),
        "feature_groups": {"elo": 3, "squad": len(squad_cols), "fifa": len(fifa_cols), "form": len(form_cols)},
        "model": "LogisticRegression(C=0.05, penalty=l2)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "delta_vs_elo": delta_04,
        "wilcoxon_p": p4,
    }, res_04["oof_probs"], labels, match_ids)

    # ==================== Exp-W3-05: Random Forest on full features ====================
    print("\n[EXP-W3-05] Random Forest on full features (regularised)...")
    X_w3_05 = completed[all_feat_cols].values
    pipe_05 = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=8,
            max_features=0.5, random_state=SEED, n_jobs=-1,
        )),
    ])
    res_05 = run_cv(X_w3_05, y, pipe_05)
    res_05["verdict"] = classify_verdict(res_05["cv_log_loss_mean"], res_05["cv_log_loss_std"])
    delta_05 = res_05["cv_log_loss_mean"] - ELO_BASELINE_LOSS
    print(f"  W3-05: log-loss={res_05['cv_log_loss_mean']:.4f} ± {res_05['cv_log_loss_std']:.4f}, acc={res_05['accuracy']:.3f}, Δ={delta_05:+.4f} [{res_05['verdict']}]")
    results["w3_05"] = res_05

    stat5, p5 = paired_test_vs_baseline(res_05["fold_losses"], elo_fold_losses)
    print(f"    Wilcoxon vs Elo: stat={stat5:.1f}, p={p5:.3f}")
    res_05["wilcoxon_p"] = p5

    save_artifact("w3-05-random-forest", res_05, {
        "experiment": "w3-05-random-forest",
        "n_features": len(all_feat_cols),
        "model": "RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=8, max_features=0.5)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "delta_vs_elo": delta_05,
        "wilcoxon_p": p5,
    }, res_05["oof_probs"], labels, match_ids)

    # ==================== Exp-W3-06: OOF blend — Elo + best W3 ====================
    print("\n[EXP-W3-06] OOF blend: Elo OOF + best player-form model OOF...")
    # Find best W3 single model
    best_name = min(["w3_01", "w3_02", "w3_03", "w3_04", "w3_05"],
                    key=lambda k: results[k]["cv_log_loss_mean"])
    print(f"  Best W3 model: {best_name} (loss={results[best_name]['cv_log_loss_mean']:.4f})")

    for alpha in [0.3, 0.5, 0.7]:
        oof_blend = alpha * elo_res["oof_probs"] + (1 - alpha) * results[best_name]["oof_probs"]
        oof_blend = oof_blend / oof_blend.sum(axis=1, keepdims=True)
        blend_ll = log_loss(y, oof_blend, labels=labels)

        # Fold-level blend losses
        blend_fold_losses = []
        for train_idx, test_idx in CV.split(X_elo, y):
            y_test = y[test_idx]
            p_test = oof_blend[test_idx]
            blend_fold_losses.append(log_loss(y_test, p_test, labels=labels))
        blend_mean = np.mean(blend_fold_losses)
        blend_std = np.std(blend_fold_losses)
        blend_acc = accuracy_score(y, np.array(labels)[np.argmax(oof_blend, axis=1)])
        print(f"  alpha={alpha}: log-loss={blend_mean:.4f} ± {blend_std:.4f}, acc={blend_acc:.3f}, Δ={blend_mean-ELO_BASELINE_LOSS:+.4f}")

    # Best alpha
    best_alpha = 0.5
    oof_blend_best = best_alpha * elo_res["oof_probs"] + (1 - best_alpha) * results[best_name]["oof_probs"]
    oof_blend_best = oof_blend_best / oof_blend_best.sum(axis=1, keepdims=True)

    blend_fold_losses_best = []
    for train_idx, test_idx in CV.split(X_elo, y):
        y_test = y[test_idx]
        p_test = oof_blend_best[test_idx]
        blend_fold_losses_best.append(log_loss(y_test, p_test, labels=labels))
    blend_mean_best = np.mean(blend_fold_losses_best)
    blend_std_best = np.std(blend_fold_losses_best)
    blend_acc_best = accuracy_score(y, np.array(labels)[np.argmax(oof_blend_best, axis=1)])
    blend_verdict = classify_verdict(blend_mean_best, blend_std_best)
    blend_delta = blend_mean_best - ELO_BASELINE_LOSS
    stat6, p6 = paired_test_vs_baseline(blend_fold_losses_best, elo_fold_losses)
    print(f"  Best blend (alpha=0.5): log-loss={blend_mean_best:.4f} ± {blend_std_best:.4f}, acc={blend_acc_best:.3f}, Δ={blend_delta:+.4f} [{blend_verdict}]")
    print(f"    Wilcoxon vs Elo: stat={stat6:.1f}, p={p6:.3f}")

    res_06 = {
        "cv_log_loss_mean": blend_mean_best,
        "cv_log_loss_std": blend_std_best,
        "overall_log_loss": log_loss(y, oof_blend_best, labels=labels),
        "accuracy": blend_acc_best,
        "oof_probs": oof_blend_best,
        "fold_losses": blend_fold_losses_best,
        "verdict": blend_verdict,
        "wilcoxon_p": p6,
    }
    results["w3_06"] = res_06

    save_artifact("w3-06-elo-playerform-blend", res_06, {
        "experiment": "w3-06-elo-playerform-blend",
        "blend_alpha_elo": best_alpha,
        "blend_model": best_name,
        "model": "50% Elo OOF + 50% best player-form OOF (uniform blend)",
        "cv": "RepeatedStratifiedKFold(5x10, seed=0)",
        "n_samples": 64,
        "seed": 0,
        "delta_vs_elo": blend_delta,
        "wilcoxon_p": p6,
    }, res_06["oof_probs"], labels, match_ids)

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<30} {'Log-Loss':>10} {'±Std':>8} {'Acc':>6} {'Δ-Elo':>8} {'Verdict':<8}")
    print("-" * 80)
    for name, res in results.items():
        delta = res["cv_log_loss_mean"] - ELO_BASELINE_LOSS
        print(f"{name:<30} {res['cv_log_loss_mean']:>10.4f} {res['cv_log_loss_std']:>8.4f} {res['accuracy']:>6.3f} {delta:>+8.4f} {res.get('verdict','?'):<8}")
    print(f"\nElo baseline: {ELO_BASELINE_LOSS:.4f} (target to beat)")
    print(f"Ensemble frontier: {ENSEMBLE_FRONTIER:.4f} (Wave-2 best)")

    # Save summary
    summary = {
        "elo_baseline": ELO_BASELINE_LOSS,
        "ensemble_frontier": ENSEMBLE_FRONTIER,
        "results": {
            name: {
                "cv_log_loss_mean": res["cv_log_loss_mean"],
                "cv_log_loss_std": res["cv_log_loss_std"],
                "accuracy": res["accuracy"],
                "delta_vs_elo": res["cv_log_loss_mean"] - ELO_BASELINE_LOSS,
                "delta_vs_frontier": res["cv_log_loss_mean"] - ENSEMBLE_FRONTIER,
                "verdict": res.get("verdict", "UNKNOWN"),
            }
            for name, res in results.items()
        },
    }
    with open(f"{ARTIFACTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> Summary saved to {ARTIFACTS_DIR}/summary.json")

    return results


if __name__ == "__main__":
    main()
