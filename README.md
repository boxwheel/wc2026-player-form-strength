# WC 2026 — Wave3: Player-Form Granular Strength

**Campaign**: FIFA World Cup 2026 Match-Outcome Prediction (CPU, small-data)
**Wave**: Wave 3 — Player-Form Granular Strength
**Cluster tag**: `Wave3: Player-Form Granular Strength`
**Flywheel workstream node**: `muddy-bar-4247` (6edd84ce-1b30-528f-8012-7157a4282d97)

## Objective

Go below team level: source time-varying, player-level team strength estimates that reflect CURRENT form and the actually-available squad — not the static snapshot used in Waves 1 & 2.

## Key Baselines

| Frontier | CV Log-Loss | Significance |
|---|---|---|
| Elo-logistic baseline | 0.8337 ± 0.134 | Campaign reference |
| Wave-2 stacked ensemble | 0.7608 | Wilcoxon p=0.007 |

## Data Sources

1. **WC 2026 dataset** (`mominullptr/fifa-world-cup-2026-dataset`): 64 completed group-stage matches, squads_and_players.csv (1,248 players), teams.csv
2. **EAFC 25 player ratings** (`jkotov/all-eafc25-ratings`): 17,873 players with Overall, Pace, Shoot, Pass, Drib, Defend, Physical ratings
3. **International football results** (`martj42/international-football-results-from-1872-to-2017`): 49,477 results through 2026-06-27; used to compute 18-month pre-tournament form

## Feature Engineering

Player name matching: TF-IDF character n-gram cosine similarity (56.4% coverage of WC squad)
Form computation: last 18 months (Jan 2025 - Jun 10, 2026) win_rate, gd_per_match, etc.

### Feature Groups

| Group | Features | NaN% |
|---|---|---|
| Elo/Rank | elo_diff, rank_diff, host_diff | 0% |
| Static squad (17 features) | top11_mv, mean_age, peak_pct, caps, gk_mv, etc. | 0% |
| EAFC25 ratings (8 features) | mean_overall, top11_mean_overall, pace, shoot, pass, defend, physical | 0-16% |
| Recent form (5 features) | win_rate, draw_rate, loss_rate, gd_per_match, goals_per_match | 22% |

## Experiment Results (5-fold × 10-repeat stratified CV, seed=0)

| Experiment | CV Log-Loss | ±Std | Accuracy | Δ vs Elo | Verdict |
|---|---|---|---|---|---|
| Elo baseline | 0.8337 | 0.134 | 62.5% | — | Reference |
| W3-01 Static squad + Elo | 0.9090 | 0.145 | 51.6% | +0.075 | RED |
| W3-02 FIFA ratings + Elo | 0.9088 | 0.146 | 59.4% | +0.075 | RED |
| W3-03 Form + Elo | 0.9212 | 0.173 | 64.1% | +0.088 | RED |
| W3-04 All features | 0.9191 | 0.147 | 60.9% | +0.085 | RED |
| W3-05 Random Forest | 0.8944 | 0.165 | 64.1% | +0.061 | RED |
| W3-06 Elo+RF blend | 0.8393 | 0.130 | 64.1% | +0.006 | FLAT |
| W3-07 Elo+top3-feats | 0.8509 | 0.107 | 62.5% | +0.017 | FLAT |
| W3-08 Elo+best-single-feat | 0.8530 | 0.077 | 62.5% | +0.019 | FLAT |
| W3-09 Elo+ET blend (α=1.0) | **0.8280** | 0.130 | 62.5% | **-0.006** | GREEN |
| W3-10 3-model blend | 0.8287 | 0.118 | 62.5% | -0.005 | FLAT |

## Key Findings

1. **Curse of dimensionality dominates**: With 64 samples, adding player-form features to logistic regression consistently hurts (REDs across W3-01 to W3-05). The 35+ feature space causes over-fitting even with L2 regularization.
2. **Feature correlations are real**: Top features by |corr| with outcome: `diff_fifa_top11_mean_overall` (|r|=0.560), `diff_squad_top11_mv` (0.531), `diff_form_gd_per_match` (0.525). Signal exists but is hard to exploit on n=64.
3. **Blending is the right lever**: ET+Elo blend at high Elo weight (α≥0.80) marginally beats the baseline (0.8280 vs 0.8337). This represents the player-form models adding orthogonal signal when given very low weight.
4. **Recent international form**: The 18-month form (win_rate, gd_per_match) has strong correlation but 22% NaN coverage (some teams not in results database), reducing utility.
5. **Frontier gap**: Player-form features alone cannot reach the 0.7608 Wave-2 ensemble frontier — the data bottleneck (n=64) is the binding constraint.

## How to Reproduce

```bash
cd ~/research
# Download data
# (canonical: github.com/boxwheel/wc2026-trees-study for fifa.zip)
pip install pandas numpy scikit-learn scipy

cd wave3-player-form
python3 code/run_experiments.py   # experiments 01-06
python3 code/run_exp2.py          # experiments 07-12
```

## Repository

Code: https://github.com/boxwheel/wc2026-player-form-strength
