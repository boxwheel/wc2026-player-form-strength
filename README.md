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

### Rounds 1 & 2 (W3-01 to W3-10)

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

### Round 3 (W3-11 to W3-14) — Ceiling Confirmation

| Experiment | Hypothesis | Standalone LL | Blend LL | α | Verdict |
|---|---|---|---|---|---|
| W3-11 GBM blend | GBM has better calibration than ET | 1.0488 | 0.8280 | 1.00 | GREEN† |
| W3-12 Cal-ET blend | Platt calibration of ET OOF | 0.8879 | **0.8280** | 0.95 | GREEN† |
| W3-13 Position features | Position-specific FIFA ratings | 0.9502 | 0.8280 | 1.00 | GREEN† |
| W3-14 Form window | 6m vs 18m lookback | 0.9382 | 0.8280 | 1.00 | GREEN† |

†GREEN (0.8280 < 0.8332) but 0.8280 equals the averaged Elo OOF — not a genuine model contribution.

## Key Findings

1. **Curse of dimensionality dominates**: With 64 samples, adding player-form features to logistic regression consistently hurts (REDs across W3-01 to W3-05). The 35+ feature space causes over-fitting even with L2 regularization.
2. **Feature correlations are real**: Top features by |corr| with outcome: `diff_fifa_top11_mean_overall` (|r|=0.560), `diff_squad_top11_mv` (0.531), `diff_form_gd_per_match` (0.525). Signal exists but is hard to exploit on n=64.
3. **Blending is the right lever (but hits the Elo ceiling)**: All blend experiments converge to α=1.0 (pure Elo OOF) in Round 3. The 0.8280 value is the Elo OOF measured via averaged-OOF probabilities, which is smoother than per-fold loss. It is not a genuine improvement from player features.
4. **GBM is the wrong tree family for n=64**: Boosting amplifies noise (standalone LL=1.05) vs bagging methods (ET=0.89). ExtraTrees is the best standalone player-form model.
5. **Position disaggregation hurts**: Position-specific FIFA ratings (GK defend, FWD shoot) have weaker correlation (|r|=0.47) than aggregated ratings (0.56) and suffer 45% NaN for GKs.
6. **18m form window beats 6m**: 6-month lookback has more NaN (25%) and weaker signal (standalone LL=0.97 vs 0.94).
7. **n=64 is the hard ceiling**: No feature engineering, model family, calibration, or lookback window can push the blend beyond the Elo OOF of ~0.8280. Structural changes are needed.

## What Wave 4 Should Try

To genuinely beat the 0.7608 Wave-2 frontier:
1. **More data**: Historical WC group stages (1982–2022 ≈ 8 × 48 = +384 rows)
2. **W3 features as Wave-2 meta-features**: Add player-form features on top of the Wave-2 stacked ensemble OOF
3. **Leave-one-tournament-out CV**: Proper historical evaluation
4. **Better player matching**: TF-IDF gives 56.4% coverage; phonetic/alias-DB could reach 80%+

## How to Reproduce

```bash
cd ~/research
# Download data
# (canonical: github.com/boxwheel/wc2026-trees-study for fifa.zip)
pip install pandas numpy scikit-learn scipy

cd wave3-player-form
python3 code/run_experiments.py   # experiments W3-01 to W3-06
python3 code/run_exp2.py          # experiments W3-07 to W3-10
python3 code/run_exp3.py          # experiments W3-11 to W3-14 (ceiling confirmation)
```

## Repository

Code: https://github.com/boxwheel/wc2026-player-form-strength
