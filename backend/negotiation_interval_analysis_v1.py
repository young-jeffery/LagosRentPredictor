# negotiation_interval_analysis_v1.py -- Statistically-derived negotiation
# interval, replacing the flat +-15% band in main.py.
#
# METHODOLOGY: 5-fold out-of-fold (OOF) predictions on the FULL dataset
# (train+test recombined), using the exact hyperparameters of the
# production model (xgb_tuned_model_v3.pkl). Mirrors the OOF discipline
# already used for target encoding in feature_engineering_pipeline_v3.py --
# no row is ever scored by a model trained on it.
#
# For each row: ratio = actual_raw_TCT / predicted_raw_TCT (OOF).
# Rows are bucketed by OOF predicted price into quantile bins, because
# residual spread is NOT constant across the price range (a Studio has a
# much tighter error band than a Detached house). A single global +-X%
# band understates uncertainty for expensive properties and overstates it
# for cheap ones.
#
# Within each bucket, the 10th/90th percentile of the ratio distribution
# becomes that bucket's lower/upper multiplier -- an empirical 80% interval
# (deliberately narrower than a 95% CI: framed as "expect to negotiate
# within this range," not a formal confidence bound).
#
# OUTPUT: negotiation_interval_map_v1.pkl -- loaded at API startup,
# consumed by ml_core.compute_negotiation_range().
#
# RUN ONCE (re-run whenever the production model/features change):
#   python negotiation_interval_analysis_v1.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import joblib

N_FOLDS = 5
N_BUCKETS = 8
LOWER_PCT = 0.20
UPPER_PCT = 0.80
RANDOM_STATE = 42
# [ADDED] Hard ceiling on how far the band can deviate from the point
# estimate, regardless of what the raw empirical quantiles say. Sparse,
# high-variance segments (few luxury listings in the data) can otherwise
# produce a mathematically-correct-but-impractically-wide band. This is a
# deliberate UX/trust decision layered on top of the real statistics, not
# a replacement for them -- document it as such.
MAX_DEVIATION = 0.25

MONOTONE_UP_FEATURES = ['Bedrooms', 'Bathrooms']


def run():
    print("Loading full feature matrix (train+test recombined)...")
    X = pd.concat([pd.read_csv('X_train_v3.csv'), pd.read_csv('X_test_v3.csv')], ignore_index=True)
    y_log = pd.concat(
        [pd.read_csv('y_train_v3.csv').squeeze(), pd.read_csv('y_test_v3.csv').squeeze()],
        ignore_index=True
    )
    print(f"Full dataset: {len(X)} rows, {X.shape[1]} features.")

    print("Extracting production hyperparameters from xgb_tuned_model_v3.pkl...")
    production_model = joblib.load('xgb_tuned_model_v3.pkl')
    params = production_model.get_params()
    # Re-derive monotone_constraints explicitly from the full column set --
    # defensive, in case get_params() round-trips it in a stale form.
    params['monotone_constraints'] = tuple(1 if col in MONOTONE_UP_FEATURES else 0 for col in X.columns)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred_log = np.zeros(len(X))

    print(f"Running {N_FOLDS}-fold OOF prediction with production hyperparameters...")
    for fold_i, (train_idx, holdout_idx) in enumerate(kf.split(X), start=1):
        fold_model = XGBRegressor(**params)
        fold_model.fit(X.iloc[train_idx], y_log.iloc[train_idx])
        oof_pred_log[holdout_idx] = fold_model.predict(X.iloc[holdout_idx])
        print(f"  Fold {fold_i}/{N_FOLDS} done.")

    y_actual_raw = np.expm1(y_log.to_numpy())
    y_pred_raw = np.expm1(oof_pred_log)
    ratio = y_actual_raw / y_pred_raw

    print("\nBucketing by OOF predicted price and computing empirical quantile bands...")
    bucket_labels, bucket_edges = pd.qcut(y_pred_raw, q=N_BUCKETS, retbins=True, duplicates='drop')
    df = pd.DataFrame({
        'y_pred_raw': y_pred_raw,
        'y_actual_raw': y_actual_raw,
        'ratio': ratio,
        'bucket': bucket_labels
    })

    lower_multipliers = []
    upper_multipliers = []
    bucket_summaries = []

    for bucket in df['bucket'].cat.categories:
        sub = df[df['bucket'] == bucket]
        lo_raw = sub['ratio'].quantile(LOWER_PCT)
        hi_raw = sub['ratio'].quantile(UPPER_PCT)
        # [ADDED] Clip to the ±MAX_DEVIATION ceiling around 1.0 (the point
        # estimate). A bucket that's naturally tighter than the cap is left
        # untouched -- the cap only ever narrows, never widens, a band.
        lo = max(lo_raw, 1 - MAX_DEVIATION)
        hi = min(hi_raw, 1 + MAX_DEVIATION)
        lower_multipliers.append(float(lo))
        upper_multipliers.append(float(hi))
        covered = ((sub['ratio'] >= lo) & (sub['ratio'] <= hi)).mean()
        was_capped = (lo_raw < 1 - MAX_DEVIATION) or (hi_raw > 1 + MAX_DEVIATION)
        bucket_summaries.append({
            'range': f"NGN {bucket.left:,.0f} - NGN {bucket.right:,.0f}",
            'n': len(sub),
            'lower_mult': lo,
            'upper_mult': hi,
            'empirical_coverage': covered,
            'was_capped': was_capped,
        })

    global_lower = max(float(df['ratio'].quantile(LOWER_PCT)), 1 - MAX_DEVIATION)
    global_upper = min(float(df['ratio'].quantile(UPPER_PCT)), 1 + MAX_DEVIATION)

    interval_map = {
        'bucket_edges': bucket_edges.tolist(),    # len N_BUCKETS+1, ascending
        'lower_multipliers': lower_multipliers,   # len N_BUCKETS, aligned to edges
        'upper_multipliers': upper_multipliers,
        'global_lower': global_lower,             # fallback for out-of-range predictions
        'global_upper': global_upper,
        'lower_pct': LOWER_PCT,
        'upper_pct': UPPER_PCT,
        'n_folds': N_FOLDS,
    }

    joblib.dump(interval_map, 'negotiation_interval_map_v1.pkl')

    print(f"\n=== NEGOTIATION INTERVAL MAP ({int((UPPER_PCT - LOWER_PCT) * 100)}% empirical, OOF-derived, capped at ±{int(MAX_DEVIATION*100)}%) ===")
    for s in bucket_summaries:
        cap_flag = " [CAPPED]" if s['was_capped'] else ""
        print(f"  {s['range']:<38} n={s['n']:<5} lower x{s['lower_mult']:.3f}  upper x{s['upper_mult']:.3f}  coverage={s['empirical_coverage']:.1%}{cap_flag}")
    print(f"\nGlobal fallback: lower x{global_lower:.3f}  upper x{global_upper:.3f}")
    print(">>> COMPARE AGAINST OLD FLAT BAND: lower x0.850  upper x1.150 <<<")
    print("Saved: negotiation_interval_map_v1.pkl")


if __name__ == "__main__":
    run()