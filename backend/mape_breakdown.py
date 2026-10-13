# mape_breakdown.py
#
# Reviewer asked for MAPE broken down by stratum instead of one aggregate
# 43.82% figure. This reproduces the exact same test split used to train
# xgb_tuned_model_v3.pkl (same random_state=42 on the same input CSV), so
# we can pull the raw Property_Type label back in for rows that only exist
# as property_type_encoded in X_test_v3.csv, then scores MAPE two ways:
#   (a) grouped by Property_Type
#   (b) grouped by price quantile bucket (same N_BUCKETS=8 pattern as
#       negotiation_interval_analysis_v1.py, for consistency with that
#       section of the paper)
#
# Run from backend/: python mape_breakdown.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

N_BUCKETS = 8

def main():
    model = joblib.load('xgb_tuned_model_v3.pkl')

    X_test = pd.read_csv('X_test_v3.csv')
    y_test = pd.read_csv('y_test_v3.csv').squeeze()

    # Reproduce the same split feature_engineering_pipeline_v3.py used, so
    # we get back the same row indices into the original capped dataset.
    full_df = pd.read_csv('lagos_rent_spatially_capped_v2.csv')
    train_idx, test_idx = train_test_split(full_df.index, test_size=0.2, random_state=42)

    # Property_Type wasn't saved as a column in the capped CSV itself --
    # it's generated fresh by the classifier during feature engineering.
    # Re-run the same classification step to get the label back for the
    # test rows (deterministic regex, so this reproduces identically).
    from property_type_classifier import add_property_type_column
    full_df['Property_Type'] = add_property_type_column(
        full_df, ['Raw_Description', 'Detailed_Attributes', 'Amenities', 'Top_Icons']
    )

    test_property_types = full_df.loc[test_idx, 'Property_Type'].reset_index(drop=True)

    assert len(test_property_types) == len(X_test), (
        f"Row count mismatch: {len(test_property_types)} vs {len(X_test)} -- "
        "the split may not match what feature_engineering_pipeline_v3.py used."
    )

    predictions_log = model.predict(X_test)
    predictions_raw = np.expm1(predictions_log)
    y_test_raw = np.expm1(y_test)

    results = pd.DataFrame({
        'Property_Type': test_property_types,
        'actual': y_test_raw.reset_index(drop=True),
        'predicted': predictions_raw,
    })
    results['abs_pct_error'] = (results['actual'] - results['predicted']).abs() / results['actual']

    # ---- Breakdown (a): by Property_Type ----
    print("=" * 70)
    print("MAPE BY PROPERTY_TYPE")
    print("=" * 70)
    by_type = results.groupby('Property_Type').agg(
        n=('abs_pct_error', 'size'),
        MAPE=('abs_pct_error', lambda x: x.mean() * 100),
    ).sort_values('MAPE')
    print(by_type.to_string(float_format=lambda x: f"{x:.2f}"))
    overall_mape = results['abs_pct_error'].mean() * 100
    print(f"\nOverall MAPE (sanity check, should match paper's 43.82%): {overall_mape:.2f}%")

    # ---- Breakdown (b): by price quantile bucket ----
    print("\n" + "=" * 70)
    print(f"MAPE BY PRICE BUCKET ({N_BUCKETS} quantile buckets on predicted TCT)")
    print("=" * 70)
    results['price_bucket'] = pd.qcut(results['predicted'], q=N_BUCKETS, duplicates='drop')
    by_bucket = results.groupby('price_bucket', observed=True).agg(
        n=('abs_pct_error', 'size'),
        MAPE=('abs_pct_error', lambda x: x.mean() * 100),
        avg_predicted=('predicted', 'mean'),
    )
    print(by_bucket.to_string(float_format=lambda x: f"{x:,.2f}"))

    results.to_csv('mape_breakdown_results.csv', index=False)
    print("\nFull per-row results saved to mape_breakdown_results.csv")


if __name__ == '__main__':
    main()