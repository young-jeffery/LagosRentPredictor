# check_monotonicity_v2.py
#
# Sweeps Bedrooms (1-10) and Bathrooms (1-6) across a sample of real test
# rows, holding all other features fixed per row, and checks whether
# predicted price is non-decreasing as each increases. Reports violations
# specifically inside the 1-6 capped range vs. outside it, so you can see
# whether the frontend cap alone already solves the problem on v2 --
# no need to touch xgb_tuned_model_v2.pkl or its features.

import pandas as pd
import numpy as np
import joblib

MODEL_PATH = 'xgb_tuned_model_v3.pkl'
X_TEST_PATH = 'X_test_v3.csv'
N_SAMPLE_ROWS = 30          # how many real rows to use as base cases
BEDROOM_RANGE = range(1, 11)
BATHROOM_RANGE = range(1, 7)
CAPPED_MAX_BED = 6
CAPPED_MAX_BATH = 6
RANDOM_STATE = 42

def sweep_feature(model, base_row, feature_name, value_range, feature_cols):
    rows = []
    for v in value_range:
        r = base_row.copy()
        r[feature_name] = v
        rows.append(r)
    batch = pd.DataFrame(rows, columns=feature_cols)
    preds_log = model.predict(batch)
    preds = np.expm1(preds_log)
    return list(value_range), preds

def find_violations(values, preds, capped_max):
    violations = []
    for i in range(1, len(preds)):
        if preds[i] < preds[i - 1]:
            in_capped_range = values[i] <= capped_max
            violations.append({
                'from_value': values[i - 1],
                'to_value': values[i],
                'from_price': preds[i - 1],
                'to_price': preds[i],
                'drop': preds[i - 1] - preds[i],
                'inside_cap': in_capped_range
            })
    return violations

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)
    feature_cols = list(X_test.columns)

    sample = X_test.sample(n=min(N_SAMPLE_ROWS, len(X_test)), random_state=RANDOM_STATE)

    bed_violations_in_cap = []
    bed_violations_out_cap = []
    bath_violations_in_cap = []
    bath_violations_out_cap = []

    for idx, base_row in sample.iterrows():
        bed_values, bed_preds = sweep_feature(model, base_row, 'Bedrooms', BEDROOM_RANGE, feature_cols)
        for v in find_violations(bed_values, bed_preds, CAPPED_MAX_BED):
            v['row_index'] = idx
            (bed_violations_in_cap if v['inside_cap'] else bed_violations_out_cap).append(v)

        bath_values, bath_preds = sweep_feature(model, base_row, 'Bathrooms', BATHROOM_RANGE, feature_cols)
        for v in find_violations(bath_values, bath_preds, CAPPED_MAX_BATH):
            v['row_index'] = idx
            (bath_violations_in_cap if v['inside_cap'] else bath_violations_out_cap).append(v)

    print(f"\n=== MONOTONICITY SWEEP: {MODEL_PATH} ({len(sample)} base rows) ===\n")

    def report(name, in_cap, out_cap, cap_val):
        print(f"--- {name} ---")
        print(f"Violations WITHIN 1-{cap_val} (capped range): {len(in_cap)}")
        for v in in_cap[:10]:
            print(f"  row {v['row_index']}: {v['from_value']}->{v['to_value']} "
                  f"₦{v['from_price']:,.0f} -> ₦{v['to_price']:,.0f} (drop ₦{v['drop']:,.0f})")
        print(f"Violations BEYOND cap (informational only): {len(out_cap)}")
        print()

    report("BEDROOMS", bed_violations_in_cap, bed_violations_out_cap, CAPPED_MAX_BED)
    report("BATHROOMS", bath_violations_in_cap, bath_violations_out_cap, CAPPED_MAX_BATH)

    total_in_cap = len(bed_violations_in_cap) + len(bath_violations_in_cap)
    print("=== VERDICT ===")
    if total_in_cap == 0:
        print("No violations inside the capped range on any sampled row.")
    else:
        pct_rows_affected = len(set(v['row_index'] for v in bed_violations_in_cap + bath_violations_in_cap)) / len(sample) * 100
        print(f"{total_in_cap} violations found inside the capped range, affecting ~{pct_rows_affected:.0f}% of sampled rows.")
    
if __name__ == "__main__":
    main()