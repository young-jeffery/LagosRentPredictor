# manual_property_type_validation.py
#
# Builds a manually-tagged validation sample for Property_Type and scores
# property_type_classifier.py's regex classifier against it.
#
# HOW TO RUN:
#   1. Copy this file into your backend/ folder (same folder as
#      property_type_classifier.py and lagos_rent_spatially_capped_v2.csv).
#   2. Run: python manual_property_type_validation.py
#   3. For each listing shown, type the number matching the correct
#      Property_Type, or 's' to skip a row you're unsure about.
#   4. When done (or you quit early with Ctrl+C), it saves your tags and
#      prints accuracy / macro-F1 / per-class report.
#
# You can stop partway through -- it saves progress after every row, and
# running it again will skip rows you've already tagged (reads from
# manual_tags.csv if it exists).

import pandas as pd
import re
import sys

from property_type_classifier import (
    classify_property_type,
    ALL_PROPERTY_TYPES,
)

INPUT_CSV = 'lagos_rent_spatially_capped_v2.csv'
TAGS_CSV = 'manual_tags.csv'          # your manual labels get saved here
SAMPLE_SIZE = 120
RANDOM_SEED = 42


def build_combined_text(row):
    """Mirrors add_property_type_column's text combination logic so what
    you read is exactly what the classifier reads."""
    parts = [
        str(row.get('Raw_Description', '') or ''),
        str(row.get('Detailed_Attributes', '') or ''),
        str(row.get('Amenities', '') or ''),
        str(row.get('Top_Icons', '') or ''),
    ]
    return ' '.join(parts)


def get_sample(df):
    """Stratified-ish sample: guarantees every rare category the
    classifier assigned gets at least a few rows in the sample, on top of
    a random draw, so accuracy isn't computed only on the easy majority
    class (Flat/Apartment)."""
    df = df.copy()
    df['_combined_text'] = df.apply(build_combined_text, axis=1)
    df['_classifier_pred'] = df['_combined_text'].apply(classify_property_type)

    pieces = []
    # Take up to 8 rows from every predicted category (some, like
    # Maisonette, may only have ~10 total -- grab all of them).
    for label in ALL_PROPERTY_TYPES:
        subset = df[df['_classifier_pred'] == label]
        take = min(len(subset), 8)
        if take > 0:
            pieces.append(subset.sample(take, random_state=RANDOM_SEED))

    stratified = pd.concat(pieces).drop_duplicates(subset=['row_id']) \
        if pieces else pd.DataFrame(columns=df.columns)

    # Top up with a random sample from the rest until we hit SAMPLE_SIZE
    remaining_needed = max(0, SAMPLE_SIZE - len(stratified))
    remaining_pool = df.drop(stratified.index, errors='ignore')
    if remaining_needed > 0 and len(remaining_pool) > 0:
        topup = remaining_pool.sample(
            min(remaining_needed, len(remaining_pool)), random_state=RANDOM_SEED
        )
        sample = pd.concat([stratified, topup])
    else:
        sample = stratified

    return sample.sample(frac=1, random_state=RANDOM_SEED)  # shuffle order


def prompt_for_label(row_num, total, text, classifier_guess):
    print(f"\n[{row_num}/{total}]")
    print("-" * 70)
    snippet = text.strip()
    print(snippet[:600] + ("..." if len(snippet) > 600 else ""))
    print("-" * 70)
    print("Classifier guessed:", classifier_guess, "(not shown as a hint before you answer above)")
    print()
    for i, label in enumerate(ALL_PROPERTY_TYPES, 1):
        print(f"  {i}. {label}")
    print("  s. Skip this row")
    print("  q. Save and quit now")

    while True:
        choice = input("Your answer: ").strip().lower()
        if choice == 'q':
            return 'QUIT'
        if choice == 's':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(ALL_PROPERTY_TYPES):
            return ALL_PROPERTY_TYPES[int(choice) - 1]
        print("Invalid input, try again.")


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    df = df.reset_index(drop=True)
    df.index.name = 'row_id'
    df = df.reset_index()

    sample = get_sample(df)

    # Resume support: skip rows already tagged in a previous run
    try:
        existing_tags = pd.read_csv(TAGS_CSV)
        already_done = set(existing_tags['row_id'])
        print(f"Found existing {TAGS_CSV} with {len(already_done)} tagged rows -- resuming.")
    except FileNotFoundError:
        existing_tags = pd.DataFrame(columns=['row_id', 'manual_label', 'classifier_label', 'combined_text'])
        already_done = set()

    todo = sample[~sample['row_id'].isin(already_done)]
    print(f"\n{len(todo)} rows left to tag out of {len(sample)} in the sample.")
    print("Type the number for the correct Property_Type. 's' to skip, 'q' to save+quit.\n")

    new_rows = []
    total = len(todo)
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        answer = prompt_for_label(i, total, row['_combined_text'], row['_classifier_pred'])
        if answer == 'QUIT':
            break
        if answer is None:
            continue
        new_rows.append({
            'row_id': row['row_id'],
            'manual_label': answer,
            'classifier_label': row['_classifier_pred'],
            'combined_text': row['_combined_text'][:300],  # trimmed for readability in the CSV
        })
        # Save progress after every row so nothing is lost if you stop early
        pd.concat([existing_tags, pd.DataFrame(new_rows)], ignore_index=True).to_csv(TAGS_CSV, index=False)

    all_tags = pd.concat([existing_tags, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset='row_id')
    all_tags.to_csv(TAGS_CSV, index=False)
    print(f"\nSaved {len(all_tags)} manually tagged rows to {TAGS_CSV}.")

    if len(all_tags) < 10:
        print("Too few tagged rows to compute meaningful metrics yet. Run the script again to continue tagging.")
        return

    score(all_tags)


def score(all_tags):
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    y_true = all_tags['manual_label']
    y_pred = all_tags['classifier_label']

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 70)
    print("PROPERTY_TYPE CLASSIFIER VALIDATION RESULTS")
    print("=" * 70)
    print(f"Sample size:  {len(all_tags)}")
    print(f"Accuracy:     {acc:.2%}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("=" * 70)
    print(f"\nUse these numbers in Section 3.3/3.4.2 of both papers:")
    print(f'  "Against this manually tagged sample (n = {len(all_tags)}), the classifier')
    print(f'  achieved {acc:.1%} accuracy and a macro-F1 of {macro_f1:.3f}."')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted -- progress up to the last completed row was already saved to", TAGS_CSV)
        sys.exit(0)