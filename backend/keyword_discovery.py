# keyword_discovery.py
# Reusable corpus-driven amenity discovery tool. Extracts n-grams from
# listing descriptions and ranks them by their statistical effect on
# log(TCT), so new candidate features can be found from evidence rather
# than guessed. Built-in stoplist automatically excludes location names
# and price-figure fragments, which would otherwise leak the target
# variable back in as a "feature" (e.g. "ikoyi", "15m", "800k").
#
# This does NOT automatically add anything to the model. It produces a
# ranked CSV for human review — a person still decides what's a genuine
# amenity signal vs. noise or leakage before anything gets added to
# feature_engineering_pipeline_v2.py.

import pandas as pd
import numpy as np
import re
from collections import Counter
from scipy import stats

MIN_FREQ = 40          # A term must appear in at least this many listings to be considered
MIN_OTHER_GROUP = 50   # Need at least this many listings WITHOUT the term too, for a fair comparison
P_VALUE_THRESHOLD = 0.01

STOPWORDS = set('''a an the is are of in on for to with and or at by from this that
property flat apartment lagos rent rental annual per year price located
available now new good very nice well spacious excellent house home
suitable located within situated off close near great fantastic superb
call whatsapp contact agent inspection viewing today asap immediately
please kindly note also all fully well maintained clean serene quiet
environment area estate street road avenue close nice self compound
bedroom bedrooms bathroom bathrooms toilet toilets flat duplex bungalow
mini semi detached terrace penthouse studio room rooms sqm sq ft
n m k naira'''.split())


def build_location_stoplist(df):
    """
    Collects every word that appears in any location-related column, so
    location names (which naturally correlate with price via the district
    they belong to) never get mistaken for a genuine text-based amenity
    signal. Rebuilt from the live dataset each run, so it stays current
    even if the location dictionary grows.
    """
    location_words = set()
    for col in ['Cleaned_Location', 'Official_Sub_Location', 'Main_District']:
        if col in df.columns:
            for val in df[col].dropna().unique():
                location_words.update(str(val).lower().split())
    return location_words


def is_price_fragment(term):
    """
    Flags n-grams that are just numbers or number+unit fragments (e.g.
    '15m', '800k', '000 000', '100k') — these are literal price mentions
    in the listing text and would leak the target variable if used as a
    feature.
    """
    words = term.split()
    for w in words:
        if re.fullmatch(r'\d+[km]?', w):
            return True
    return False


def get_ngrams(text, n, extra_stopwords):
    words = [w for w in text.split() if w not in STOPWORDS and w not in extra_stopwords and len(w) > 2]
    return set(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))


def discover_keywords(input_csv, output_csv='keyword_discovery_results.csv'):
    print("--- CORPUS-DRIVEN KEYWORD DISCOVERY ---")
    df = pd.read_csv(input_csv)

    df['Property_Profile'] = (
        df['Raw_Description'].fillna('') + ' ' +
        df['Detailed_Attributes'].fillna('') + ' ' +
        df['Amenities'].fillna('')
    ).str.lower()

    def clean_text(t):
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    df['clean'] = df['Property_Profile'].apply(clean_text)
    df['log_tct'] = np.log1p(df['TCT'])

    location_stoplist = build_location_stoplist(df)
    print(f"Built location stoplist: {len(location_stoplist)} words excluded (district/sub-location names)")

    all_ngrams = Counter()
    for text in df['clean']:
        for n in [1, 2, 3]:
            all_ngrams.update(get_ngrams(text, n, location_stoplist))

    candidates = [term for term, count in all_ngrams.items() if count >= MIN_FREQ]
    candidates = [term for term in candidates if not is_price_fragment(term)]
    print(f"Candidate terms after frequency + leakage filtering: {len(candidates)}")

    results = []
    for term in candidates:
        has_term = df['clean'].str.contains(re.escape(term), regex=True)
        n_with = has_term.sum()
        n_without = (~has_term).sum()
        if n_with < MIN_FREQ or n_without < MIN_OTHER_GROUP:
            continue

        mean_with = df.loc[has_term, 'log_tct'].mean()
        mean_without = df.loc[~has_term, 'log_tct'].mean()
        pct_effect = (np.expm1(mean_with) - np.expm1(mean_without)) / np.expm1(mean_without) * 100
        t_stat, p_val = stats.ttest_ind(
            df.loc[has_term, 'log_tct'], df.loc[~has_term, 'log_tct'], equal_var=False
        )

        results.append({
            'term': term,
            'frequency': n_with,
            'pct_price_effect': round(pct_effect, 1),
            'p_value': p_val
        })

    res_df = pd.DataFrame(results)
    res_df = res_df[res_df['p_value'] < P_VALUE_THRESHOLD]
    res_df = res_df.sort_values('pct_price_effect', ascending=False)

    res_df.to_csv(output_csv, index=False)
    print(f"\nDone. {len(res_df)} statistically significant candidate terms saved to '{output_csv}'.")
    print("IMPORTANT: This is a candidate list for human review, not an automatic feature list.")
    print("Check each term for redundancy with existing features before adding it to")
    print("feature_engineering_pipeline_v2.py.")


if __name__ == "__main__":
    discover_keywords('lagos_rent_spatially_capped_v2.csv')