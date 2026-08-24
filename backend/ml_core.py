# ml_core.py
import pandas as pd
import numpy as np
from schemas import PropertyInput, RecommendationInput, PROPERTY_TYPES, PROPERTY_TYPE_BED_BATH_CONFIG

FEATURE_COLUMN_ORDER = [
    'location_encoded',
    'property_type_encoded',
    'has_prepaid_meter', 'has_screeded_walls', 'has_fitted_kitchen',
    'has_clean_water', 'has_steady_power', 'has_ac', 'has_water_heater',
    'has_internet', 'is_ensuite', 'has_bq', 'has_pop_ceiling',
    'has_wardrobe', 'has_balcony', 'is_secured', 'in_gated_estate',
    'has_cctv', 'has_pool', 'has_gym', 'has_smart_home', 'has_elevator',
    'has_parking', 'is_tarred', 'has_easy_access', 'prefers_working_class',
    'has_jacuzzi', 'is_premium_marketed', 'is_urgent_listing',
    'is_standalone_fenced',
    'Bedrooms', 'Bathrooms', 'Is_Serviced', 'Is_Furnished',
]


def encode_location(sub_location: str, main_district: str, encoding_map: dict) -> float:
    if sub_location in encoding_map['sub_map']:
        return encoding_map['sub_map'][sub_location]
    if main_district in encoding_map['district_map']:
        return encoding_map['district_map'][main_district]
    return encoding_map['global_mean']


def encode_property_type(property_type: str, encoding_map: dict) -> float:
    if property_type in encoding_map['type_map']:
        return encoding_map['type_map'][property_type]
    return encoding_map['global_mean']


def execute_prediction(model, location_encoding_map: dict, property_type_encoding_map: dict,
                        input_data: PropertyInput) -> float:
    input_dict = input_data.model_dump()

    location_encoded = encode_location(
        input_dict.pop('Official_Sub_Location'),
        input_dict.pop('Main_District'),
        location_encoding_map
    )
    input_dict['location_encoded'] = location_encoded

    property_type_encoded = encode_property_type(
        input_dict.pop('Property_Type'),
        property_type_encoding_map
    )
    input_dict['property_type_encoded'] = property_type_encoded

    input_df = pd.DataFrame([{col: input_dict[col] for col in FEATURE_COLUMN_ORDER}])

    log_prediction = model.predict(input_df)[0]
    raw_naira_prediction = np.expm1(log_prediction)

    return float(raw_naira_prediction)


def compute_negotiation_range(estimated_tct: float, negotiation_interval_map: dict) -> tuple:
    """
    Looks up the OOF-derived empirical multiplier band (see
    negotiation_interval_analysis_v1.py) for the price bucket the
    prediction falls into, and applies it to get
    (negotiation_lower, negotiation_upper). Predictions outside the
    training price range fall back to the global 10th/90th percentile
    multipliers instead of extrapolating a bucket with no supporting data.
    """
    edges = negotiation_interval_map['bucket_edges']
    lowers = negotiation_interval_map['lower_multipliers']
    uppers = negotiation_interval_map['upper_multipliers']

    if estimated_tct < edges[0] or estimated_tct > edges[-1]:
        lower_mult = negotiation_interval_map['global_lower']
        upper_mult = negotiation_interval_map['global_upper']
    else:
        bucket_idx = np.searchsorted(edges, estimated_tct, side='right') - 1
        bucket_idx = max(0, min(bucket_idx, len(lowers) - 1))
        lower_mult = lowers[bucket_idx]
        upper_mult = uppers[bucket_idx]

    return estimated_tct * lower_mult, estimated_tct * upper_mult


# ============================================================
# Budget-based recommender
# ============================================================

MAX_RESULTS = 50
MAX_PER_SUB_LOCATION = 2
MAX_PER_DISTRICT = 10

# [ADDED] Fallback for any type somehow missing from the config (shouldn't
# happen with all 12 fixed categories, kept for safety).
DEFAULT_BED_BATH_CONFIG = {"bed_min": 1, "bed_max": 6, "bed_default": 2, "bath_min": 1, "bath_max": 6, "bath_default": 2}

ALL_FEATURE_FLAGS = [
    'has_prepaid_meter', 'has_screeded_walls', 'has_fitted_kitchen',
    'has_clean_water', 'has_steady_power', 'has_ac', 'has_water_heater',
    'has_internet', 'is_ensuite', 'has_bq', 'has_pop_ceiling',
    'has_wardrobe', 'has_balcony', 'is_secured', 'in_gated_estate',
    'has_cctv', 'has_pool', 'has_gym', 'has_smart_home', 'has_elevator',
    'has_parking', 'is_tarred', 'has_easy_access', 'prefers_working_class',
    'has_jacuzzi', 'is_premium_marketed', 'is_urgent_listing',
    'is_standalone_fenced', 'Is_Serviced', 'Is_Furnished',
]


def generate_recommendations(model, location_encoding_map: dict, property_type_encoding_map: dict,
                              all_locations: list, request: RecommendationInput) -> dict:
    preferred = set(request.preferred_features)

    rows = []
    for loc in all_locations:
        for ptype in PROPERTY_TYPES:
            # [MODIFIED] Bedrooms/Bathrooms are now resolved PER PROPERTY
            # TYPE, not globally. If the user gave an explicit preference,
            # it's clamped into whatever range is realistic for THIS row's
            # type (e.g. a user asking for 5 bedrooms won't get served a
            # "5-bedroom Studio" candidate -- it clamps to 1). If they left
            # it blank, each type falls back to its own typical default
            # instead of one generic number for every type.
            config = PROPERTY_TYPE_BED_BATH_CONFIG.get(ptype, DEFAULT_BED_BATH_CONFIG)

            if request.Bedrooms is not None:
                beds = min(max(request.Bedrooms, config['bed_min']), config['bed_max'])
            else:
                beds = config['bed_default']

            if request.Bathrooms is not None:
                baths = min(max(request.Bathrooms, config['bath_min']), config['bath_max'])
            else:
                baths = config['bath_default']

            row = {
                'sub_location': loc['sub_location'],
                'district': loc['district'],
                'property_type': ptype,
                'Bedrooms': beds,
                'Bathrooms': baths,
            }
            for flag in ALL_FEATURE_FLAGS:
                row[flag] = 1 if flag in preferred else 0
            rows.append(row)

    df = pd.DataFrame(rows)

    sub_map = location_encoding_map['sub_map']
    district_map = location_encoding_map['district_map']
    global_mean_loc = location_encoding_map['global_mean']

    df['location_encoded'] = df['sub_location'].map(sub_map)
    missing_sub = df['location_encoded'].isna()
    df.loc[missing_sub, 'location_encoded'] = df.loc[missing_sub, 'district'].map(district_map)
    df['location_encoded'] = df['location_encoded'].fillna(global_mean_loc)

    type_map = property_type_encoding_map['type_map']
    global_mean_type = property_type_encoding_map['global_mean']
    df['property_type_encoded'] = df['property_type'].map(type_map).fillna(global_mean_type)

    X = df[FEATURE_COLUMN_ORDER]
    log_preds = model.predict(X)
    df['estimated_tct'] = np.expm1(log_preds)

    df['value_score'] = (df['location_encoded'] + df['property_type_encoded']) / df['estimated_tct']

    if request.Official_Sub_Location:
        location_mask = df['sub_location'] == request.Official_Sub_Location
    elif request.Main_District:
        location_mask = df['district'] == request.Main_District
    else:
        location_mask = pd.Series(True, index=df.index)

    if request.Property_Type:
        type_mask = df['property_type'] == request.Property_Type
    else:
        type_mask = pd.Series(True, index=df.index)

    budget_mask = df['estimated_tct'] <= request.budget

    full_mask = location_mask & type_mask & budget_mask
    if request.sort_by == "closest_to_budget":
        matches_df = df[full_mask].sort_values('estimated_tct', ascending=False)
    elif request.sort_by == "cheapest":
        matches_df = df[full_mask].sort_values('estimated_tct', ascending=True)
    else:  # "value" (default)
        matches_df = df[full_mask].sort_values('value_score', ascending=False)

    if len(matches_df) == 0:
        # [MODIFIED] Clearer, action-first no-match messaging. Each tip
        # states plainly what to change and exactly what it unlocks,
        # instead of "dropping filter X" jargon. A tip is only offered if
        # it actually resolves the budget problem -- no vague partial
        # improvements.
        has_location_filter = bool(request.Official_Sub_Location or request.Main_District)
        has_type_filter = bool(request.Property_Type)

        cheapest_relaxing_location = None
        cheapest_relaxing_type = None

        if has_location_filter:
            relax_location_df = df[type_mask]
            if len(relax_location_df) > 0:
                cheapest_relaxing_location = float(relax_location_df['estimated_tct'].min())

        if has_type_filter:
            relax_type_df = df[location_mask]
            if len(relax_type_df) > 0:
                cheapest_relaxing_type = float(relax_type_df['estimated_tct'].min())

        overall_pool = df[type_mask & location_mask]
        cheapest_overall = float(overall_pool['estimated_tct'].min()) if len(overall_pool) > 0 else float(df['estimated_tct'].min())

        if request.Official_Sub_Location:
            location_desc = request.Official_Sub_Location + (f", {request.Main_District}" if request.Main_District else "")
        elif request.Main_District:
            location_desc = request.Main_District
        else:
            location_desc = None

        headline = f"No properties matched your \u20a6{request.budget:,.0f} budget with your current filters."

        tips = []
        if cheapest_relaxing_location is not None and cheapest_relaxing_location <= request.budget:
            tips.append(
                f"You don't have to search in {location_desc} specifically -- if you're open to other "
                f"locations, options start from \u20a6{cheapest_relaxing_location:,.0f}."
            )
        if cheapest_relaxing_type is not None and cheapest_relaxing_type <= request.budget:
            tips.append(
                f"You don't have to stick with {request.Property_Type} -- if you're open to other property "
                f"types, options start from \u20a6{cheapest_relaxing_type:,.0f}."
            )

        # [FIXED] This must be the price for the CURRENT filters exactly as
        # specified (both location AND property type, if given) -- NOT the
        # cheapest of the relaxed scenarios above. Previously this took
        # min(cheapest_relaxing_location, cheapest_relaxing_type,
        # cheapest_overall), which meant "keep your filters" could quote a
        # price that only worked if you'd actually changed location or type
        # -- exactly backwards from what the tip claimed.
        if not tips:
            tips.append(
                f"Your budget itself is the limiting factor here. Raise it to at least "
                f"\u20a6{cheapest_overall:,.0f} to get a match with your current location and property type."
            )
        else:
            tips.append(
                f"If you'd rather keep your current filters exactly as they are, raise your budget to "
                f"at least \u20a6{cheapest_overall:,.0f} instead."
            )

        return {
            'matches': [],
            'total_candidates_considered': len(df),
            'total_matches_before_limit': 0,
            'no_match_suggestion': {
                'headline': headline,
                'tips': tips,
                'cheapest_relaxing_location': cheapest_relaxing_location,
                'cheapest_relaxing_property_type': cheapest_relaxing_type,
                'cheapest_overall': cheapest_overall,
            }
        }

    selected = []
    skipped = []
    sub_counts = {}
    district_counts = {}

    for _, row in matches_df.iterrows():
        if len(selected) >= MAX_RESULTS:
            break
        sub_count = sub_counts.get(row['sub_location'], 0)
        dist_count = district_counts.get(row['district'], 0)
        if sub_count >= MAX_PER_SUB_LOCATION or dist_count >= MAX_PER_DISTRICT:
            skipped.append(row)
            continue
        selected.append(row)
        sub_counts[row['sub_location']] = sub_count + 1
        district_counts[row['district']] = dist_count + 1

    if len(selected) < MAX_RESULTS and skipped:
        for row in skipped:
            if len(selected) >= MAX_RESULTS:
                break
            selected.append(row)

    matches_output = [
        {
            'Official_Sub_Location': row['sub_location'],
            'Main_District': row['district'],
            'Property_Type': row['property_type'],
            'Bedrooms': int(row['Bedrooms']),
            'Bathrooms': int(row['Bathrooms']),
            'estimated_tct': float(row['estimated_tct']),
            'value_score': float(row['value_score']),
        }
        for row in selected
    ]

    return {
        'matches': matches_output,
        'total_candidates_considered': len(df),
        'total_matches_before_limit': len(matches_df),
        'no_match_suggestion': None,
    }