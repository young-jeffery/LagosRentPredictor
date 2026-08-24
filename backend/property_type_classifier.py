# property_type_classifier.py
#
# Corpus-driven Property_Type classification. A property is fundamentally
# ONE structural type -- unlike amenities, which can coexist, these
# categories are mutually exclusive by construction, so this returns a
# single label per row rather than a set of binary flags.
#
# PRIORITY ORDER IS DELIBERATE AND SAFETY-CRITICAL:
# Semi-detached MUST be checked before Detached. "semi-detached" contains
# "detached" as a literal substring -- checking Detached first would
# silently misclassify every semi-detached listing as Detached, collapsing
# two categories with genuinely different price levels (₦13.05M vs ₦16.4M
# median on validation) into one. Terrace/Duplex are checked after both,
# so a "detached duplex" or "semi-detached duplex" resolves to the
# attachment-type category, not the generic structural one.
#
# Validated on lagos_rent_spatially_capped_v2.csv: all 2982 rows classified
# (no fallback loss), 12 categories, every category statistically
# significant vs. the rest (worst case: Maisonette n=10, p=3.3e-4).

import re

# Order matters. Do not reorder without re-validating substring overlaps.
CLASSIFICATION_PATTERNS = [
    ('Penthouse',       r'penthouse'),
    ('Semi-detached',   r'semi.?detached'),
    ('Detached',        r'detached'),
    ('Terrace',         r'terrace|terraced'),
    ('Duplex',          r'duplex'),
    ('Maisonette',      r'maisonette|mansionette'),
    ('Block of flats',  r'block\s*of\s*flat'),
    ('Bungalow',        r'bungalow'),
    ('Self-contain',    r'self\s*.?contain|selfcon'),
    ('Mini flat',       r'mini\s*.?flat'),
    ('Studio',          r'studio'),
]

FALLBACK_TYPE = 'Flat/Apartment'

ALL_PROPERTY_TYPES = [name for name, _ in CLASSIFICATION_PATTERNS] + [FALLBACK_TYPE]


def classify_property_type(text):
    """Classifies a single text blob into one of ALL_PROPERTY_TYPES.
    Falls back to 'Flat/Apartment' if nothing more specific matches --
    this is the largest category (~50% of listings) and the correct
    default for a plain flat/apartment listing with no other structural
    signal."""
    if text is None or (isinstance(text, float)):
        text = ''
    text = str(text).lower()
    for label, pattern in CLASSIFICATION_PATTERNS:
        if re.search(pattern, text):
            return label
    return FALLBACK_TYPE


def add_property_type_column(df, text_columns):
    """Builds a combined lowercase text blob from text_columns and adds
    a 'Property_Type' column. text_columns should include Top_Icons in
    addition to the usual description fields -- property type is
    sometimes stated there even when absent from the free-text description."""
    combined = df[text_columns[0]].fillna('')
    for col in text_columns[1:]:
        combined = combined + ' ' + df[col].fillna('')
    return combined.str.lower().apply(classify_property_type)