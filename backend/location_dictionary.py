# location_dictionary.py
#
# Parses Cleaned_Lagos_Dictionary.txt into a list of {sub_location, district}
# dicts -- the full 335-entry candidate space the recommender searches over.
# This is the same source file lagosLocations.js on the frontend was
# generated from, so the two stay in sync as long as both are regenerated
# together when the dictionary changes.

def load_location_dictionary(filepath='Cleaned_Lagos_Dictionary.txt'):
    locations = []
    current_district = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                current_district = stripped[1:-1]
            elif stripped.startswith('- ') and current_district:
                sub_location = stripped[2:].strip()
                locations.append({'sub_location': sub_location, 'district': current_district})

    return locations