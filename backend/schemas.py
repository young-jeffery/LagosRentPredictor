# schemas.py
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional
from typing import Literal

PROPERTY_TYPES = [
    "Penthouse", "Semi-detached", "Detached", "Terrace", "Duplex",
    "Maisonette", "Block of flats", "Bungalow", "Self-contain",
    "Mini flat", "Studio", "Flat/Apartment",
]

# [ADDED] Bedroom/bathroom bounds per property type, backed by real
# training-data distributions (see chat history for the analysis run).
# Enforced server-side via the validator below -- this is defense in
# depth, since the frontend already prevents invalid combos in the UI,
# but a direct API call (e.g. via /docs) shouldn't be able to ask for a
# "1-bedroom Detached house" or a "3-bedroom Studio" and get a real
# prediction for a combination that doesn't meaningfully exist.
#
# [CRITICAL] Must stay in sync with PROPERTY_TYPE_BED_BATH_CONFIG in
# frontend/src/constants/propertyTypeConstraints.js.
PROPERTY_TYPE_BED_BATH_CONFIG = {
    "Studio":         {"bed_min": 1, "bed_max": 1, "bed_default": 1, "bath_min": 1, "bath_max": 1, "bath_default": 1},
    "Self-contain":   {"bed_min": 1, "bed_max": 1, "bed_default": 1, "bath_min": 1, "bath_max": 1, "bath_default": 1},
    "Mini flat":      {"bed_min": 1, "bed_max": 2, "bed_default": 1, "bath_min": 1, "bath_max": 2, "bath_default": 1},
    "Flat/Apartment": {"bed_min": 1, "bed_max": 6, "bed_default": 2, "bath_min": 1, "bath_max": 6, "bath_default": 2},
    "Block of flats": {"bed_min": 1, "bed_max": 4, "bed_default": 2, "bath_min": 1, "bath_max": 4, "bath_default": 2},
    "Bungalow":       {"bed_min": 1, "bed_max": 6, "bed_default": 3, "bath_min": 1, "bath_max": 5, "bath_default": 3},
    "Penthouse":      {"bed_min": 2, "bed_max": 6, "bed_default": 3, "bath_min": 2, "bath_max": 6, "bath_default": 3},
    "Detached":       {"bed_min": 2, "bed_max": 6, "bed_default": 4, "bath_min": 2, "bath_max": 6, "bath_default": 4},
    "Duplex":         {"bed_min": 2, "bed_max": 7, "bed_default": 4, "bath_min": 2, "bath_max": 7, "bath_default": 4},
    "Semi-detached":  {"bed_min": 2, "bed_max": 6, "bed_default": 4, "bath_min": 2, "bath_max": 5, "bath_default": 4},
    "Terrace":        {"bed_min": 2, "bed_max": 5, "bed_default": 3, "bath_min": 2, "bath_max": 6, "bath_default": 3},
    "Maisonette":     {"bed_min": 2, "bed_max": 4, "bed_default": 3, "bath_min": 2, "bath_max": 4, "bath_default": 3},
}


class PropertyInput(BaseModel):
    Official_Sub_Location: str = Field(..., example="Ikate")
    Main_District: str = Field(..., example="Lekki")
    Property_Type: str = Field(..., example="Duplex")
    Bedrooms: int = Field(..., example=3)
    Bathrooms: int = Field(..., example=3)
    Is_Serviced: int = Field(..., example=0)
    Is_Furnished: int = Field(..., example=0)

    has_prepaid_meter: int
    has_screeded_walls: int
    has_fitted_kitchen: int
    has_clean_water: int
    has_steady_power: int
    has_ac: int
    has_water_heater: int
    has_internet: int
    is_ensuite: int
    has_bq: int
    has_pop_ceiling: int
    has_wardrobe: int
    has_balcony: int
    is_secured: int
    in_gated_estate: int
    has_cctv: int
    has_pool: int
    has_gym: int
    has_smart_home: int
    has_elevator: int
    has_parking: int
    is_tarred: int
    has_easy_access: int
    prefers_working_class: int
    has_jacuzzi: int
    is_premium_marketed: int
    is_urgent_listing: int
    is_standalone_fenced: int

    # [ADDED] Rejects Bedrooms/Bathrooms combinations that don't realistically
    # exist for the chosen Property_Type (e.g. a 1-bedroom Detached house, a
    # 3-bedroom Studio). Runs automatically on every request, including
    # direct API calls that bypass the frontend's own clamping.
    @model_validator(mode="after")
    def validate_bed_bath_for_property_type(self):
        config = PROPERTY_TYPE_BED_BATH_CONFIG.get(self.Property_Type)
        if config:
            if not (config["bed_min"] <= self.Bedrooms <= config["bed_max"]):
                raise ValueError(
                    f"{self.Property_Type} properties must have between "
                    f"{config['bed_min']} and {config['bed_max']} bedrooms (got {self.Bedrooms})."
                )
            if not (config["bath_min"] <= self.Bathrooms <= config["bath_max"]):
                raise ValueError(
                    f"{self.Property_Type} properties must have between "
                    f"{config['bath_min']} and {config['bath_max']} bathrooms (got {self.Bathrooms})."
                )
        return self


class TCTBreakdown(BaseModel):
    base_rent: float
    agency_fee: float
    legal_fee: float
    caution_deposit: float

class PredictionOutput(BaseModel):
    estimated_tct: float
    negotiation_lower: float
    negotiation_upper: float
    breakdown: TCTBreakdown


# ============================================================
# Budget-based recommender schemas
# ============================================================

class RecommendationInput(BaseModel):
    budget: float = Field(..., example=5000000, description="Maximum acceptable Total Cost to Tenant, in Naira.")
    Official_Sub_Location: Optional[str] = Field(None, example="Ikate")
    Main_District: Optional[str] = Field(None, example="Lekki")
    Property_Type: Optional[str] = Field(None, example="Duplex")
    Bedrooms: Optional[int] = Field(None, example=3)
    Bathrooms: Optional[int] = Field(None, example=3)
    preferred_features: List[str] = Field(default_factory=list)
    # [ADDED] User-selectable ranking. "value" (default) = prestige-per-naira
    # heuristic, "closest_to_budget" = uses as much budget as possible,
    # "cheapest" = lowest price wins.
    sort_by: Literal["value", "closest_to_budget", "cheapest"] = Field("value")    


class RecommendedProperty(BaseModel):
    Official_Sub_Location: str
    Main_District: str
    Property_Type: str
    Bedrooms: int
    Bathrooms: int
    estimated_tct: float
    value_score: float


# [MODIFIED] Restructured from a single opaque message string into a clear
# headline + a list of discrete, actionable tips -- each tip states
# explicitly what to change and what it gets you, rather than jargon like
# "dropping the location filter".
class NoMatchSuggestion(BaseModel):
    headline: str
    tips: List[str]
    cheapest_relaxing_location: Optional[float] = None
    cheapest_relaxing_property_type: Optional[float] = None
    cheapest_overall: Optional[float] = None


class RecommendationOutput(BaseModel):
    matches: List[RecommendedProperty]
    total_candidates_considered: int
    total_matches_before_limit: int
    no_match_suggestion: Optional[NoMatchSuggestion] = None