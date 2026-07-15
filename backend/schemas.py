# schemas.py
from pydantic import BaseModel, Field

# I am defining the exact structure I expect from the React frontend. 
# If the frontend sends something missing or with the wrong data type, FastAPI will automatically reject it.
class PropertyInput(BaseModel):
    Consolidated_Location: str = Field(..., example="Yaba")
    Bedrooms: int = Field(..., example=3)
    Bathrooms: int = Field(..., example=3)
    Is_Serviced: int = Field(..., example=0)
    Is_Furnished: int = Field(..., example=0)
    
    # My 24 extracted domain-expert features
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

# I am defining the structure for the breakdown dictionary so my response is strictly typed
class TCTBreakdown(BaseModel):
    base_rent: float
    agency_fee: float
    legal_fee: float
    caution_deposit: float

# This is the final JSON structure my API will send back to the frontend
class PredictionOutput(BaseModel):
    estimated_tct: float
    negotiation_lower: float
    negotiation_upper: float
    breakdown: TCTBreakdown