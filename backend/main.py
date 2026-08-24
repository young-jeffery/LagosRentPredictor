# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib

from schemas import PropertyInput, PredictionOutput, TCTBreakdown, RecommendationInput, RecommendationOutput
from ml_core import execute_prediction, generate_recommendations
from location_dictionary import load_location_dictionary
from ml_core import execute_prediction, generate_recommendations, compute_negotiation_range

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Loading Machine Learning artifacts into memory...")
        ml_models["location_encoding"] = joblib.load("location_encoding_map_v3.pkl")
        ml_models["property_type_encoding"] = joblib.load("property_type_encoding_map_v3.pkl")
        ml_models["xgboost"] = joblib.load("xgb_tuned_model_v3.pkl")
        ml_models["negotiation_interval"] = joblib.load("negotiation_interval_map_v1.pkl")
        # [ADDED] Full 335-location candidate space for the recommender.
        ml_models["all_locations"] = load_location_dictionary("Cleaned_Lagos_Dictionary.txt")
        print(f"Loaded {len(ml_models['all_locations'])} locations for recommender.")
        print("Models successfully loaded. Server is ready.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ML files. {e}")

    yield

    ml_models.clear()
    print("Memory cleared. Server shutting down.")

app = FastAPI(
    title="Lagos Rent Predictor API",
    description="Backend AI engine for residential TCT valuation",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionOutput)
async def predict_rent(property_details: PropertyInput):
    if "xgboost" not in ml_models or "location_encoding" not in ml_models or "property_type_encoding" not in ml_models or "negotiation_interval" not in ml_models:
        raise HTTPException(status_code=503, detail="Machine Learning models are not loaded.")

    try:
        tct_estimate = execute_prediction(
            model=ml_models["xgboost"],
            location_encoding_map=ml_models["location_encoding"],
            property_type_encoding_map=ml_models["property_type_encoding"],
            input_data=property_details
        )

        negotiation_lower, negotiation_upper = compute_negotiation_range(
            tct_estimate, ml_models["negotiation_interval"]
        )

        breakdown = TCTBreakdown(
            base_rent=tct_estimate * 0.70,
            agency_fee=tct_estimate * 0.10,
            legal_fee=tct_estimate * 0.10,
            caution_deposit=tct_estimate * 0.10
        )

        return PredictionOutput(
            estimated_tct=tct_estimate,
            negotiation_lower=negotiation_lower,
            negotiation_upper=negotiation_upper,
            breakdown=breakdown
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# [ADDED] Budget-based recommender endpoint
@app.post("/recommend", response_model=RecommendationOutput)
async def recommend_properties(criteria: RecommendationInput):
    if "xgboost" not in ml_models or "location_encoding" not in ml_models or "property_type_encoding" not in ml_models:
        raise HTTPException(status_code=503, detail="Machine Learning models are not loaded.")

    try:
        result = generate_recommendations(
            model=ml_models["xgboost"],
            location_encoding_map=ml_models["location_encoding"],
            property_type_encoding_map=ml_models["property_type_encoding"],
            all_locations=ml_models["all_locations"],
            request=criteria
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")