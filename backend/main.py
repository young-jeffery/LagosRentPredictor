# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib

# Importing my custom modules
from schemas import PropertyInput, PredictionOutput, TCTBreakdown
from ml_core import execute_prediction

# I am creating a global dictionary to hold my heavy machine learning models in active memory
ml_models = {}

# I am using an asynchronous lifespan context manager. This ensures my server loads the heavy 
# .pkl files into memory BEFORE it starts accepting HTTP requests, preventing crashes.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the files
    try:
        print("Loading Machine Learning artifacts into memory...")
        ml_models["preprocessor"] = joblib.load("feature_preprocessor.pkl")
        ml_models["xgboost"] = joblib.load("xgb_tuned_model.pkl")
        print("Models successfully loaded. Server is ready.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ML files. {e}")
        
    yield # The server runs and handles traffic here
    
    # Shutdown: Clear the memory safely when the server is stopped
    ml_models.clear()
    print("Memory cleared. Server shutting down.")

# Initializing the FastAPI application with my lifespan manager
app = FastAPI(
    title="Lagos Rent Predictor API",
    description="Backend AI engine for residential TCT valuation",
    lifespan=lifespan
)

# I am configuring CORS so my React frontend (which will run on a different port like localhost:3000) 
# is mathematically allowed to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # I will lock this down to my specific frontend URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creating the core POST endpoint that the React app will call
@app.post("/predict", response_model=PredictionOutput)
async def predict_rent(property_details: PropertyInput):
    # First, I need to ensure the models are actually loaded in memory
    if "xgboost" not in ml_models or "preprocessor" not in ml_models:
        raise HTTPException(status_code=503, detail="Machine Learning models are not loaded.")
    
    try:
        # 1. Execute the raw prediction using my ml_core logic
        tct_estimate = execute_prediction(
            model=ml_models["xgboost"],
            preprocessor=ml_models["preprocessor"],
            input_data=property_details
        )
        
        # 2. Implement the Negotiation Range (±15% buffer)
        # This protects the model's credibility and gives users a realistic market window
        negotiation_lower = tct_estimate * 0.85
        negotiation_upper = tct_estimate * 1.15
        
        # 3. Reverse-engineer the TCT Breakdown based on standard Lagos market percentages
        # Standard assumption: Base rent is ~70%, while Agency, Legal, and Caution are ~10% each.
        breakdown = TCTBreakdown(
            base_rent=tct_estimate * 0.70,
            agency_fee=tct_estimate * 0.10,
            legal_fee=tct_estimate * 0.10,
            caution_deposit=tct_estimate * 0.10
        )
        
        # 4. Construct and return the final JSON payload
        return PredictionOutput(
            estimated_tct=tct_estimate,
            negotiation_lower=negotiation_lower,
            negotiation_upper=negotiation_upper,
            breakdown=breakdown
        )
        
    except Exception as e:
        # If anything breaks during the math, I return a clean 500 error to the frontend
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")