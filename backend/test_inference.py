import pandas as pd
import numpy as np
import joblib
import warnings

# Hiding warnings to keep my terminal clean during the test
warnings.filterwarnings('ignore')

def test_single_prediction():
    print("--- RUNNING LOCAL INFERENCE TEST ---")
    
    # Loading my exported champion model and the preprocessor pipeline so I don't have to retrain anything
    print("Loading XGBoost model and Preprocessor...")
    preprocessor = joblib.load('feature_preprocessor.pkl')
    model = joblib.load('xgb_tuned_model.pkl')
    
    # Simulating the exact JSON payload my React frontend will eventually send to the FastAPI backend.
    # I am using 'Yaba' as a test, but I can change this to 'Lekki', 'Ikorodu', etc., to see how the price reacts.
    user_input = {
        'Consolidated_Location': 'Ikorodu',
        'Bedrooms': 4,
        'Bathrooms': 4,
        'Is_Serviced': 1,
        'Is_Furnished': 1,
        # All 24 amenities from my regex dictionary need to be passed as 0 (No) or 1 (Yes)
        'has_prepaid_meter': 1, 
        'has_screeded_walls': 1, 
        'has_fitted_kitchen': 1,
        'has_clean_water': 1, 
        'has_steady_power': 1, 
        'has_ac': 1, 
        'has_water_heater': 1,
        'has_internet': 0, 
        'is_ensuite': 1, 
        'has_bq': 1, 
        'has_pop_ceiling': 1,
        'has_wardrobe': 1, 
        'has_balcony': 1, 
        'is_secured': 1, 
        'in_gated_estate': 1,
        'has_cctv': 0, 
        'has_pool': 0, 
        'has_gym': 0, 
        'has_smart_home': 0,
        'has_elevator': 0, 
        'has_parking': 1, 
        'is_tarred': 1, 
        'has_easy_access': 1,
        'prefers_working_class': 0
    }
    
    # Wrapping my dictionary in a list so Pandas can convert it into a 1-row DataFrame 
    input_df = pd.DataFrame([user_input])
    
    print(f"Processing input for: {user_input['Consolidated_Location']}...")
    
    # Transforming the raw DataFrame through my fitted preprocessor.
    # This converts the text location into the dense One-Hot columns that the XGBoost model mathematically expects.
    processed_input = preprocessor.transform(input_df)
    
    # Running the prediction (The raw output will be in log-space)
    predicted_log_price = model.predict(processed_input)[0]
    
    # Reversing the log transformation to get the actual, real-world Naira value
    predicted_naira_price = np.expm1(predicted_log_price)
    
    print("\n=== PREDICTION RESULT ===")
    print(f"Estimated Total Cost of Tenancy (TCT): ₦{predicted_naira_price:,.2f}")
    
    # Calculating the negotiation range using a standard ±15% buffer to reflect the reality of the Lagos market
    lower_bound = predicted_naira_price * 0.85
    upper_bound = predicted_naira_price * 1.15
    print(f"Suggested Negotiation Range: ₦{lower_bound:,.2f} - ₦{upper_bound:,.2f}")
    print("=========================")

if __name__ == "__main__":
    test_single_prediction()