# ml_core.py
import pandas as pd
import numpy as np
from schemas import PropertyInput

def execute_prediction(model, preprocessor, input_data: PropertyInput) -> float:
    # I am converting the Pydantic model into a dictionary, then wrapping it in a list 
    # to create a single-row Pandas DataFrame which my preprocessor expects.
    input_dict = input_data.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # I am transforming the raw text and integers into the exact One-Hot sparse format 
    # that my XGBoost champion model was trained on.
    processed_matrix = preprocessor.transform(input_df)
    
    # The prediction output is currently in logarithmic scale because of my np.log1p transformation
    log_prediction = model.predict(processed_matrix)[0]
    
    # I must reverse the log transformation to get the actual raw Naira value
    raw_naira_prediction = np.expm1(log_prediction)
    
    # I am forcing the output to be a standard Python float so it serializes to JSON correctly
    return float(raw_naira_prediction)