from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from typing import List
import joblib
import torch.nn as nn

app = FastAPI()

class TouristData(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    crowd_density: int
    cluster_id: str
    heatmap_intensity: int | float 
    number_of_tourists_in_cluster: int
    movement_speed_in_cluster: float | int 
    event_type: str
    time_of_day: str

class RiskPredictor(nn.Module):
    def __init__(self, input_size):
        super(RiskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


preprocessor = joblib.load('preprocessor.pkl')
model = RiskPredictor(input_size=13)  # Adjusted to match the trained model's input size (from error log)
model.load_state_dict(torch.load('risk_predictor_model.pth', map_location=torch.device('cpu')))  # CPU for Vercel
model.eval()

@app.get("/")
async def root():
    return {"message": '''FastAPI is running | Powered By Moksh Bhardwaj'''}

@app.post("/predict_risk/")
async def predict_risk(data: List[TouristData]):
    df = pd.DataFrame([item.dict() for item in data])
    print(df.head(2), "data set is loaded")  # Fixed the print statement

    # Extract features for preprocessing
    X = df[['crowd_density', 'heatmap_intensity', 'number_of_tourists_in_cluster', 
            'movement_speed_in_cluster', 'event_type', 'time_of_day']]
    
    # Preprocess
    X_processed = preprocessor.transform(X)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
    
    # Prepare response (matches the provided output format)
    results = []
    for idx, row in df.iterrows():
        pred = predictions[idx]
        # Improved risk alert logic (more granular for hackathon appeal)
        if pred > 0.7:
            risk_alert = f"High risk: Evacuate or monitor closely due to {row['event_type'] if row['event_type'] != 'none' else 'extreme crowd density'}"
        elif pred > 0.4:
            risk_alert = f"Moderate risk: Caution advised due to {row['event_type'] if row['event_type'] != 'none' else 'increasing crowd'}"
        elif pred > 0.2:
            risk_alert = f"Low-moderate risk: Keep an eye on {row['event_type'] if row['event_type'] != 'none' else 'crowd movements'}"
        else:
            risk_alert = "Low risk: Safe area"
        
        result = {
            "cluster_id": row['cluster_id'],
            "predicted_risk_level": float(np.round(pred, 2)),  # Round for readability
            "centroid_location": {"latitude": row['latitude'], "longitude": row['longitude']},
            "risk_alert": risk_alert,
            "timestamp": row['timestamp']
        }
        results.append(result)
    
    return {"predictions": results}