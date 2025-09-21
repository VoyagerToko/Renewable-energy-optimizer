# create_baseline_models.py
"""
Create simple baseline models for renewable energy simulation
when training data is limited. These models use realistic
physical patterns instead of complex ML.
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

class SolarModel:
    """Physics-based solar generation model"""
    
    def __init__(self, max_generation=8000):
        self.max_generation = max_generation
    
    def predict(self, X):
        """
        Predict solar generation based on time of day
        X: array of [hour, minute] or similar time features
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for features in X:
            hour = features[0] if len(features) > 0 else 12
            minute = features[1] if len(features) > 1 else 0
            
            time_decimal = hour + minute / 60.0
            
            # Solar generation (bell curve from 5 AM to 7 PM)
            if 5 <= time_decimal <= 19:
                # Peak at solar noon (12 PM)
                solar_factor = np.sin((time_decimal - 5) / 14 * np.pi) ** 2
                # Add some realistic variation
                solar_factor *= (0.8 + 0.4 * np.sin(2 * np.pi * minute / 60))
                solar = self.max_generation * solar_factor
            else:
                solar = 0
            
            predictions.append(max(0, solar))
        
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - np.mean((y - predictions) ** 2) / np.var(y)

class WindModel:
    """Wind generation model with daily patterns"""
    
    def __init__(self, base_generation=4000, min_generation=1000):
        self.base_generation = base_generation
        self.min_generation = min_generation
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for features in X:
            hour = features[0] if len(features) > 0 else 12
            minute = features[1] if len(features) > 1 else 0
            
            time_decimal = hour + minute / 60.0
            
            # Wind has different patterns - often stronger at night
            # Daily pattern - stronger in evening/night
            daily_factor = 1 + 0.5 * np.sin(2 * np.pi * (time_decimal - 6) / 24)
            
            # Add some turbulence/variation
            variation = 0.9 + 0.2 * np.sin(8 * np.pi * time_decimal / 24)
            
            wind = self.base_generation * daily_factor * variation
            predictions.append(max(self.min_generation, wind))
        
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - np.mean((y - predictions) ** 2) / np.var(y)

class DemandModel:
    """Demand model with realistic daily patterns"""
    
    def __init__(self, base_demand=6000, min_demand=2000):
        self.base_demand = base_demand
        self.min_demand = min_demand
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for features in X:
            hour = features[0] if len(features) > 0 else 12
            minute = features[1] if len(features) > 1 else 0
            
            time_decimal = hour + minute / 60.0
            
            # Typical demand pattern - low at night, peaks in evening
            # Morning ramp-up
            if 6 <= time_decimal <= 10:
                factor = 0.7 + 0.4 * (time_decimal - 6) / 4
            # Daytime steady
            elif 10 <= time_decimal <= 17:
                factor = 1.1 + 0.1 * np.sin(2 * np.pi * (time_decimal - 10) / 7)
            # Evening peak
            elif 17 <= time_decimal <= 22:
                factor = 1.2 + 0.3 * np.sin(np.pi * (time_decimal - 17) / 5)
            # Night low
            else:
                factor = 0.6 + 0.1 * np.sin(2 * np.pi * time_decimal / 24)
            
            demand = self.base_demand * factor
            predictions.append(max(self.min_demand, demand))
        
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - np.mean((y - predictions) ** 2) / np.var(y)

def main():
    """Create and save baseline models"""
    print("🔄 Creating physics-based baseline models...")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Create models
    solar_model = SolarModel()
    wind_model = WindModel()
    demand_model = DemandModel()
    
    # Save models
    joblib.dump(solar_model, "models/solar_baseline_model.pkl")
    joblib.dump(wind_model, "models/wind_baseline_model.pkl")
    joblib.dump(demand_model, "models/demand_baseline_model.pkl")
    
    # Create feature info
    feature_info = {
        'features': ['hour', 'minute'],
        'description': 'Physics-based baseline models using time patterns'
    }
    
    joblib.dump(feature_info, "models/baseline_model_info.pkl")
    
    # Test the models
    print("\n🧪 Testing baseline models...")
    test_times = np.array([
        [6, 0],    # 6:00 AM
        [12, 0],   # 12:00 PM
        [18, 0],   # 6:00 PM
        [22, 0]    # 10:00 PM
    ])
    
    solar_pred = solar_model.predict(test_times)
    wind_pred = wind_model.predict(test_times)
    demand_pred = demand_model.predict(test_times)
    
    print("\n🔮 Sample predictions:")
    for i, (hour, minute) in enumerate(test_times):
        print(f"   {hour:2d}:{minute:02d} - Solar:{solar_pred[i]:6.0f}, Wind:{wind_pred[i]:6.0f}, Demand:{demand_pred[i]:6.0f}")
    
    # Generate full day predictions
    full_day = []
    for hour in range(24):
        for minute in range(0, 60, 10):  # Every 10 minutes
            full_day.append([hour, minute])
    
    full_day = np.array(full_day)
    
    results_df = pd.DataFrame({
        'hour': full_day[:, 0],
        'minute': full_day[:, 1],
        'solar': solar_model.predict(full_day),
        'wind': wind_model.predict(full_day),
        'demand': demand_model.predict(full_day)
    })
    
    results_df.to_csv("models/baseline_predictions_24h.csv", index=False)
    
    print(f"\n✅ Baseline models created and saved!")
    print(f"📁 Files: solar_baseline_model.pkl, wind_baseline_model.pkl, demand_baseline_model.pkl")
    print(f"📈 24-hour predictions saved to baseline_predictions_24h.csv")
    print(f"🎯 These models use realistic physical patterns for simulation")

if __name__ == "__main__":
    main()