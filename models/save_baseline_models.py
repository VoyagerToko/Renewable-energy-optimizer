"""
Script to create and save baseline models using the proper module structure
This ensures the models can be loaded correctly by the main simulation system
"""
import os
import joblib
import numpy as np
from baseline_models import SolarModel, WindModel, DemandModel

def create_and_save_models():
    """Create and save baseline models"""
    print("🔄 Creating physics-based baseline models...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Create models
    solar_model = SolarModel()
    wind_model = WindModel()
    demand_model = DemandModel()
    
    # Save models using joblib
    joblib.dump(solar_model, "models/solar_baseline_model.pkl")
    joblib.dump(wind_model, "models/wind_baseline_model.pkl")
    joblib.dump(demand_model, "models/demand_baseline_model.pkl")
    
    # Create feature info
    feature_info = {
        'features': ['hour', 'minute'],
        'description': 'Physics-based baseline models using time patterns'
    }
    
    joblib.dump(feature_info, "models/baseline_model_info.pkl")
    
    print("✅ Models saved successfully!")
    
    # Test the models
    print("\n🧪 Testing baseline models...")
    test_times = np.array([
        [6, 0],    # 6:00 AM
        [12, 0],   # 12:00 PM
        [18, 0],   # 6:00 PM
        [22, 0]    # 10:00 PM
    ])
    
    print("\n🔮 Sample predictions:")
    for i, (hour, minute) in enumerate(test_times):
        solar = solar_model.predict(test_times[i:i+1])[0]
        wind = wind_model.predict(test_times[i:i+1])[0]
        demand = demand_model.predict(test_times[i:i+1])[0]
        print(f"   {hour:2d}:{minute:02d} - Solar: {solar:5.0f}, Wind: {wind:5.0f}, Demand: {demand:5.0f}")
    
    # Generate 24-hour predictions for analysis
    hours = np.arange(0, 24, 1)  # Every hour
    minutes = np.zeros_like(hours)
    time_features = np.column_stack([hours, minutes])
    
    solar_24h = solar_model.predict(time_features)
    wind_24h = wind_model.predict(time_features)
    demand_24h = demand_model.predict(time_features)
    
    # Save predictions to CSV for analysis
    import pandas as pd
    predictions_df = pd.DataFrame({
        'hour': hours,
        'solar': solar_24h,
        'wind': wind_24h,  
        'demand': demand_24h
    })
    predictions_df.to_csv('models/baseline_predictions_24h.csv', index=False)
    
    print(f"\n📁 Files created:")
    print(f"   - models/solar_baseline_model.pkl")
    print(f"   - models/wind_baseline_model.pkl") 
    print(f"   - models/demand_baseline_model.pkl")
    print(f"   - models/baseline_model_info.pkl")
    print(f"   - models/baseline_predictions_24h.csv")
    print(f"\n🎯 These models use realistic physical patterns for simulation.")
    print(f"💡 They can be used when ML training data is insufficient.")

if __name__ == "__main__":
    create_and_save_models()