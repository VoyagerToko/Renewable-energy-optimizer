"""
Physics-based baseline models for renewable energy prediction
These models use realistic daily patterns when ML training data is insufficient
"""
import numpy as np

class SolarModel:
    """Solar power generation model based on sun angle and time of day"""
    
    def __init__(self, peak_generation=8000, sunrise_hour=6, sunset_hour=18):
        self.peak_generation = peak_generation
        self.sunrise_hour = sunrise_hour
        self.sunset_hour = sunset_hour
    
    def predict(self, X):
        """
        Predict solar generation based on time features
        X should be array with columns [hour, minute, ...]
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for features in X:
            hour = features[0] if len(features) > 0 else 12
            minute = features[1] if len(features) > 1 else 0
            
            time_decimal = hour + minute / 60.0
            
            # Solar generation only during daylight hours
            if time_decimal < self.sunrise_hour or time_decimal > self.sunset_hour:
                solar_power = 0
            else:
                # Peak at solar noon (around 12:00)
                day_length = self.sunset_hour - self.sunrise_hour
                solar_noon = self.sunrise_hour + day_length / 2
                
                # Sine curve for solar generation throughout the day
                angle = np.pi * (time_decimal - self.sunrise_hour) / day_length
                solar_factor = np.sin(angle)
                
                # Additional boost around solar noon
                noon_boost = 1 + 0.3 * np.exp(-((time_decimal - solar_noon) ** 2) / 4)
                
                solar_power = self.peak_generation * solar_factor * noon_boost
            
            predictions.append(max(0, solar_power))
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate R² score for model evaluation"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


class WindModel:
    """Wind generation model with daily patterns"""
    
    def __init__(self, base_generation=4000, min_generation=1000):
        self.base_generation = base_generation
        self.min_generation = min_generation
    
    def predict(self, X):
        """
        Predict wind generation based on time features
        X should be array with columns [hour, minute, ...]
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for features in X:
            hour = features[0] if len(features) > 0 else 12
            minute = features[1] if len(features) > 1 else 0
            
            time_decimal = hour + minute / 60.0
            
            # Wind patterns - often stronger at night and during weather transitions
            # Daily pattern - stronger in evening/night, calmer in afternoon
            daily_factor = 1 + 0.5 * np.sin(2 * np.pi * (time_decimal - 6) / 24)
            
            # Add some turbulence/variation throughout the day
            variation = 0.9 + 0.2 * np.sin(8 * np.pi * time_decimal / 24)
            
            # Evening wind pickup
            if 16 <= time_decimal <= 20:
                evening_boost = 1.2
            else:
                evening_boost = 1.0
            
            wind = self.base_generation * daily_factor * variation * evening_boost
            predictions.append(max(self.min_generation, wind))
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate R² score for model evaluation"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


class DemandModel:
    """Energy demand model with realistic daily patterns"""
    
    def __init__(self, base_demand=6000, min_demand=2000):
        self.base_demand = base_demand
        self.min_demand = min_demand
    
    def predict(self, X):
        """
        Predict energy demand based on time features
        X should be array with columns [hour, minute, ...]
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for features in X:
            hour = features[0] if len(features) > 0 else 12
            minute = features[1] if len(features) > 1 else 0
            
            time_decimal = hour + minute / 60.0
            
            # Realistic demand patterns throughout the day
            # Morning ramp-up (6-10 AM)
            if 6 <= time_decimal <= 10:
                factor = 0.7 + 0.4 * (time_decimal - 6) / 4
            # Daytime steady with small variations (10 AM - 5 PM)
            elif 10 <= time_decimal <= 17:
                factor = 1.1 + 0.1 * np.sin(2 * np.pi * (time_decimal - 10) / 7)
            # Evening peak (5-10 PM)
            elif 17 <= time_decimal <= 22:
                factor = 1.2 + 0.3 * np.sin(np.pi * (time_decimal - 17) / 5)
            # Night low (10 PM - 6 AM)
            else:
                factor = 0.6 + 0.1 * np.sin(2 * np.pi * time_decimal / 24)
            
            demand = self.base_demand * factor
            predictions.append(max(self.min_demand, demand))
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate R² score for model evaluation"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def create_baseline_models():
    """Create and return baseline models for solar, wind, and demand"""
    return {
        'solar': SolarModel(),
        'wind': WindModel(),
        'demand': DemandModel()
    }


if __name__ == "__main__":
    # Test the models
    import matplotlib.pyplot as plt
    
    print("Testing baseline models...")
    
    # Create test data for 24 hours
    hours = np.arange(0, 24, 0.5)  # Every 30 minutes
    test_data = np.column_stack([hours, np.zeros_like(hours)])  # [hour, 0 minutes]
    
    # Create models
    solar_model = SolarModel()
    wind_model = WindModel()
    demand_model = DemandModel()
    
    # Get predictions
    solar_pred = solar_model.predict(test_data)
    wind_pred = wind_model.predict(test_data)
    demand_pred = demand_model.predict(test_data)
    
    # Print sample results
    print(f"\nSample predictions:")
    for i in [0, 24, 36, 44]:  # 6AM, 12PM, 6PM, 10PM
        h = int(hours[i])
        print(f"  {h:2d}:00 - Solar: {solar_pred[i]:5.0f}, Wind: {wind_pred[i]:5.0f}, Demand: {demand_pred[i]:5.0f}")
    
    print(f"\nDaily totals:")
    print(f"  Solar:  {np.sum(solar_pred):7.0f} kWh")
    print(f"  Wind:   {np.sum(wind_pred):7.0f} kWh") 
    print(f"  Demand: {np.sum(demand_pred):7.0f} kWh")