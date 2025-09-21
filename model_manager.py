"""
Model Manager for RenewAI Platform
Handles loading ML models with fallback to physics-based baseline models
"""
import os
import sys
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Add models directory to Python path for imports
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if models_dir not in sys.path:
    sys.path.append(models_dir)

try:
    from models.baseline_models import SolarModel, WindModel, DemandModel
except ImportError:
    # Fallback import structure
    try:
        sys.path.append('models')
        from baseline_models import SolarModel, WindModel, DemandModel
    except ImportError:
        print("Warning: Could not import baseline models")
        SolarModel = WindModel = DemandModel = None

class ModelManager:
    """Manages ML and baseline models for renewable energy prediction"""
    
    def __init__(self, models_path: str = "models"):
        self.models_path = models_path
        self.ml_models = {}
        self.baseline_models = {}
        self.model_info = {}
        self.using_baseline = {"solar": False, "wind": False, "demand": False}
        
        self._load_models()
    
    def _load_models(self, force_ml=True):
        """Load ML models if available, otherwise create baseline models"""
        print("🔄 Loading energy prediction models...")
        
        # Try to load ML models first (now with higher priority)
        ml_model_files = {
            'solar': 'solar_model.pkl',
            'wind': 'wind_model.pkl', 
            'demand': 'demand_model.pkl'
        }
        
        baseline_model_files = {
            'solar': 'solar_baseline_model.pkl',
            'wind': 'wind_baseline_model.pkl',
            'demand': 'demand_baseline_model.pkl'
        }
        
        for model_type in ['solar', 'wind', 'demand']:
            # Try ML model first (prioritized for real models)
            ml_path = os.path.join(self.models_path, "models", ml_model_files[model_type])
            baseline_path = os.path.join(self.models_path, baseline_model_files[model_type])
            
            model_loaded = False
            
            # Try ML model from models/ subdirectory
            if os.path.exists(ml_path):
                try:
                    model = joblib.load(ml_path)
                    # Test the model with dummy data
                    test_data = np.array([[12.0, 0.0, 0.0, 1.0, 0.0, 12, 0, 1.0]])  # Full feature set
                    prediction = model.predict(test_data)
                    
                    if not np.isnan(prediction).any() and not np.isinf(prediction).any() and prediction[0] >= 0:
                        self.ml_models[model_type] = model
                        self.using_baseline[model_type] = False
                        print(f"✅ Loaded REAL ML {model_type} model (RandomForest)")
                        model_loaded = True
                    else:
                        print(f"⚠️  ML {model_type} model produces invalid predictions: {prediction}")
                except Exception as e:
                    print(f"⚠️  Failed to load ML {model_type} model: {str(e)}")
                    # Try alternative feature set
                    try:
                        test_data = np.array([[12, 0]])  # Simple feature set
                        prediction = model.predict(test_data)
                        if not np.isnan(prediction).any() and not np.isinf(prediction).any():
                            self.ml_models[model_type] = model
                            self.using_baseline[model_type] = False
                            print(f"✅ Loaded ML {model_type} model (simple features)")
                            model_loaded = True
                    except:
                        pass
            
            # Try baseline model if ML failed
            if not model_loaded and os.path.exists(baseline_path):
                try:
                    model = joblib.load(baseline_path)
                    # Test the model
                    test_data = np.array([[12, 0]])
                    prediction = model.predict(test_data)
                    
                    if not np.isnan(prediction).any() and not np.isinf(prediction).any():
                        self.baseline_models[model_type] = model
                        self.using_baseline[model_type] = True
                        print(f"✅ Loaded baseline {model_type} model")
                        model_loaded = True
                    else:
                        print(f"⚠️  Baseline {model_type} model produces invalid predictions")
                except Exception as e:
                    print(f"⚠️  Failed to load baseline {model_type} model: {str(e)}")
            
            # Create new baseline model if nothing loaded
            if not model_loaded:
                try:
                    if model_type == 'solar' and SolarModel:
                        model = SolarModel()
                    elif model_type == 'wind' and WindModel:
                        model = WindModel()
                    elif model_type == 'demand' and DemandModel:
                        model = DemandModel()
                    else:
                        raise ImportError("Baseline model classes not available")
                    
                    self.baseline_models[model_type] = model
                    self.using_baseline[model_type] = True
                    print(f"✅ Created new baseline {model_type} model")
                    model_loaded = True
                except Exception as e:
                    print(f"❌ Failed to create baseline {model_type} model: {str(e)}")
            
            if not model_loaded:
                print(f"❌ No working model available for {model_type}")
        
        # Load model info if available
        info_path = os.path.join(self.models_path, "baseline_model_info.pkl")
        if os.path.exists(info_path):
            try:
                self.model_info = joblib.load(info_path)
            except:
                self.model_info = {'features': ['hour', 'minute'], 'description': 'Time-based models'}
        
        print(f"📊 Model status: ML={sum(1 for x in self.using_baseline.values() if not x)}, Baseline={sum(1 for x in self.using_baseline.values() if x)}")
    
    def predict(self, model_type: str, datetime_obj: datetime) -> float:
        """
        Predict solar/wind/demand for a given datetime
        Args:
            model_type: 'solar', 'wind', or 'demand'
            datetime_obj: datetime object for prediction
        Returns:
            Predicted value in kW
        """
        if model_type not in ['solar', 'wind', 'demand']:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Extract features
        hour = datetime_obj.hour
        minute = datetime_obj.minute
        
        # Get prediction from appropriate model
        if self.using_baseline[model_type]:
            # Use baseline model with simple features
            features = np.array([[hour, minute]])
            if model_type in self.baseline_models:
                model = self.baseline_models[model_type]
            else:
                raise ValueError(f"No baseline model available for {model_type}")
        else:
            # Use ML model with full feature set
            if model_type in self.ml_models:
                model = self.ml_models[model_type]
                # Create full feature set for ML models
                time_of_day = hour + minute / 60.0
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                minute_sin = np.sin(2 * np.pi * minute / 60)
                minute_cos = np.cos(2 * np.pi * minute / 60)
                solar_factor = max(0, np.sin(np.pi * (time_of_day - 6) / 12)) if 6 <= time_of_day <= 18 else 0
                
                features = np.array([[time_of_day, hour_sin, hour_cos, minute_sin, minute_cos, 
                                    hour, minute, solar_factor]])
            else:
                raise ValueError(f"No ML model available for {model_type}")
        
        try:
            prediction = model.predict(features)[0]
            return max(0.0, float(prediction))  # Ensure non-negative
        except Exception as e:
            print(f"⚠️  Prediction failed for {model_type}: {str(e)}")
            # Return reasonable fallback values
            fallback_values = {'solar': 2000.0, 'wind': 3000.0, 'demand': 5000.0}
            return fallback_values.get(model_type, 1000.0)
    
    def predict_horizon(self, model_type: str, start_datetime: datetime, horizon_hours: int = 24) -> List[float]:
        """
        Predict values for a time horizon
        Args:
            model_type: 'solar', 'wind', or 'demand'
            start_datetime: starting datetime
            horizon_hours: number of hours to predict
        Returns:
            List of predictions for each hour
        """
        predictions = []
        current_time = start_datetime
        
        for _ in range(horizon_hours):
            prediction = self.predict(model_type, current_time)
            predictions.append(prediction)
            current_time += timedelta(hours=1)
        
        return predictions
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            'ml_models_loaded': list(self.ml_models.keys()),
            'baseline_models_loaded': list(self.baseline_models.keys()),
            'using_baseline': self.using_baseline.copy(),
            'model_info': self.model_info
        }
    
    def force_ml_models(self, enabled: bool = True):
        """Force usage of ML models over baseline models when available"""
        for model_type in ['solar', 'wind', 'demand']:
            if enabled and model_type in self.ml_models:
                self.using_baseline[model_type] = False
                print(f"🔄 Switched to REAL ML model for {model_type}")
            elif not enabled and model_type in self.baseline_models:
                self.using_baseline[model_type] = True
                print(f"🔄 Switched to baseline model for {model_type}")
        
        ml_count = sum(1 for x in self.using_baseline.values() if not x)
        baseline_count = sum(1 for x in self.using_baseline.values() if x)
        print(f"📊 Updated status: REAL ML={ml_count}, Baseline={baseline_count}")

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def predict_renewable_generation(model_type: str, datetime_obj: datetime) -> float:
    """Convenience function for renewable energy prediction"""
    manager = get_model_manager()
    return manager.predict(model_type, datetime_obj)

if __name__ == "__main__":
    # Test the model manager
    from datetime import datetime
    
    manager = ModelManager()
    print("\n" + "="*50)
    print("Testing Model Manager")
    print("="*50)
    
    test_time = datetime(2023, 9, 15, 12, 0)  # Noon
    
    for model_type in ['solar', 'wind', 'demand']:
        try:
            prediction = manager.predict(model_type, test_time)
            status = "baseline" if manager.using_baseline[model_type] else "ML"
            print(f"{model_type.capitalize()} ({status}): {prediction:.0f} kW")
        except Exception as e:
            print(f"{model_type.capitalize()}: Error - {str(e)}")
    
    print(f"\nModel Status: {manager.get_model_status()}")