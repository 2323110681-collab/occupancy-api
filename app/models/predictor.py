import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

class Predictor:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        self.supports_proba = hasattr(self.model, "predict_proba")
        
        # Detectar automáticamente el orden de características que espera el modelo
        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
            print(f"✅ Modelo {path.name} espera: {self.feature_order}")
        else:
            # Fallback para modelos antiguos
            self.feature_order = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
            print(f"⚠️ Modelo {path.name} no tiene feature_names_in_, usando orden por defecto")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Construir diccionario con nombres correctos (mapeo de nombres)
        normalized_features = {}
        
        # Mapear los nombres que vienen del request a los nombres que espera el modelo
        for key, value in features.items():
            # Buscar coincidencia sin importar mayúsculas/minúsculas
            lower_key = key.lower()
            if lower_key == "temperature":
                normalized_features["Temperature"] = float(value)
            elif lower_key == "humidity":
                normalized_features["Humidity"] = float(value)
            elif lower_key == "light":
                normalized_features["Light"] = float(value)
            elif lower_key == "co2":
                normalized_features["CO2"] = float(value)
            elif lower_key == "humidity_ratio":
                normalized_features["HumidityRatio"] = float(value)
            else:
                normalized_features[key] = float(value)
        
        # Construir el array en el orden que espera el modelo
        try:
            # Intentar con DataFrame (más robusto)
            df_data = []
            for col in self.feature_order:
                if col in normalized_features:
                    df_data.append(normalized_features[col])
                else:
                    # Buscar por nombre sin importar mayúsculas
                    found = False
                    for k, v in normalized_features.items():
                        if k.lower() == col.lower():
                            df_data.append(v)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Característica '{col}' no encontrada en los datos")
            
            df = pd.DataFrame([df_data], columns=self.feature_order)
            prediction = self.model.predict(df)
            
        except Exception as e:
            # Fallback: usar el método antiguo
            print(f"Error con DataFrame, usando método alternativo: {e}")
            features_vector = [normalized_features.get(col, 0) for col in self.feature_order]
            prediction = self.model.predict([features_vector])
        
        if isinstance(prediction, np.ndarray):
            predicted_value = int(prediction[0])
        else:
            predicted_value = int(prediction)

        result = {
            "prediction": predicted_value,
            "occupied": bool(predicted_value == 1),
            "probability": None
        }

        if self.supports_proba:
            try:
                # Reconstruir datos para probabilidad
                df_proba = pd.DataFrame([df_data], columns=self.feature_order)
                proba = self.model.predict_proba(df_proba)
                if hasattr(proba, "shape") and proba.shape[1] > 1:
                    result["probability"] = float(proba[0][1])
                else:
                    result["probability"] = float(proba[0][0])
            except Exception:
                result["probability"] = None

        return result
