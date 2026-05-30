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
        
        # Detectar el orden de características que espera el modelo
        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
            print(f"✅ Modelo {path.name} espera: {self.feature_order}")
        else:
            # Si el modelo no tiene la información, usar orden por defecto
            self.feature_order = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
            print(f"⚠️ Modelo {path.name} no tiene feature_names_in_")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        # Construir el array en el orden exacto que espera el modelo
        feature_values = []
        
        for expected_feature in self.feature_order:
            # Buscar el valor en el diccionario de features
            found = False
            for key, value in features.items():
                # Comparar sin importar mayúsculas/minúsculas
                if key.lower() == expected_feature.lower():
                    feature_values.append(float(value))
                    found = True
                    break
            
            if not found:
                # Si no encuentra, intentar con nombres alternativos
                if expected_feature.lower() == "temperature":
                    feature_values.append(float(features.get("temperature", 0)))
                elif expected_feature.lower() == "humidity":
                    feature_values.append(float(features.get("humidity", 0)))
                elif expected_feature.lower() == "light":
                    feature_values.append(float(features.get("light", 0)))
                elif expected_feature.lower() == "co2":
                    feature_values.append(float(features.get("co2", 0)))
                elif expected_feature.lower() == "humidityratio":
                    feature_values.append(float(features.get("humidity_ratio", 0)))
                else:
                    raise ValueError(f"No se encontró valor para la característica: {expected_feature}")
        
        # Crear array 2D para la predicción
        X = np.array([feature_values])
        
        # Hacer predicción
        try:
            prediction = self.model.predict(X)
        except Exception as e:
            # Intentar con DataFrame como fallback
            df = pd.DataFrame([feature_values], columns=self.feature_order)
            prediction = self.model.predict(df)
        
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
                proba = self.model.predict_proba(X)
                if hasattr(proba, "shape") and proba.shape[1] > 1:
                    result["probability"] = float(proba[0][1])
                else:
                    result["probability"] = float(proba[0][0])
            except Exception:
                result["probability"] = None

        return result
