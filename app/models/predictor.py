import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

FEATURE_ORDER = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


class Predictor:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = joblib.load(path)
        self.supports_proba = hasattr(self.model, "predict_proba")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        features_vector = self._build_feature_vector(features)
        prediction = None
        data_candidate = None

        try:
            df_upper = self._build_feature_dataframe(features, lower=False)
            prediction = self.model.predict(df_upper)
            data_candidate = df_upper
        except Exception:
            try:
                df_lower = self._build_feature_dataframe(features, lower=True)
                prediction = self.model.predict(df_lower)
                data_candidate = df_lower
            except Exception:
                prediction = self.model.predict([features_vector])
                data_candidate = [features_vector]

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
                proba = self.model.predict_proba(data_candidate)
                if hasattr(proba, "shape") and proba.shape[1] > 1:
                    result["probability"] = float(proba[0][1])
                else:
                    result["probability"] = float(proba[0][0])
            except Exception:
                result["probability"] = None

        return result

    def _normalize_key(self, key: str) -> str:
        return ''.join(ch for ch in key.lower() if ch.isalnum())

    def _find_feature_value(self, features: Dict[str, Any], name: str) -> Any:
        normalized_name = self._normalize_key(name)
        if name in features:
            return features[name]
        for key, value in features.items():
            if self._normalize_key(key) == normalized_name:
                return value
        raise ValueError(f"Missing feature: {name}")

    def _build_feature_dataframe(self, features: Dict[str, Any], lower: bool) -> pd.DataFrame:
        row = {}
        for name in FEATURE_ORDER:
            value = self._find_feature_value(features, name)
            key = name.lower() if lower else name
            row[key] = float(value)
        return pd.DataFrame([row])

    def _build_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        vector = []
        for name in FEATURE_ORDER:
            value = self._find_feature_value(features, name)
            vector.append(float(value))
        return vector
