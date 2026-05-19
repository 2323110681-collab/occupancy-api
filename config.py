from pathlib import Path

# Directorio base del proyecto (donde está config.py)
BASE_DIR = Path(__file__).resolve().parent
# Directorio donde están los modelos, relativo a este proyecto
MODELS_DIR = BASE_DIR / "models"

HOST = "0.0.0.0"
PORT = 8000

MODEL_FILES = {
    "xgboost": "modelo_xgboost.joblib",
    "decision_tree": "modelo_decision_tree.joblib",
    "neural_network": "redes_neuronales.joblib",
    "naive_bayes": "modelo_naive_bayes_occupancy.joblib",
    "logistic_regression": "regresion_logistica.joblib"  
}

# Construir rutas de los modelos
MODEL_PATHS = {}
for name, filename in MODEL_FILES.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        MODEL_PATHS[name] = model_path
        print(f"✓ Modelo {name} cargado desde: {model_path}")
    else:
        print(f"⚠️ Advertencia: No se encuentra {filename} en {MODELS_DIR}")
        MODEL_PATHS[name] = model_path

FEATURE_NAMES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

TEMPLATES_DIR = BASE_DIR / "app" / "views" / "templates"