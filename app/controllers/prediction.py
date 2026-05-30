from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import config

router = APIRouter(prefix="/prediction", tags=["Prediction"])
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


class PredictionRequest(BaseModel):
    temperature: float
    humidity: float
    light: float
    co2: float
    humidity_ratio: float


@router.get("/", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse(request, "prediction.html")


@router.get('/models')
async def list_models():
    from app.models.registry import registry
    try:
        models = list(registry.list_models())
        return JSONResponse({"models": models})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/predict")
async def predict(request: PredictionRequest):
    from app.models.registry import registry
    try:
        model = registry.get("xgboost")
        result = model.predict({
            "temperature": request.temperature,
            "humidity": request.humidity,
            "light": request.light,
            "co2": request.co2,
            "humidity_ratio": request.humidity_ratio
        })
        return result
    except ValueError as e:
        # If model not found, return available models to help debugging
        msg = str(e)
        try:
            available = list(registry.list_models())
        except Exception:
            available = []
        return JSONResponse({"error": msg, "available_models": available}, status_code=400)
    except Exception as e:
        msg = str(e)
        # Detect common sklearn deserialization/version mismatch error
        if isinstance(e, AttributeError) and "_sklearn_tags_" in msg:
            hint = (
                "Incompatibilidad detectada entre el modelo serializado y la versión de scikit-learn en tiempo de ejecución. "
                "Solución: fije la versión de scikit-learn usada para desplegar a la misma que la usada al entrenar/serializar los modelos (por ejemplo, add 'scikit-learn==1.4.2' en requirements.txt) "
                "y vuelva a desplegar. También puede reentrenar y serializar los modelos con la versión actual de scikit-learn."
            )
            return JSONResponse({"error": msg, "hint": hint}, status_code=500)

        return JSONResponse({"error": msg}, status_code=500)