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
    return templates.TemplateResponse("prediction.html", {"request": request})


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
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
