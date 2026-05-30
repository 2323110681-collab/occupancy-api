from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from io import StringIO
import config
from app.utils.validators import map_dataframe_columns, read_csv_flexible

router = APIRouter(prefix="/comparison", tags=["Comparison"])
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


class ComparisonRequest(BaseModel):
    temperature: float
    humidity: float
    light: float
    co2: float
    humidity_ratio: float
    model: Optional[str] = None
    models: Optional[List[str]] = None


@router.get("/", response_class=HTMLResponse)
async def comparison_page(request: Request):
    return templates.TemplateResponse(request, "comparison.html", {
        "models": ["xgboost", "decision_tree", "neural_network", "naive_bayes", "logistic_regression"]
    })


@router.post("/predict")
async def compare_predict(request: ComparisonRequest):
    from app.models.registry import registry
    
    features = {
        "temperature": request.temperature,
        "humidity": request.humidity,
        "light": request.light,
        "co2": request.co2,
        "humidity_ratio": request.humidity_ratio
    }
    
    try:
        results = {}
        all_models = registry.get_all()
        if not all_models:
            raise ValueError("No models loaded. Please place your .joblib files in the local models/ folder.")
        
        if request.models:
            for model_name in request.models:
                model = registry.get(model_name)
                results[model_name] = model.predict(features)
        elif request.model:
            model = registry.get(request.model)
            results[request.model] = model.predict(features)
        else:
            for name, model in all_models.items():
                results[name] = model.predict(features)
        
        return results
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/compare_csv")
async def compare_csv(file: UploadFile = File(...)):
    from app.models.registry import registry
    
    try:
        content = await file.read()
        text = content.decode('utf-8', errors='replace')
        df = read_csv_flexible(text)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV file: {e}"}, status_code=400)

    feature_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    try:
        df = map_dataframe_columns(df, feature_cols)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    
    results = {}
    all_models = registry.get_all()
    
    for model_name, model in all_models.items():
        predictions = []
        for _, row in df.iterrows():
            features = {
                "temperature": row['Temperature'],
                "humidity": row['Humidity'],
                "light": row['Light'],
                "co2": row['CO2'],
                "humidity_ratio": row['HumidityRatio']
            }
            pred = model.predict(features)
            predictions.append(pred['prediction'])
        
        df_result = df.copy()
        df_result['Prediction'] = predictions
        results[model_name] = df_result.to_dict(orient='records')
    
    return results