from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from io import StringIO
import base64
import io
import matplotlib
from app.utils.validators import map_dataframe_columns, read_csv_flexible
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

router = APIRouter(prefix="/mass", tags=["Mass Prediction"])
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def create_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@router.get("/", response_class=HTMLResponse)
async def mass_page(request: Request):
    return templates.TemplateResponse(request, "mass_prediction.html")


@router.post("/predict")
async def mass_predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    from app.models.registry import registry
    
    try:
        content1 = await file1.read()
        content2 = await file2.read()
        text1 = content1.decode('utf-8', errors='replace')
        text2 = content2.decode('utf-8', errors='replace')
        df1 = read_csv_flexible(text1)
        df2 = read_csv_flexible(text2)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV file(s): {e}"}, status_code=400)
    
    feature_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    try:
        df1 = map_dataframe_columns(df1, feature_cols)
        df2 = map_dataframe_columns(df2, feature_cols)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    
    results = {
        "dataset1": {},
        "dataset2": {}
    }
    
    all_models = registry.get_all()
    if not all_models:
        return JSONResponse({"error": "No models loaded. Please place your .joblib files in the local models/ folder."}, status_code=500)
    
    for model_name, model in all_models.items():
        preds1 = []
        for _, row in df1.iterrows():
            features = {
                "temperature": row['Temperature'],
                "humidity": row['Humidity'],
                "light": row['Light'],
                "co2": row['CO2'],
                "humidity_ratio": row['HumidityRatio']
            }
            preds1.append(model.predict(features)['prediction'])
        
        preds2 = []
        for _, row in df2.iterrows():
            features = {
                "temperature": row['Temperature'],
                "humidity": row['Humidity'],
                "light": row['Light'],
                "co2": row['CO2'],
                "humidity_ratio": row['HumidityRatio']
            }
            preds2.append(model.predict(features)['prediction'])
        
        results["dataset1"][model_name] = {
            "predictions": preds1,
            "occupied_count": int(sum(preds1)),
            "total": len(preds1)
        }
        results["dataset2"][model_name] = {
            "predictions": preds2,
            "occupied_count": int(sum(preds2)),
            "total": len(preds2)
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models_names = list(all_models.keys())
    ds1_counts = [results["dataset1"][m]["occupied_count"] for m in models_names]
    ds2_counts = [results["dataset2"][m]["occupied_count"] for m in models_names]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    axes[0].bar(x - width/2, ds1_counts, width, label='Dataset 1', color='steelblue')
    axes[0].bar(x + width/2, ds2_counts, width, label='Dataset 2', color='coral')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Predicted Occupied')
    axes[0].set_title('Occupancy Prediction Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_names, rotation=45, ha='right')
    axes[0].legend()
    
    ds1_total = sum(ds1_counts)
    ds2_total = sum(ds2_counts)
    comparison_data = [ds1_total, ds2_total]
    axes[1].bar(['Dataset 1', 'Dataset 2'], comparison_data, color=['steelblue', 'coral'])
    axes[1].set_ylabel('Total Predicted Occupied')
    axes[1].set_title('Total Occupancy Comparison')
    
    plots = {}
    plots['comparison'] = create_plot_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([ds1_total, ds2_total], labels=['Dataset 1', 'Dataset 2'], 
           autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    ax.set_title('Occupancy Distribution Between Datasets')
    plots['pie'] = create_plot_to_base64(fig)
    
    results["plots"] = plots
    
    return results