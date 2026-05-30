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

router = APIRouter(prefix="/analysis", tags=["Analysis"])
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def create_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@router.get("/", response_class=HTMLResponse)
async def analysis_page(request: Request):
    return templates.TemplateResponse(request, "analysis.html")


@router.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode('utf-8', errors='replace')
        df = read_csv_flexible(text)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Failed to read CSV file: {e}"}, status_code=400)

    try:
        feature_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        df = map_dataframe_columns(df, feature_cols)

        if 'Occupancy' in df.columns:
            target_col = 'Occupancy'
        elif 'occupancy' in df.columns:
            target_col = 'occupancy'
        else:
            target_col = None

        analysis = {
            "columns": list(df.columns),
            "rows": len(df),
            "features": {}
        }

        # Calcular estadísticas de características
        for col in feature_cols:
            analysis["features"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }

        plots = {}

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, col in enumerate(feature_cols):
            axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

        axes[5].axis('off')
        plots['distributions'] = create_plot_to_base64(fig)

        if target_col:
            fig, ax = plt.subplots(figsize=(8, 6))
            occupancy_counts = df[target_col].value_counts()
            ax.pie(occupancy_counts.values, labels=['Not Occupied', 'Occupied'], 
                   autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
            ax.set_title('Occupancy Distribution')
            plots['occupancy_pie'] = create_plot_to_base64(fig)
            analysis["occupancy_count"] = occupancy_counts.to_dict()

        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[feature_cols].corr()
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(feature_cols)))
        ax.set_yticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_yticklabels(feature_cols)
        ax.set_title('Feature Correlation Matrix')
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha='center', va='center', color='black', fontsize=8)
        plt.colorbar(im, ax=ax)
        plots['correlation'] = create_plot_to_base64(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        means = df[feature_cols].mean()
        ax.bar(range(len(feature_cols)), means.values, color='steelblue')
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Means')
        plots['feature_means'] = create_plot_to_base64(fig)

        analysis["plots"] = plots
        return analysis
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
