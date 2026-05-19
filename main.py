from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
import uvicorn
import config
from app.controllers import health, prediction, comparison, analysis

app = FastAPI(
    title="API de Detección de Ocupación",
    description="Detección de ocupación de habitaciones con modelos de aprendizaje automático",
    version="1.0.0"
)

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(comparison.router)
app.include_router(analysis.router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )