from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import config

router = APIRouter(tags=["Health"])
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


@router.get("/")
async def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/health")
async def health_check():
    from app.models.registry import registry
    return {
        "status": "healthy",
        "models_loaded": registry.list_models()
    }
