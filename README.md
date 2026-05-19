# Occupancy API

Aplicación web para detección de ocupación de habitaciones usando modelos de aprendizaje automático.

## Estructura

- `main.py` - Punto de entrada de la aplicación FastAPI.
- `config.py` - Configuración de rutas, puertos y modelos.
- `requirements.txt` - Dependencias Python.
- `app/` - Código fuente de la aplicación.
  - `controllers/` - Rutas y controladores.
  - `models/` - Carga y registro de modelos ML.
  - `utils/` - Utilidades y validadores.
  - `views/templates/` - Plantillas HTML de la interfaz.
- `models/` - Modelos `.joblib` usados para las predicciones.

## Requisitos

- Python 3.11+ recomendado
- Dependencias en `requirements.txt`

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución local

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Luego abre en el navegador:

```text
http://localhost:8000/
```
