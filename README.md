# Occupancy API

Esta aplicación web sirve para predecir si una habitación está ocupada o no utilizando modelos de aprendizaje automático.

## Qué hace

- Muestra una interfaz en el navegador para ingresar datos de sensores como temperatura, humedad, luz, CO2 y razón de humedad.
- Al enviar los datos, la aplicación usa modelos entrenados (`.joblib`) para determinar la ocupación.
- Permite ver resultados de predicción, comparar modelos y realizar análisis rápidos desde la misma aplicación.

## Cómo funciona

1. `main.py` crea la aplicación FastAPI y registra las rutas.
2. `config.py` define los directorios, carga los archivos de modelo y toma el puerto del entorno.
3. Los controladores en `app/controllers/` manejan las solicitudes del usuario y devuelven las páginas HTML.
4. Los modelos ubicados en `models/` son cargados por la aplicación para hacer predicciones.
5. `app/views/templates/` contiene las plantillas HTML que muestran los formularios y los resultados.

## Estructura principal

- `main.py` - Punto de entrada de la aplicación.
- `config.py` - Configuración general y rutas de los modelos.
- `requirements.txt` - Lista de dependencias Python.
- `build.sh` - Script simple para instalar dependencias.
- `render.yaml` - Configuración de despliegue en Render.
- `app/controllers/` - Rutas de la API y lógica de la aplicación.
- `app/models/` - Carga y registro de los modelos ML.
- `app/utils/` - Validaciones y utilidades.
- `app/views/templates/` - Plantillas HTML para la interfaz.
- `models/` - Archivos de modelos `.joblib` usados para predecir.

## Requisitos

- Python 3.11 o superior
- Dependencias instaladas con:

```bash
pip install -r requirements.txt
```

## Ejecutar localmente

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Abrir en el navegador:

```text
http://localhost:8000/
```

## Despliegue básico

- `render.yaml` está configurado para usar Python y ejecutar `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Render instala dependencias con `pip install -r requirements.txt`.
- `config.py` usa la variable de entorno `PORT`, lo que permite que la aplicación funcione correctamente en la nube.

## Nota

Los modelos `.joblib` deben estar disponibles en la carpeta `models/` para que la predicción funcione correctamente.
