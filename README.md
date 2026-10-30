# Mantenimiento predictivo de máquinas (MLOps)

Proyecto final de Data Science G8. Predice si una máquina de producción **va a fallar**
a partir de las lecturas de sus sensores, y lo sirve por una **API (FastAPI)**, una
**interfaz web (Gradio)**, **contenedores (Docker)** y **CI/CD (GitHub Actions + Render)**.

- **Problema:** anticipar fallas para programar mantenimiento antes de que la máquina se detenga.
- **Usuario:** el jefe de planta / técnico de mantenimiento, que ingresa las lecturas y recibe
  la probabilidad de falla.
- **Datos:** [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
  (10 000 máquinas, 3.4 % con falla). Archivo en `data/raw/ai4i2020.csv`.
- **Modelo:** Random Forest dentro de un `Pipeline` de scikit-learn (variables derivadas + one-hot + clasificador).

| | |
|---|---|
| Repositorio | [github.com/edex10ed-source/mantenimiento-mlops](https://github.com/edex10ed-source/mantenimiento-mlops) |
| API desplegada | [mantenimiento-mlops-api.onrender.com](https://mantenimiento-mlops-api.onrender.com) (documentación en `/docs`) |
| Interfaz desplegada | [mantenimiento-mlops-ui.onrender.com](https://mantenimiento-mlops-ui.onrender.com) |

## Ejecutar desde cero

Requisitos: [uv](https://docs.astral.sh/uv/) (instala Python 3.13 solo) y, opcionalmente, Docker.

```bash
git clone https://github.com/edex10ed-source/mantenimiento-mlops && cd mantenimiento-mlops
uv sync                      # instala dependencias exactas de uv.lock
uv run preparar-datos        # lee, valida y resume el dataset
uv run entrenar              # entrena, evalúa, aplica la puerta de calidad y guarda models/
uv run predecir              # prueba rápida con una muestra de ejemplo
uv run pytest -v             # pruebas de datos, modelo y API
```

Levantar la API y la interfaz (en dos terminales):

```bash
uv run fastapi dev src/mantenimiento_mlops/api.py   # API en http://127.0.0.1:8000/docs
uv run python app/gradio_app.py                     # interfaz en http://127.0.0.1:7860
```

Con Docker (API + interfaz con un solo comando):

```bash
docker compose up --build    # API: http://localhost:8000/docs · UI: http://localhost:7860
```

## Estructura

```
src/mantenimiento_mlops/
  config.py     # rutas, semilla, rangos válidos, hiperparámetros, umbrales (sin números mágicos)
  data.py       # carga, validación (columnas, nulos, tipos, rangos, clases) y split estratificado
  features.py   # variables derivadas físicas (diferencia de temperatura, potencia, desgaste*torque)
  train.py      # entrenamiento reproducible, métricas, puerta de calidad y ficha del modelo
  predict.py    # carga el modelo una sola vez y predice
  api.py        # FastAPI: /, /model-info, /predict (validación con Pydantic)
app/gradio_app.py   # interfaz que consume la API por HTTP (no carga el modelo)
tests/              # test_data.py, test_model.py, test_api.py
Dockerfile · compose.yaml · .dockerignore
.github/workflows/  # ci.yml (datos → entrenar → pytest) y cd.yml (despliegue en Render)
models/             # (generado) maintenance_model.joblib y metrics.json
```

## API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Salud del servicio |
| GET | `/model-info` | Ficha del modelo (`metrics.json`): versión, fecha, parámetros, métricas |
| POST | `/predict` | Predice la falla. 422 si la entrada es inválida, 503 si no hay modelo |

Ejemplo:
Puedes probar la API en `/docs`, que trae un ejemplo cargado.

## Decisiones técnicas

- **Métrica: F1 de la clase "falla".** Solo el 3.4 % de las filas son fallas; un modelo que
  siempre responde "sin falla" tendría ~96.6 % de accuracy y sería inútil. F1 equilibra
  precisión (no dar falsas alarmas) y recall (no dejar pasar fallas).
- **Puerta de calidad:** `uv run entrenar` falla y **no guarda** el modelo si el F1 en test
  es menor que `MIN_F1 = 0.70` (`config.py`). Ese mínimo está bastante por debajo del
  resultado actual para tolerar la variación entre entrenamientos, pero muy por encima
  de un modelo trivial (F1 = 0).
- **Sin fuga de datos:** las columnas `TWF, HDF, PWF, OSF, RNF` indican el *tipo* de falla y solo
  se conocen cuando ya ocurrió; no se usan como entrada. Tampoco `UDI` ni `Product ID`.
- **Variables derivadas:** `temp_diff_k`, `power_w` y `wear_torque` reflejan los mecanismos
  físicos de falla. Subieron el F1 de ~0.67 a ~0.89 con el mismo modelo.
- **Reproducibilidad:** semilla fija (`RANDOM_STATE = 42`), split estratificado 80/20,
  dependencias fijadas en `uv.lock`.
- **El modelo se carga una vez** (`lru_cache`), no en cada petición.

### Resultados (conjunto de test, 2 000 filas)

| F1 | Precisión | Recall | ROC AUC | PR AUC | F1 en validación cruzada (5 folds) |
|---|---|---|---|---|---|
| 0.889 | 0.966 | 0.824 | 0.973 | 0.891 | 0.890 ± 0.021 |

De 68 fallas reales en test: 56 detectadas, 12 omitidas y 2 falsas alarmas.
Los valores exactos de cada entrenamiento quedan en `models/metrics.json`.

### Limitaciones y mejoras

- El dataset es **sintético** (publicado por UCI para investigación); en una planta real habría
  que reentrenar con datos propios y validar el umbral de decisión con el costo real de una
  parada frente a una inspección.
- Con pocas fallas en test (68), las métricas tienen incertidumbre: por eso se reporta también
  validación cruzada.
- Mejoras: registrar experimentos con MLflow, monitorear *data drift* de los sensores,
  ajustar `DECISION_THRESHOLD` según el costo de cada error y reentrenar periódicamente.

## CI/CD y despliegue en Render

- **CI** (`ci.yml`, en cada push a `main`): `uv sync --locked` → `uv run preparar-datos` →
  `uv run entrenar` (con puerta de calidad) → `uv run pytest -v` → guarda `models/` como artefacto.
- **CD** (`cd.yml`): solo si el CI termina en verde, llama a los *deploy hooks* de Render.
  Los hooks están en **GitHub → Settings → Secrets and variables → Actions**
  (`RENDER_DEPLOY_API` y `RENDER_DEPLOY_UI`); nunca van en el código.

Servicios en Render (tipo *Web Service*, entorno Docker, mismo repositorio):

1. **API:** usa el `Dockerfile` tal cual (ya respeta la variable `PORT`).
2. **Interfaz:** mismo repositorio, con el comando de Docker `python app/gradio_app.py`
   y estas variables de entorno: `API_URL` (URL pública de la API),
   `GRADIO_SERVER_NAME=0.0.0.0` y `PORT=7860`.

El **Auto-Deploy** de ambos servicios está en *Off*: solo despliega el CD, y solo si el CI pasó.

En el plan Free el servicio se duerme tras 15 minutos sin tráfico: la primera petición puede tardar
cerca de un minuto. Si falla la demo por eso, `docker compose up` muestra lo mismo en local.
git 