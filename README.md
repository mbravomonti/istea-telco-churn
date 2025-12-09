# Telco Customer Churn Prediction

## Descripción
Proyecto de MLOps para predecir la rotación de clientes (churn) en una empresa de telecomunicaciones.

## Estructura del Proyecto
- `src/`: Código fuente del proyecto (preprocesamiento, entrenamiento, evaluación).
- `data/`: Datos del proyecto (raw, processed).
- `models/`: Modelos entrenados y serializados.
- `.github/workflows/`: Workflows de CI/CD para GitHub Actions.

## Instalación
1. Crear un entorno virtual:
   ```bash
   conda create -n churn-env python=3.9
   conda activate churn-env
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso
### Ejecución del Pipeline (DVC)
Para reproducir todo el pipeline (preprocesamiento, entrenamiento y evaluación):
```bash
dvc pull
dvc repro
```

### Ejecución Manual
1. Preparación de datos:
   ```bash
   python src/data_prep.py
   ```

## Entrega Final
Para ver el video de presentación y el reporte detallado, consulta [ENTREGA_FINAL.md](ENTREGA_FINAL.md) y [REPORT.md](REPORT.md).
