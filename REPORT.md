# Reporte Final - Predicción de Churn

## 1. Comparación de Experimentos

Se realizaron múltiples experimentos variando el parámetro de regularización `C` del modelo de Regresión Logística.

| Experimento | C | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Baseline (main) | 1.0 | 0.688 | 0.583 | 0.408 | 0.480 |
| Strong Reg (exp-strong-reg) | 0.1 | 0.687 | 0.584 | 0.392 | 0.469 |
| Weak Reg (exp-weak-reg) | 10.0 | 0.688 | 0.583 | 0.408 | 0.480 |

**Conclusión:**
El modelo con `C=10.0` (Weak Regularization) obtuvo un desempeño similar al baseline y ligeramente superior a la regularización fuerte en términos de Recall y F1 Score. Se seleccionó este modelo para producción.

## 2. Justificación del Modelo Final

El modelo de Regresión Logística ofrece un buen balance entre interpretabilidad y rendimiento base. Aunque el accuracy ronda el 69%, es un punto de partida sólido.
Para futuras iteraciones, se recomienda probar modelos no lineales como Random Forest o XGBoost para capturar relaciones más complejas.

## 3. Estrategia de Despliegue (Producción)

Para llevar este modelo a un entorno productivo, proponemos la siguiente arquitectura:

1.  **API REST con FastAPI**:
    *   Crear un endpoint `/predict` que reciba los datos del cliente en formato JSON.
    *   El servicio cargará el modelo `model.pkl` al inicio.
    *   Se aplicará el mismo preprocesamiento (`src/data_prep.py` logic) antes de la predicción.

2.  **Contenerización**:
    *   Empaquetar la API en una imagen Docker.
    *   `Dockerfile` incluirá `requirements.txt` y el código fuente.

3.  **Orquestación**:
    *   Desplegar el contenedor en un servicio como **Azure Container Apps** o **AWS ECS** para escalabilidad automática.

4.  **Monitoreo**:
    *   Integrar herramientas como **Prometheus** o **Grafana** para monitorear la latencia y el throughput.
    *   Implementar detección de **Data Drift** comparando las distribuciones de entrada en producción vs. entrenamiento.
