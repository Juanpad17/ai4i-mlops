# AI4I Predictive Maintenance - MLOps

Proyecto de Machine Learning Operations para detección de anomalías en equipos industriales usando el dataset AI4I 2020 Predictive Maintenance. El objetivo es construir un pipeline reproducible de ingestión, validación, entrenamiento, monitoreo y estrategia de retraining.

## 1. Objetivo del proyecto

El proyecto implementa un flujo completo de MLOps para resolver la  detección de comportamientos anómalos de maquinaria que puedan estar asociados con fallas.:

- descargar y preparar datos,
- validar calidad y calidad de datos,
- entrenar un modelo de detección de anomalías,
- registrar y comparar experimentos con MLflow,
- monitorear drift en producción,
- evaluar degradación del modelo,
- decidir si se debe disparar retraining.

## 2. Estructura del repositorio

```text
ai4i-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── api/
│   ├── data_quality/
│   ├── features/
│   ├── ingestion/
│   ├── monitoring/
│   └── training/
├── reports/
│   └── monitoring/
├── artifacts/
├── tests/
├── run_pipeline.py
├── run_simulations_pq.py
├── params.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## 3. Requisitos previos

### Software recomendado

- Python 3.10 o 3.11
- Git
- PowerShell o terminal Bash
- MLflow corriendo localmente en el puerto 5000

### Dependencias

Instala las librerías del proyecto:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si estás usando entorno virtual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Instrucciones completas de reproducción

### 4.1 Activar el entorno virtual

En Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Verifica que el entorno esté activo:

```powershell
python --version
```

### 4.2 Ejecutar el pipeline completo

Desde la raíz del proyecto:

```powershell
python run_pipeline.py
```

Este pipeline ejecuta, en orden:

1. ingestión de datos
2. validación de calidad de datos
3. quality gates
4. entrenamiento del modelo base
5. monitoreo de drift
6. monitoreo del modelo
7. gate de retraining
8. experimentación y comparación de modelos
9. cierre y registro del modelo ganador

### 4.3 Ejecutar cada etapa por separado

#### Ingesta

```powershell
python src/ingestion/ingest.py
```

#### Validación

```powershell
python src/data_quality/validate.py
```

#### Quality gates

```powershell
python src/data_quality/quality_gates.py
```

#### Entrenamiento

```powershell
python src/training/train.py
```

#### Experimentación

```powershell
python src/training/experiment.py
```

#### Finalización del modelo

```powershell
python src/training/finalize_model.py
```

#### Monitoreo de drift

```powershell
python src/monitoring/data_monitoring.py
```

#### Monitoreo del modelo

```powershell
python src/monitoring/model_monitoring.py
```

#### Gate de retraining

```powershell
python src/monitoring/retraining_gate.py
```

## 5. Ejecución de la API

El proyecto incluye una API FastAPI para predicción en tiempo real.

### 5.1 Iniciar la API

```powershell
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 5.2 Verificar salud de la API

```powershell
curl http://localhost:8000/health
```

### 5.3 Realizar predicción

```powershell
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"type\":\"L\",\"air_temperature_k\":298.0,\"process_temperature_k\":308.0,\"rotational_speed_rpm\":1551.0,\"torque_nm\":42.0,\"tool_wear_min\":100}"
```

## 6. Simulaciones y reportes

El repositorio incluye simulaciones para mostrar drift y calidad de datos.

### Ejecutar simulaciones

```powershell
python run_simulations_pq.py
```

Esto genera reportes en:

- reports/monitoring/drift_simulation_report.json
- reports/monitoring/quality_issues_simulation_report.json
- reports/monitoring/simulaciones_p_q_consolidadas.json

### Simulación de drift específica

```powershell
python src/monitoring/simulate_production_drift.py
```

## 7. Monitoreo y estrategia de retraining

El proyecto cuenta con un mecanismo de decisión para activar retraining de manera inteligente. La lógica no se basa únicamente en change de distribución ni solo en degradación del modelo, sino en la combinación de ambas.

### Regla aplicada

Se activa retraining solamente si:

- existe drift severo en la distribución de entrada, y
- la performance del modelo cae por debajo de un umbral aceptable.

En términos prácticos:

```
IF PSI > 0.25 OR drift generalizado
AND F1 < 0.80 OR Recall < 0.90 OR False Positive Rate > 0.15
THEN trigger retraining
```

Esto evita reentrenar por cambios benignos o aislados en los datos, y solo reacciona cuando hay evidencia real de que el modelo ya no está funcionando adecuadamente.

La lógica está implementada en:

- src/monitoring/retraining_policy.py
- src/monitoring/retraining_gate.py

## 8. Archivos clave del proyecto

### Datos

- data/raw/ai4i2020.csv: dataset original
- data/processed/validated.csv: datos limpios y validados

### Monitoreo

- src/monitoring/data_monitoring.py: detección de data drift
- src/monitoring/model_monitoring.py: métricas de producción y evaluación offline
- src/monitoring/retraining_policy.py: lógica de decisión de retraining
- src/monitoring/retraining_gate.py: gate de activación de retraining

### Entrenamiento

- src/training/train.py: entrenamiento del modelo base
- src/training/experiment.py: experimentos comparativos
- src/training/finalize_model.py: registro del modelo ganador

### API

- src/api/app.py: API REST con predicción y logging

## 9. Ejecución de pruebas

Para validar el comportamiento básico del proyecto:

```powershell
pytest -q
```

También pueden ejecutarse pruebas específicas:

```powershell
pytest tests/test_retraining_policy.py -q
pytest tests/test_model.py -q
pytest tests/test_data.py -q
```

## 10. Reportes generados

El pipeline produce artefactos y reportes en las siguientes ubicaciones:

- artifacts/
- reports/monitoring/
- mlartifacts/

Los reportes más relevantes son:

- reports/monitoring/data_drift_summary.json
- reports/monitoring/model_monitoring_summary.json
- reports/monitoring/retraining_decision.json
- reports/monitoring/system_summary.json

## 11. Recomendaciones de uso

- Ejecuta el pipeline completo la primera vez para generar la validación y el modelo base.
- Luego revisa los reportes de drift y performance antes de decidir retraining.
- Usa la API para generar tráfico real y alimentar los logs de producción.
- Si cambian significativamente las condiciones operativas del equipo, revisa el drift y la performance antes de reentrenar.

## 12. Resumen

Este proyecto demuestra un flujo MLOps completo para un problema de mantenimiento predictivo con anomalías. No solo entrena un modelo, sino que además incorpora monitorización, validación de la calidad del dato, evaluación de drift y una estrategia racional de retraining, evitando decisiones automáticas basadas en una sola señal.

## 13. Contacto y uso académico

Este repositorio está pensado como proyecto académico y de práctica de MLOps para clasificación/anomalías en sistemas industriales. Se puede ampliar con:

-alertas por email,
-automatización de retraining en Jenkins/GitHub Actions,
-MLflow con experiment tracking más avanzado,
-integración con dashboards de monitoreo,
-y despliegue en contenedores.
