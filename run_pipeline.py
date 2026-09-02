import subprocess
import sys
from pathlib import Path

# Configuramos la variable de entorno PYTHONPATH para blindar las importaciones de 'src'
import os
os.environ["PYTHONPATH"] = "."

def run_stage(command: list, description: str) -> None:
    """Ejecuta una etapa del pipeline en la terminal y controla errores."""
    print("\n" + "="*60)
    print(f"🚀 INICIANDO: {description}")
    print("="*60)
    
    # Ejecuta el comando capturando el progreso en tiempo real
    process = subprocess.run(command, text=True)
    
    # Si la etapa devuelve un código de salida distinto de 0, significa que falló
    if process.returncode != 0:
        print(f"\n❌ ERROR CRÍTICO: La etapa [{description}] falló.")
        print("El pipeline completo ha sido detenido para proteger el entorno.")
        sys.exit(process.returncode)
        
    print(f"✅ FINALIZADO CON ÉXITO: {description}\n")


def main():
    print("============================================================")
    print("      PIPELINE DE MANTENIMIENTO PREDICTIVO AI4I (MLLOPS)    ")
    print("============================================================")
    
    # Ejecutable de Python del entorno virtual actual
    python_bin = sys.executable

    # 1. ETAPA DE INGESTA DE DATOS
    run_stage(
        [python_bin, "src/ingestion/ingest.py"], 
        "1. Ingesta de Datos (Descarga desde la UCI)"
    )

    # 2. DIAGNÓSTICO DE CALIDAD DE DATOS
    run_stage(
        [python_bin, "src/data_quality/validate.py"], 
        "2. Diagnóstico de Calidad (Generación de Reporte JSON)"
    )

    # 3. COMPUERTAS DE CALIDAD (Data Quality Gates)
    run_stage(
        [python_bin, "src/data_quality/quality_gates.py"], 
        "3. Filtros y Puertas de Calidad (Generación de validated.csv)"
    )

    # 4. ENTRENAMIENTO MODELO BASE
    run_stage(
        [python_bin, "src/training/train.py"], 
        "4. Entrenamiento del Modelo Base (Isolation Forest)"
    )

    # 5. MONITOREO DE DRIFT Y DESEMPEÑO DEL MODELO
    run_stage(
        [python_bin, "src/monitoring/data_monitoring.py"],
        "5. Monitoreo de Drift (Data Monitoring)"
    )

    run_stage(
        [python_bin, "src/monitoring/model_monitoring.py"],
        "6. Monitoreo del Modelo (Model Monitoring)"
    )

    # 7. EVALUACIÓN DE REENTRENAMIENTO
    run_stage(
        [python_bin, "src/monitoring/retraining_gate.py"],
        "7. Evaluación de Trigger de Retraining"
    )

    # 8. EXPERIMENTACIÓN MASIVA Y COMPARACIÓN
    run_stage(
        [python_bin, "src/training/experiment.py"], 
        "8. Comparación Masiva de Modelos en MLflow"
    )

    # 9. REGISTRO Y FINALIZACIÓN DEL MODELO GANADOR
    run_stage(
        [python_bin, "src/training/finalize_model.py"], 
        "9. Cierre de Pipeline y Registro del Modelo Ganador en Producción"
    )

    print("="*60)
    print("🎉 ¡PIPELINE EJECUTADO DE PUNTA A PUNTA DE FORMA IMPECABLE! 🎉")
    print("Todos los resultados, el modelo final y la decisión de retraining están disponibles.")
    print("="*60)

if __name__ == "__main__":
    main()
