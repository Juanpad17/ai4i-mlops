"""
SCRIPT MAESTRO DE SIMULACIONES
==============================================

Ejecuta ambas simulaciones de forma coordinada:
  - Simulación de producción y DRIFT
  - Simulación de problemas de calidad

Este script orquesta la ejecución y genera un reporte consolidado.
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime, timezone


def ejecutar_modulo(nombre_modulo: str, ruta_script: str) -> bool:
    """
    Ejecuta un módulo Python y verifica su salida.
    
    Args:
        nombre_modulo: Nombre descriptivo del módulo
        ruta_script: Ruta del script a ejecutar
    
    Returns:
        bool: True si ejecutó exitosamente, False si falló
    """
    print(f"\n{'='*70}")
    print(f"EJECUTANDO: {nombre_modulo}")
    print(f"{'='*70}")
    
    try:
        resultado = subprocess.run(
            [sys.executable, ruta_script],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=False,
            text=True,
            env={
                **__import__("os").environ,
                "PYTHONPATH": str(Path(__file__).resolve().parent),
            },
        )
        print(f"\n✓ {nombre_modulo} ejecutado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error en {nombre_modulo}")
        print(f"  Código de salida: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ No se encontró el archivo: {ruta_script}")
        return False


def generar_reporte_consolidado() -> None:
    """
    Genera un reporte JSON consolidado de ambas simulaciones.
    """
    reportes = {}
    
    # Intentamos cargar el reporte de Simulación de producción y DRIFT
    drift_report = Path("reports/monitoring/drift_simulation_report.json")
    if drift_report.exists():
        with open(drift_report, "r", encoding="utf-8") as f:
            reportes["punto_p_drift"] = json.load(f)
    
    # Intentamos cargar el reporte de Simulación de problemas de calidad
    quality_report = Path("reports/monitoring/quality_issues_simulation_report.json")
    if quality_report.exists():
        with open(quality_report, "r", encoding="utf-8") as f:
            reportes["punto_q_calidad"] = json.load(f)
    
    # Creamos reporte consolidado
    reporte_consolidado = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulaciones_completadas": list(reportes.keys()),
        "datos": reportes,
    }
    
    # Guardamos
    reporte_path = Path("reports/monitoring/simulaciones_p_q_consolidadas.json")
    reporte_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte_consolidado, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Reporte consolidado guardado en: {reporte_path}")
    
    return reporte_consolidado


def main():
    """
    Ejecuta la orquestación de ambas simulaciones.
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PUNTOS P Y Q - SIMULACIÓN DE PRODUCCIÓN Y CALIDAD".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Detectamos la raíz del proyecto
    proyecto_root = Path(__file__).resolve().parent
    print(f"\nRaíz del proyecto: {proyecto_root}")
    
    # Verificamos que los archivos de entrada existen
    validated_csv = proyecto_root / "data/processed/validated.csv"
    if not validated_csv.exists():
        print(f"\n✗ ERROR: No se encontró {validated_csv}")
        print("  Ejecuta primero el pipeline completo:")
        print("  $ python run_pipeline.py")
        return False
    
    print(f"✓ Dataset validado encontrado: {validated_csv.name}")
    
    # Ejecutamos Simulación de producción y DRIFT
    script_p = proyecto_root / "src/monitoring/simulate_production_drift.py"
    exito_p = ejecutar_modulo("PUNTO P - Simulación de Drift", str(script_p))
    
    # Ejecutamos Simulación de problemas de calidad
    script_q = proyecto_root / "src/data_quality/simulate_quality_issues.py"
    exito_q = ejecutar_modulo("PUNTO Q - Simulación de Calidad", str(script_q))
    
    # Generamos reporte consolidado
    if exito_p or exito_q:
        reporte = generar_reporte_consolidado()
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    
    print(f"\nPunto P (Drift):   {'✓ EXITOSO' if exito_p else '✗ FALLÓ'}")
    print(f"Punto Q (Calidad): {'✓ EXITOSO' if exito_q else '✗ FALLÓ'}")
    
    if exito_p and exito_q:
        print(f"\n✓ AMBAS SIMULACIONES COMPLETADAS EXITOSAMENTE")
        print(f"\nReportes generados:")
        print(f"  - reports/monitoring/drift_simulation_report.json (Punto P)")
        print(f"  - reports/monitoring/quality_issues_simulation_report.json (Punto Q)")
        print(f"  - reports/monitoring/simulaciones_p_q_consolidadas.json (Consolidado)")
        return True
    else:
        print(f"\n✗ Una o más simulaciones fallaron")
        return False


if __name__ == "__main__":
    exito = main()
    sys.exit(0 if exito else 1)
