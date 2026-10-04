"""Simulación reproducible de una Máquina de Boltzmann sin bibliotecas de modelos.

Las unidades son binarias (0 o 1). Este módulo contiene explícitamente las
operaciones de energía, activación sigmoidal y actualización de Gibbs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SEMILLA = 20261004


def validar_parametros(estado: np.ndarray, pesos: np.ndarray, sesgos: np.ndarray) -> None:
    """Verifica las restricciones de una red de Boltzmann binaria."""
    if pesos.ndim != 2 or pesos.shape[0] != pesos.shape[1]:
        raise ValueError("La matriz de pesos debe ser cuadrada.")
    if sesgos.shape != (pesos.shape[0],):
        raise ValueError("El vector de sesgos debe tener una entrada por neurona.")
    if estado.shape != (pesos.shape[0],):
        raise ValueError("El estado debe tener una entrada por neurona.")
    if not np.allclose(pesos, pesos.T):
        raise ValueError("La matriz de pesos debe ser simétrica.")
    if not np.allclose(np.diag(pesos), 0.0):
        raise ValueError("La diagonal de la matriz de pesos debe ser cero.")
    if not np.all(np.isin(estado, (0, 1))):
        raise ValueError("El estado debe contener únicamente unidades binarias 0 o 1.")


def energia(estado: np.ndarray, pesos: np.ndarray, sesgos: np.ndarray) -> float:
    """Calcula E(s) = -1/2 sᵀWs - bᵀs para un estado binario."""
    estado = np.asarray(estado, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    sesgos = np.asarray(sesgos, dtype=float)
    validar_parametros(estado, pesos, sesgos)
    return float(-0.5 * estado @ pesos @ estado - sesgos @ estado)


def probabilidad_activacion(
    indice: int,
    estado: np.ndarray,
    pesos: np.ndarray,
    sesgos: np.ndarray,
    temperatura: float,
) -> float:
    """Devuelve P(s_i=1 | s_-i) mediante la sigmoide del campo local."""
    if temperatura <= 0:
        raise ValueError("La temperatura debe ser mayor que cero.")
    estado = np.asarray(estado, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    sesgos = np.asarray(sesgos, dtype=float)
    validar_parametros(estado, pesos, sesgos)
    if not 0 <= indice < len(estado):
        raise IndexError("El índice de neurona está fuera de rango.")

    campo_local = float(pesos[indice] @ estado + sesgos[indice])
    argumento = np.clip(campo_local / temperatura, -700, 700)
    return float(1.0 / (1.0 + np.exp(-argumento)))


def actualizar_unidad(
    indice: int,
    estado: np.ndarray,
    pesos: np.ndarray,
    sesgos: np.ndarray,
    temperatura: float,
    generador: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Actualiza una unidad con una muestra Bernoulli y devuelve estado y P(1)."""
    nuevo_estado = np.asarray(estado, dtype=int).copy()
    probabilidad = probabilidad_activacion(
        indice, nuevo_estado, pesos, sesgos, temperatura
    )
    nuevo_estado[indice] = int(generador.random() < probabilidad)
    return nuevo_estado, probabilidad


def simular(
    estado_inicial: np.ndarray,
    pesos: np.ndarray,
    sesgos: np.ndarray,
    temperatura: float,
    iteraciones: int,
    semilla: int,
) -> list[dict[str, Any]]:
    """Ejecuta actualizaciones estocásticas y registra cada estado y energía.

    En cada paso se selecciona una neurona al azar y se actualiza con Gibbs.
    El primer registro (paso 0) corresponde al estado inicial.
    """
    if iteraciones < 0:
        raise ValueError("El número de iteraciones no puede ser negativo.")

    estado = np.asarray(estado_inicial, dtype=int).copy()
    pesos = np.asarray(pesos, dtype=float)
    sesgos = np.asarray(sesgos, dtype=float)
    validar_parametros(estado, pesos, sesgos)
    generador = np.random.default_rng(semilla)

    registros: list[dict[str, Any]] = [
        {
            "paso": 0,
            "neurona_actualizada": "inicial",
            "probabilidad_activacion": "",
            "estado": "".join(map(str, estado)),
            "energia": energia(estado, pesos, sesgos),
        }
    ]
    for paso in range(1, iteraciones + 1):
        indice = int(generador.integers(len(estado)))
        estado, probabilidad = actualizar_unidad(
            indice, estado, pesos, sesgos, temperatura, generador
        )
        registros.append(
            {
                "paso": paso,
                "neurona_actualizada": indice,
                "probabilidad_activacion": probabilidad,
                "estado": "".join(map(str, estado)),
                "energia": energia(estado, pesos, sesgos),
            }
        )
    return registros


def exportar_resultados(registros: list[dict[str, Any]], directorio: Path) -> None:
    """Guarda el recorrido y las frecuencias empíricas de cada estado en CSV."""
    directorio.mkdir(parents=True, exist_ok=True)
    with (directorio / "trayectoria.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(registros[0]))
        escritor.writeheader()
        escritor.writerows(registros)

    frecuencias = Counter(fila["estado"] for fila in registros)
    total = len(registros)
    with (directorio / "frecuencias_estados.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["estado", "frecuencia", "proporcion_observada"]
        )
        escritor.writeheader()
        for estado, frecuencia in sorted(frecuencias.items()):
            escritor.writerow(
                {"estado": estado, "frecuencia": frecuencia, "proporcion_observada": frecuencia / total}
            )


def graficar_energia(registros: list[dict[str, Any]], ruta: Path) -> None:
    """Genera la gráfica de energía contra paso de simulación."""
    plt.figure(figsize=(8, 4))
    plt.plot([r["paso"] for r in registros], [r["energia"] for r in registros], color="#1f77b4")
    plt.xlabel("Paso de simulación")
    plt.ylabel("Energía")
    plt.title("Dinámica energética de la red de Boltzmann")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def configuracion_ejemplo() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve una red pequeña con pesos simétricos y diagonal nula."""
    pesos = np.array(
        [[0.0, 0.8, -0.4, 0.2], [0.8, 0.0, 0.6, -0.3], [-0.4, 0.6, 0.0, 0.7], [0.2, -0.3, 0.7, 0.0]]
    )
    sesgos = np.array([0.2, -0.1, 0.15, -0.2])
    estado_inicial = np.array([1, 0, 1, 0])
    return estado_inicial, pesos, sesgos


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulación de una Máquina de Boltzmann desde cero.")
    parser.add_argument("--iteraciones", type=int, default=1_000)
    parser.add_argument("--temperatura", type=float, default=1.0)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    argumentos = parser.parse_args()

    estado_inicial, pesos, sesgos = configuracion_ejemplo()
    registros = simular(estado_inicial, pesos, sesgos, argumentos.temperatura, argumentos.iteraciones, argumentos.semilla)
    exportar_resultados(registros, argumentos.salida)
    graficar_energia(registros, argumentos.salida / "energia.png")
    print(f"Semilla: {argumentos.semilla}")
    print(f"Registros: {len(registros)}")
    print(f"Energía final: {registros[-1]['energia']:.4f}")
    print(f"Archivos exportados en: {argumentos.salida}")


if __name__ == "__main__":
    main()
