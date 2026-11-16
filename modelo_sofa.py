"""Salud a tiempo: modelo de matrices y funciones para el puntaje SOFA simplificado."""
import numpy as np

# Orden de las variables en el vector x (un paciente):
# [PaO2/FiO2, plaquetas, bilirrubina, PAM, Glasgow, creatinina]
SISTEMAS = ["respiratorio", "coagulación", "hígado", "cardiovascular", "neurológico", "renal"]

# Matriz de umbrales T (6x4): umbral para sumar 1, 2, 3 y 4 puntos en cada sistema
T = np.array([
    [400, 300, 200, 100],     # PaO2/FiO2 (menor es peor)
    [150, 100,  50,  20],     # plaquetas (menor es peor)
    [1.2, 2.0, 6.0, 12.0],    # bilirrubina (mayor es peor)
    [ 70,   0,   0,   0],     # PAM (simplificado: sin vasopresores, solo 0 o 1)
    [ 15,  13,  10,   6],     # Glasgow (menor es peor)
    [1.2, 2.0, 3.5,  5.0],    # creatinina (mayor es peor)
])

# True si en ese sistema un valor MENOR indica mayor disfunción
MENOS_ES_PEOR = np.array([True, True, False, True, True, False])


def puntajes(X):
    """Matriz S (n x 6) con el puntaje 0-4 de cada sistema, para n pacientes."""
    X = np.atleast_2d(X)[:, :, None]                  # (n, 6, 1)
    cumple = np.where(MENOS_ES_PEOR[None, :, None], X < T, X >= T)  # (n, 6, 4)
    return cumple.sum(axis=2)


def sofa(X):
    """Vector con el SOFA total de cada paciente: S · 1."""
    return puntajes(X).sum(axis=1)


def alerta(X_actual, X_basal, delta=2):
    """True para cada paciente cuyo SOFA subió 'delta' o más respecto a su basal."""
    return (sofa(X_actual) - sofa(X_basal)) >= delta


if __name__ == "__main__":
    basal = np.array([[450, 200, 0.8, 85, 15, 0.9]])
    actual = np.array([[250, 90, 2.5, 65, 13, 1.5]])
    print("Puntajes por sistema:", dict(zip(SISTEMAS, puntajes(actual)[0])))
    print("SOFA basal:", sofa(basal)[0], "| SOFA actual:", sofa(actual)[0])
    print("Alerta:", alerta(actual, basal)[0])
