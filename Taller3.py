"""
Taller 3 - Concurrencia (MapReduce)
Problema: balanceo de clases en Baldur's Gate 3.

Se analizan los registros de combates de muchas partidas (personaje, clase
que estaba usando, si ganó el combate y el daño que hizo) para saber:
  1. Qué personajes y qué clases se usaron más durante el juego.
  2. Qué clases se deberían buffear (subir) o nerfear para balancear.

Se usa MapReduce:
  - Map:    cada proceso genera y procesa SU PROPIA sección de registros
            (así no hay que copiar los datos entre procesos) y devuelve
            conteos parciales.
  - Reduce: se combinan los conteos parciales en un único resultado.

Se compara la versión secuencial contra la versión con procesos.

NOTA: los datos son simulados (no hay telemetría real del juego).
"""

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------
# Datos del juego (usados para simular las partidas)
# ---------------------------------------------------------------
# clase: (peso de uso, probabilidad de ganar un combate, daño promedio)
CLASES = {
    "Mago":      (0.14, 0.62, 48),
    "Hechicero": (0.11, 0.58, 44),
    "Paladin":   (0.12, 0.60, 46),
    "Picaro":    (0.10, 0.54, 40),
    "Guerrero":  (0.09, 0.52, 38),
    "Bardo":     (0.08, 0.50, 30),
    "Clerigo":   (0.08, 0.53, 28),
    "Barbaro":   (0.07, 0.49, 41),
    "Druida":    (0.06, 0.50, 33),
    "Explorador": (0.05, 0.44, 31),
    "Monje":     (0.05, 0.42, 29),
    "Brujo":     (0.05, 0.47, 36),
}

# personaje: (peso de uso, clase que trae por defecto)
PERSONAJES = {
    "Astarion":   (0.16, "Picaro"),
    "Gale":       (0.14, "Mago"),
    "Shadowheart": (0.12, "Clerigo"),
    "Laezel":     (0.12, "Guerrero"),
    "Karlach":    (0.12, "Barbaro"),
    "Wyll":       (0.08, "Brujo"),
    "Jaheira":    (0.07, "Druida"),
    "Halsin":     (0.05, "Druida"),
    "Minsc":      (0.05, "Explorador"),
    "Minthara":   (0.04, "Paladin"),
    "Tav":        (0.05, None),  # personaje propio del jugador
}

N_SECCIONES = os.cpu_count() or 4   # una sección (y un proceso) por núcleo
UMBRAL_USO = 0.25        # +-25% respecto al uso esperado se considera alto/bajo
UMBRAL_VICTORIAS = 0.03  # +-3 puntos porcentuales respecto a la media


# ---------------------------------------------------------------
# MAP: genera una sección de registros y devuelve conteos parciales
# ---------------------------------------------------------------
def map_function(tarea):
    """
    tarea = (n_registros, semilla). Cada proceso genera sus propios
    registros con su propia semilla, así solo se envían dos números
    al proceso y un diccionario pequeño de vuelta.
    """
    n, semilla = tarea
    rng = random.Random(semilla)

    nombres_p = list(PERSONAJES)
    pesos_p = [PERSONAJES[p][0] for p in nombres_p]
    nombres_c = list(CLASES)
    pesos_c = [CLASES[c][0] for c in nombres_c]

    personajes = rng.choices(nombres_p, weights=pesos_p, k=n)
    clases_libres = rng.choices(nombres_c, weights=pesos_c, k=n)

    usos_personaje = {}
    usos_clase = {}
    usos_par = {}        # (personaje, clase) -> veces usado
    victorias_clase = {}
    dano_clase = {}

    for personaje, libre in zip(personajes, clases_libres):
        # --- generar el registro del combate ---
        defecto = PERSONAJES[personaje][1]
        # 40% de las veces se deja la clase por defecto del personaje
        clase = defecto if (defecto and rng.random() < 0.4) else libre
        _, p_ganar, dano_prom = CLASES[clase]
        gano = rng.random() < p_ganar
        dano = max(0, rng.gauss(dano_prom, 8))

        # --- contar (parte Map) ---
        usos_personaje[personaje] = usos_personaje.get(personaje, 0) + 1
        usos_clase[clase] = usos_clase.get(clase, 0) + 1
        par = (personaje, clase)
        usos_par[par] = usos_par.get(par, 0) + 1
        if gano:
            victorias_clase[clase] = victorias_clase.get(clase, 0) + 1
        dano_clase[clase] = dano_clase.get(clase, 0) + dano

    return {
        "personaje": usos_personaje,
        "clase": usos_clase,
        "par": usos_par,
        "victorias": victorias_clase,
        "dano": dano_clase,
    }


# ---------------------------------------------------------------
# REDUCE: combina los conteos parciales de todas las secciones
# ---------------------------------------------------------------
def reduce_function(mapped_results):
    total = {"personaje": {}, "clase": {}, "par": {}, "victorias": {}, "dano": {}}
    for parcial in mapped_results:
        for categoria, conteo in parcial.items():
            for clave, valor in conteo.items():
                total[categoria][clave] = total[categoria].get(clave, 0) + valor
    return total


# ---------------------------------------------------------------
# División del trabajo en tareas (una por sección)
# ---------------------------------------------------------------
def crear_tareas(n_registros, n_secciones):
    tam = n_registros // n_secciones
    tareas = [(tam, 42 + i) for i in range(n_secciones)]
    # los registros sobrantes se añaden a la última sección
    tareas[-1] = (tam + n_registros % n_secciones, 42 + n_secciones - 1)
    return tareas


# ---------------------------------------------------------------
# Dos formas de ejecutar el MapReduce
# ---------------------------------------------------------------
def ejecutar_secuencial(n_registros, n_secciones):
    # Mismas tareas y mismas semillas, pero una tras otra en un solo proceso
    tareas = crear_tareas(n_registros, n_secciones)
    return reduce_function([map_function(t) for t in tareas])


def ejecutar_procesos(n_registros, n_secciones):
    tareas = crear_tareas(n_registros, n_secciones)
    with ProcessPoolExecutor(max_workers=n_secciones) as executor:
        results = list(executor.map(map_function, tareas))
    return reduce_function(results)

# ---------------------------------------------------------------
# Análisis de balance (con el resultado del Reduce)
# ---------------------------------------------------------------
def analizar_balance(total, n_registros):
    uso_esperado = 1 / len(CLASES)
    media_victorias = sum(total["victorias"].values()) / n_registros

    filas = []
    for clase, usos in total["clase"].items():
        uso = usos / n_registros
        victorias = total["victorias"].get(clase, 0) / usos
        dano = total["dano"].get(clase, 0) / usos

        uso_alto = uso > uso_esperado * (1 + UMBRAL_USO)
        uso_bajo = uso < uso_esperado * (1 - UMBRAL_USO)
        win_alto = victorias > media_victorias + UMBRAL_VICTORIAS
        win_bajo = victorias < media_victorias - UMBRAL_VICTORIAS

        if uso_alto and win_alto:
            veredicto = "NERF"
        elif uso_bajo and win_bajo:
            veredicto = "BUFF"
        elif uso_alto or uso_bajo or win_alto or win_bajo:
            veredicto = "Vigilar"
        else:
            veredicto = "Balanceada"
        filas.append((clase, uso, victorias, dano, veredicto))

    filas.sort(key=lambda f: f[1], reverse=True)
    return filas, uso_esperado, media_victorias


def reportar(total, n_registros):
    print(f"\n--- Personajes más usados ({n_registros:,} combates) ---")
    for p, v in sorted(total["personaje"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{p:<12} {v:>9,}  ({v / n_registros:6.2%})")

    print("\n--- Combinaciones personaje + clase más usadas ---")
    for (p, c), v in sorted(total["par"].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{p:<12} como {c:<10} {v:>9,}  ({v / n_registros:6.2%})")

    filas, esperado, media = analizar_balance(total, n_registros)
    print(f"\n--- Balance por clase (uso esperado {esperado:.2%}, victorias medias {media:.2%}) ---")
    print(f"{'Clase':<11} {'Uso':>7} {'Victorias':>10} {'Daño prom.':>11}  Veredicto")
    for clase, uso, vic, dano, ver in filas:
        print(f"{clase:<11} {uso:>7.2%} {vic:>10.2%} {dano:>11.1f}  {ver}")




# ---------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------
def medir(funcion, n_registros, n_secciones):
    inicio = time.perf_counter()
    resultado = funcion(n_registros, n_secciones)
    return resultado, time.perf_counter() - inicio


if __name__ == "__main__":
    print(f"Núcleos disponibles: {N_SECCIONES}")
    for n in (50_000_000, 100_000_000, 200_000_000):  
        print(f"\n===== {n:,} registros de combate =====")

        res_sec, t_sec = medir(ejecutar_secuencial, n, N_SECCIONES)
        print(f"[Secuencial] Tiempo: {t_sec:.4f} s")

        res_pro, t_pro = medir(ejecutar_procesos, n, N_SECCIONES)
        print(f"[Procesos (ProcessPoolExecutor)] Tiempo: {t_pro:.4f} s")
        print(f"[Procesos (ProcessPoolExecutor)] Aceleración vs secuencial: {t_sec / t_pro:.2f}x")

        # Las dos versiones deben dar exactamente el mismo resultado
        iguales = (res_sec["clase"] == res_pro["clase"]
                   and res_sec["par"] == res_pro["par"])
        print(f"Resultados idénticos en las 2 versiones: {iguales}")

    reportar(res_sec, n)