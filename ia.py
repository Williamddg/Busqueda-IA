import csv
import os
import numpy as np # Se usará para crear la matriz de ceros
from queue import PriorityQueue

def cargar_matriz(nombre_archivo):
    """Carga una matriz y etiquetas de un archivo CSV con encabezados."""
    matriz = []
    etiquetas = []
    if not os.path.exists(nombre_archivo):
        return None, None
    try:
        with open(nombre_archivo, mode='r', newline='', encoding='utf-8') as archivo_csv:
            lector_csv = csv.reader(archivo_csv, delimiter=';')
            encabezados = next(lector_csv)[1:]
            etiquetas = encabezados
            for fila in lector_csv:
                if not fila:
                    continue
                fila_numerica = [int(valor) for valor in fila[1:]]
                matriz.append(fila_numerica)
    except (ValueError, IndexError):
        print(f"Error: El archivo CSV '{nombre_archivo}' no tiene el formato esperado o contiene valores no numéricos.")
        return None, None
    return matriz, etiquetas

def busqueda_ramificacion_poda(matriz_principal, matriz_heuristica, etiquetas, inicio, objetivo, Hmax):
    """
    Realiza una búsqueda de ramificación y poda usando una matriz principal para costos 'g'
    y una matriz de heurísticas para costos 'h'.
    """
    try:
        idx_inicio = etiquetas.index(inicio)
        idx_objetivo = etiquetas.index(objetivo)
    except ValueError as e:
        print(f"Error: Nodo '{e.args[0].split()[0]}' no encontrado en las etiquetas.")
        return None, float('inf')

    pq = PriorityQueue()
    
    g_inicial = 0
    # h se calcula desde la matriz de heurísticas respecto al objetivo
    h_inicial = matriz_heuristica[idx_inicio][idx_objetivo]
    f_inicial = g_inicial + h_inicial
    
    # La tupla en la cola de prioridad es: (f, índice_nodo, g, ruta)
    # Se incluye el índice del nodo para desempate: menor índice es mejor.
    pq.put((f_inicial, idx_inicio, g_inicial, [idx_inicio]))

    best_g = {i: float('inf') for i in range(len(etiquetas))}
    best_g[idx_inicio] = g_inicial

    print("\n--- Iniciando Búsqueda de Ramificación y Poda ---")
    print(f"Inicio: {inicio}, Objetivo: {objetivo}, Límite Hmax = {Hmax}\n")

    while not pq.empty():
        f, nodo_actual_idx, g, ruta = pq.get()

        nodo_actual_etiqueta = etiquetas[nodo_actual_idx]
        h_actual = matriz_heuristica[nodo_actual_idx][idx_objetivo]
        
        # Comprobar si el 'g' actual es peor que uno ya encontrado
        if g > best_g[nodo_actual_idx]:
            continue

        print(f"\n>> Expandir: {nodo_actual_etiqueta} (g={g}, h={h_actual}, f={f})")

        if nodo_actual_idx == idx_objetivo:
            print("\n✅--- ¡Objetivo encontrado! ---")
            ruta_etiquetas = [etiquetas[i] for i in ruta]
            return ruta_etiquetas, g

        # Generar hijos (vecinos en el grafo principal)
        for vecino_idx, peso in enumerate(matriz_principal[nodo_actual_idx]):
            if peso > 0: # Si hay una arista en el grafo real
                vecino_etiqueta = etiquetas[vecino_idx]
                
                if vecino_idx in ruta:
                    print(f"  - Hijo: {vecino_etiqueta}. DECISIÓN: CICLO")
                    continue

                g_hijo = g + peso
                # La heurística del hijo es su distancia estimada al objetivo
                h_hijo = matriz_heuristica[vecino_idx][idx_objetivo]
                f_hijo = g_hijo + h_hijo
                
                print(f"  - Padre: {nodo_actual_etiqueta}, Hijo: {vecino_etiqueta}, peso: {peso}, g_hijo: {g_hijo}, h_hijo: {h_hijo}, f_hijo: {f_hijo}")

                if f_hijo > Hmax:
                    print(f"    DECISIÓN: PODADO (f_hijo={f_hijo} > Hmax={Hmax})")
                    continue
                
                if g_hijo >= best_g[vecino_idx]:
                    print(f"    DECISIÓN: NO MEJOR (g_hijo={g_hijo} >= mejor_g[{vecino_etiqueta}]={best_g[vecino_idx]})")
                    continue

                print(f"    DECISIÓN: ENCOLADO")
                best_g[vecino_idx] = g_hijo
                nueva_ruta = ruta + [vecino_idx]
                pq.put((f_hijo, vecino_idx, g_hijo, nueva_ruta))
                
    print("\n--- No se encontró solución ---")
    return None, float('inf')

if __name__ == "__main__":
    direccion = os.path.dirname(os.path.abspath(__file__))
    nombre_matriz_principal = os.path.join(direccion, 'matriz_de_grafo.csv')
    nombre_matriz_intermedia = os.path.join(direccion, 'matriz_intermedia.csv')
    
    # 1. Cargar la matriz principal (caminos reales)
    matriz_principal, etiquetas = cargar_matriz(nombre_matriz_principal)
    
    if not matriz_principal:
        print(f"\nError: No se pudo cargar la matriz principal '{nombre_matriz_principal}', Terminando")
    else:
        print("\n--- Matriz Principal Cargada Correctamente ---")
        print("Nodos encontrados:", etiquetas)
        print("-" * 35)

        # 2. Cargar la matriz de heurísticas (intermedia)
        matriz_heuristica, etiquetas_heuristica = cargar_matriz(nombre_matriz_intermedia)
        
        if not matriz_heuristica:
            print("\n⚠️  No se encontro 'matriz_intermedia.csv'. Se usara heuristica h=0 por defecto.")
            # Crear una matriz de ceros con las dimensiones correctas
            num_nodos = len(etiquetas)
            matriz_heuristica = np.zeros((num_nodos, num_nodos), dtype=int).tolist()
        else:
            print("\n--- Matriz Intermedia Cargada Correctamente ---")
            # Validar que las etiquetas coinciden
            if etiquetas != etiquetas_heuristica:
                print("❗ Los nodos de las matrices no coinciden, Terminando")
                exit()
        print("-" * 35)

        # 3. Pedir HMAX por consola (debe ser entero > 0)
        while True:
            try:
                HMAX = int(input("Ingrese el valor de H (entero mayor que 0): "))
                if HMAX > 0:
                    break
                else:
                    print("\n⚠️  El valor negativo invalido. Ingrese un numero positivo\n")
            except ValueError:
                print("\n⚠️  Valor decimal invalido. Ingrese un numero entero\n")

        # 4. Definir parámetros y ejecutar la búsqueda
        INICIO = 'A'
        OBJETIVO = 'Z'

        if INICIO not in etiquetas or OBJETIVO not in etiquetas:
            print(f"❗ El nodo de inicio '{INICIO}' o de objetivo '{OBJETIVO}' no se encuentran en el grafo.")
        else:
            ruta_final, costo_final = busqueda_ramificacion_poda(
                matriz_principal, matriz_heuristica, etiquetas, INICIO, OBJETIVO, HMAX
            )

            # 5. Mostrar resultados
            if ruta_final:
                print("\n\n--- Resultado Final ---")
                print(f"Mejor ruta calculada: {' -> '.join(ruta_final)}")
                print(f"Costo total : {costo_final}")
                print("-" * 35)
            else:
                print("\nNo se encontró una ruta al objetivo que cumpla con las restricciones.")
