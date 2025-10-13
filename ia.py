# ia.py - Versión en Español

import csv
import os
import numpy as np
from queue import PriorityQueue

# --- Bloque de Importación de Librerías de Visualización ---
# Se intenta importar networkx y matplotlib. Estas librerías son opcionales y solo
# se utilizan para dibujar el árbol de búsqueda. Si no están instaladas, el algoritmo
# funcionará igualmente, pero no generará ninguna visualización gráfica.
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    LIBRERIAS_GRAFICAS_DISPONIBLES = True
except ImportError:
    LIBRERIAS_GRAFICAS_DISPONIBLES = False

def cargar_matriz_desde_csv(ruta_archivo):
    """
    Carga una matriz de adyacencia y los nombres de los nodos desde un archivo CSV.

    Args:
        ruta_archivo (str): La ruta al archivo CSV.

    Returns:
        tuple: Una tupla (matriz, nombres_nodos). Si hay error, retorna (None, None).
    """
    matriz = []
    nombres_nodos = []
    if not os.path.exists(ruta_archivo):
        return None, None
    try:
        with open(ruta_archivo, mode='r', newline='', encoding='utf-8') as archivo_csv:
            lector_csv = csv.reader(archivo_csv, delimiter=';')
            encabezados = next(lector_csv)[1:]
            nombres_nodos = encabezados
            for fila in lector_csv:
                if not fila:
                    continue
                fila_numerica = [int(valor) for valor in fila[1:]]
                matriz.append(fila_numerica)
    except (ValueError, IndexError):
        print(f"Error: El archivo CSV '{ruta_archivo}' no tiene el formato esperado.")
        return None, None
    return matriz, nombres_nodos

def busqueda_ramificacion_y_poda(matriz_costos, matriz_heuristica, nombres_nodos, nodo_inicio, nodo_objetivo, costo_maximo_f):
    """
    Implementa una búsqueda de Ramificación y Poda con Subestimación (Branch and Bound).

    Características de esta implementación:
    1.  **Visualización Completa**: Sigue explorando ramas "podadas" para generar un árbol visual completo.
    2.  **Sin Optimización de Ruta a Nodo**: No usa 'best_g', para asegurar que el árbol visual contenga todas las rutas.
    3.  **Primera Solución**: Devuelve la primera ruta válida encontrada, pero continúa para completar el árbol.
    4.  **Parada en Objetivo**: Una rama de exploración se detiene al alcanzar el nodo objetivo.
    """
    try:
        indice_inicio = nombres_nodos.index(nodo_inicio)
        indice_objetivo = nombres_nodos.index(nodo_objetivo)
    except ValueError as e:
        print(f"Error: Nodo '{e.args[0].split()[0]}' no encontrado en los nombres de nodos.")
        return None, float('inf'), None

    cola_prioridad = PriorityQueue()
    arbol_visual = nx.DiGraph() if LIBRERIAS_GRAFICAS_DISPONIBLES else None

    # --- Inicialización del Nodo Raíz ---
    costo_g_inicial = 0
    costo_h_inicial = matriz_heuristica[indice_inicio][indice_objetivo]
    costo_f_inicial = costo_g_inicial + costo_h_inicial
    ruta_inicial = [indice_inicio]
    id_nodo_inicial = nombres_nodos[indice_inicio]
    
    # La tupla en la cola contiene: (costo_f, indice_nodo, costo_g, ruta, es_rama_podada)
    cola_prioridad.put((costo_f_inicial, indice_inicio, costo_g_inicial, ruta_inicial, False))

    if arbol_visual is not None:
        texto_etiqueta = f"{nombres_nodos[indice_inicio]}\ng={costo_g_inicial} h={costo_h_inicial} f={costo_f_inicial}"
        arbol_visual.add_node(id_nodo_inicial, label_text=texto_etiqueta, pruned=False)

    primera_ruta_valida = None
    costo_primera_ruta = float('inf')

    print("\n" + "="*50)
    print("--- Iniciando Búsqueda (Modo Visualización Total) ---")
    print("="*50)
    print(f"Inicio: {nodo_inicio}, Objetivo: {nodo_objetivo}, Límite Hmax = {costo_maximo_f}\n")

    # --- Bucle Principal de Búsqueda ---
    while not cola_prioridad.empty():
        costo_f, indice_nodo_actual, costo_g, ruta_actual, es_rama_podada = cola_prioridad.get()
        
        nombre_nodo_actual = nombres_nodos[indice_nodo_actual]
        id_nodo_actual = '->'.join([nombres_nodos[i] for i in ruta_actual])
        costo_h_actual = matriz_heuristica[indice_nodo_actual][indice_objetivo]

        if es_rama_podada and arbol_visual is not None:
            arbol_visual.nodes[id_nodo_actual]['pruned'] = True

        # --- Bloque de Información en Terminal ---
        print(f"\n--------------------------------------------------")
        print(f"Ruta Actual: {id_nodo_actual}")
        print(f"-- Expandiendo Nodo: [{nombre_nodo_actual}]")
        print(f"   Dist Recorrida (g)    = {costo_g}")
        print(f"   Dist Por Recorrer (h) = {costo_h_actual}")
        print(f"   Total (f)             = {costo_f}")
        if es_rama_podada:
            print("   (Rama previamente marcada como podada)")

        # --- Verificación de Nodo Objetivo ---
        if indice_nodo_actual == indice_objetivo:
            if not es_rama_podada and primera_ruta_valida is None:
                print("\n✅ Nodo Objetivo Encontrado")
                print(f"   Costo (Dist Recorrida): {costo_g}")
                primera_ruta_valida = [nombres_nodos[i] for i in ruta_actual]
                costo_primera_ruta = costo_g
            
            print(f"   -> Rama finalizada en el objetivo. No se expande más.")
            continue

        # --- Expansión de Nodos Vecinos (Hijos) ---
        for indice_vecino, costo_arista in enumerate(matriz_costos[indice_nodo_actual]):
            if costo_arista > 0:
                nombre_vecino = nombres_nodos[indice_vecino]
                
                print(f"\n    -> Evaluando hijo: [{nombre_vecino}]")

                if indice_vecino in ruta_actual:
                    print(f"       DECISIÓN: CICLO (El nodo ya está en la ruta actual)")
                    continue

                costo_g_hijo = costo_g + costo_arista
                costo_h_hijo = matriz_heuristica[indice_vecino][indice_objetivo]
                costo_f_hijo = costo_g_hijo + costo_h_hijo
                
                print(f"       - Peso del camino:       {costo_arista}")
                print(f"       - Dist Recorrida (g):    {costo_g_hijo}")
                print(f"       - Dist Por Recorrer (h): {costo_h_hijo}")
                print(f"       - Total (f):             {costo_f_hijo}")

                nueva_ruta = ruta_actual + [indice_vecino]
                id_hijo = '->'.join([nombres_nodos[i] for i in nueva_ruta])
                texto_etiqueta = f"{nombre_vecino}\ng={costo_g_hijo} h={costo_h_hijo} f={costo_f_hijo}"

                marcar_hijo_como_podado = es_rama_podada or (costo_f_hijo > costo_maximo_f)

                if arbol_visual is not None:
                    arbol_visual.add_node(id_hijo, label_text=texto_etiqueta, pruned=marcar_hijo_como_podado)
                    arbol_visual.add_edge(id_nodo_actual, id_hijo)

                if costo_f_hijo > costo_maximo_f and not es_rama_podada:
                    print(f"       DECISIÓN: MARCADO COMO PODADO (Total={costo_f_hijo} > Hmax={costo_maximo_f}), pero se sigue explorando.")
                
                print(f"       DECISIÓN: ENCOLADO")
                cola_prioridad.put((costo_f_hijo, indice_vecino, costo_g_hijo, nueva_ruta, marcar_hijo_como_podado))
                
    print("\n" + "="*50)
    print("--- Búsqueda Finalizada ---")
    print("="*50)
    return primera_ruta_valida, costo_primera_ruta, arbol_visual

def dibujar_arbol_de_busqueda(arbol_visual, ruta_optima=None, titulo="Árbol de Búsqueda - Ramificación y Poda"):
    """
    Dibuja el árbol de búsqueda completo generado por el algoritmo.
    """
    if arbol_visual is None or len(arbol_visual) == 0:
        print("No hay árbol de búsqueda para dibujar (faltan librerías o árbol vacío)")
        return

    def calcular_posicion_jerarquica(grafo, nodo_raiz=None, ancho=1.0, espacio_vertical=0.4, posicion_vertical=0, centro_x=0.5, posiciones=None, nodo_padre=None):
        if posiciones is None:
            posiciones = {nodo_raiz: (centro_x, posicion_vertical)}
        else:
            posiciones[nodo_raiz] = (centro_x, posicion_vertical)
        nodos_hijos = list(grafo.successors(nodo_raiz))
        if len(nodos_hijos) != 0:
            delta_x = ancho / len(nodos_hijos)
            siguiente_x = centro_x - ancho / 2 - delta_x / 2
            for hijo in nodos_hijos:
                siguiente_x += delta_x
                posiciones = calcular_posicion_jerarquica(grafo, nodo_raiz=hijo, ancho=delta_x, espacio_vertical=espacio_vertical,
                                    posicion_vertical=posicion_vertical - espacio_vertical, centro_x=siguiente_x, posiciones=posiciones, nodo_padre=nodo_raiz)
        return posiciones

    posibles_raices = [n for n in arbol_visual.nodes if arbol_visual.in_degree(n) == 0]
    nodo_raiz = posibles_raices[0] if posibles_raices else list(arbol_visual.nodes)[0]
    posiciones = calcular_posicion_jerarquica(arbol_visual, nodo_raiz=nodo_raiz, ancho=2.5, espacio_vertical=0.3)

    # --- Verificación de Seguridad para Colores ---
    for id_nodo in list(arbol_visual.nodes):
        try:
            ancestros = nx.ancestors(arbol_visual, id_nodo)
            for ancestro in ancestros:
                if arbol_visual.nodes[ancestro].get('pruned', False):
                    arbol_visual.nodes[id_nodo]['pruned'] = True
                    break
        except nx.NetworkXError:
            pass

    # --- Configuración de Estilos de Nodos y Etiquetas ---
    colores_nodos = []
    bordes_nodos = []
    etiquetas_nodos = {}

    def pertenece_a_ruta(id_nodo_arbol):
        if not ruta_optima:
            return False
        posibles_rutas_str = ['->'.join(ruta_optima[:i+1]) for i in range(len(ruta_optima))]
        return id_nodo_arbol in posibles_rutas_str

    for id_nodo, datos_nodo in arbol_visual.nodes(data=True):
        etiqueta_final = datos_nodo.get('label_text', id_nodo).split('\n')[0]
        
        if datos_nodo.get('pruned', False):
            etiqueta_final += " ❌"
            colores_nodos.append('#ffb3b3')
            bordes_nodos.append('darkred')
        elif pertenece_a_ruta(id_nodo):
            colores_nodos.append('#7CFC00')
            bordes_nodos.append('green')
        else:
            colores_nodos.append('#ADD8E6')
            bordes_nodos.append('gray')
        etiquetas_nodos[id_nodo] = etiqueta_final

    # --- Dibujo del Grafo con Matplotlib ---
    plt.figure(figsize=(14, 9))
    nx.draw_networkx_nodes(arbol_visual, posiciones, node_color=colores_nodos, node_size=1800, edgecolors=bordes_nodos, linewidths=1.8)
    nx.draw_networkx_edges(arbol_visual, posiciones, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=14)
    nx.draw_networkx_labels(arbol_visual, posiciones, labels=etiquetas_nodos, font_size=8, font_weight='bold')
    plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_matriz_costos = os.path.join(directorio_script, 'matriz_de_grafo.csv')
    ruta_matriz_heuristica = os.path.join(directorio_script, 'matriz_intermedia.csv')
    
    matriz_costos, nombres_nodos = cargar_matriz_desde_csv(ruta_matriz_costos)
    
    if not matriz_costos:
        print(f"\n❗ Error: No se pudo cargar la matriz de costos '{ruta_matriz_costos}', Terminando")
    else:
        print("\n🆗 --- Matriz de Costos Cargada Correctamente ---")
        print("Nodos encontrados:", nombres_nodos)
        print("-" * 35)

        matriz_heuristica, nombres_nodos_heuristica = cargar_matriz_desde_csv(ruta_matriz_heuristica)
        
        if not matriz_heuristica:
            print(f"\n⚠️  No se encontró '{{ruta_matriz_heuristica}}'. Se usará heurística h=0.")
            cantidad_nodos = len(nombres_nodos)
            matriz_heuristica = np.zeros((cantidad_nodos, cantidad_nodos), dtype=int).tolist()
        else:
            print("\n🆗 --- Matriz Heurística Cargada Correctamente ---")
            if nombres_nodos != nombres_nodos_heuristica:
                print("❗ Los nodos de las matrices no coinciden. Terminando.")
                exit()
        print("-" * 35)

        while True:
            try:
                COSTO_MAXIMO_F = int(input("Ingrese el valor de H (costo máximo total, entero > 0): "))
                if COSTO_MAXIMO_F > 0:
                    break
                else:
                    print("\n⚠️  El valor debe ser un número positivo.\n")
            except ValueError:
                print("\n⚠️  Valor inválido. Ingrese un número entero.\n")

        NODO_INICIO = 'A'
        NODO_OBJETIVO = 'Z'

        if NODO_INICIO not in nombres_nodos or NODO_OBJETIVO not in nombres_nodos:
            print(f"❗ El nodo de inicio '{NODO_INICIO}' o de objetivo '{NODO_OBJETIVO}' no se encuentran en el grafo.")
        else:
            ruta_solucion, costo_solucion, arbol_resultado = busqueda_ramificacion_y_poda(
                matriz_costos, matriz_heuristica, nombres_nodos, NODO_INICIO, NODO_OBJETIVO, COSTO_MAXIMO_F
            )

            if ruta_solucion:
                print("\n\n--- Resultado Final ---")
                print(f"🔔 Ruta encontrada: {' -> '.join(ruta_solucion)}")
                print(f"Costo total: {costo_solucion}")
                print("-" * 35)
            else:
                print("\n🔔 No se encontró una ruta al objetivo que cumpla con las restricciones.")

            if LIBRERIAS_GRAFICAS_DISPONIBLES:
                dibujar_arbol_de_busqueda(arbol_resultado, ruta_solucion)
            else:
                print("\nPara visualizar el árbol de búsqueda, instale 'networkx' y 'matplotlib'.")
