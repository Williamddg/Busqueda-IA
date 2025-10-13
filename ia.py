# ia.py con comentarios detallados

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
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

def cargar_matriz(nombre_archivo):
    """
    Carga una matriz de adyacencia y las etiquetas de los nodos desde un archivo CSV.

    El formato esperado del CSV es:
    - La primera fila contiene las etiquetas de los nodos (ej: A;B;C;...). 
    - Las filas siguientes representan la matriz, donde la primera columna es la etiqueta
      del nodo de origen y el resto de los valores son los pesos de las aristas.
    - El delimitador debe ser punto y coma (;).

    Args:
        nombre_archivo (str): La ruta al archivo CSV.

    Returns:
        tuple: Una tupla conteniendo la matriz (list of lists) y las etiquetas (list of str).
               Si el archivo no existe o hay un error de formato, retorna (None, None).
    """
    matriz = []
    etiquetas = []
    # Comprueba si el archivo existe antes de intentar abrirlo.
    if not os.path.exists(nombre_archivo):
        return None, None
    try:
        with open(nombre_archivo, mode='r', newline='', encoding='utf-8') as archivo_csv:
            lector_csv = csv.reader(archivo_csv, delimiter=';')
            
            # La primera fila son los encabezados (etiquetas de los nodos). Se omite el primer elemento.
            encabezados = next(lector_csv)[1:]
            etiquetas = encabezados
            
            # Itera sobre las filas restantes del CSV para construir la matriz.
            for fila in lector_csv:
                if not fila:  # Ignorar filas vacías
                    continue
                # Convierte los valores de la fila a enteros, omitiendo la primera columna (etiqueta).
                fila_numerica = [int(valor) for valor in fila[1:]]
                matriz.append(fila_numerica)
    except (ValueError, IndexError):
        # Captura errores si el CSV no tiene el formato numérico esperado o está mal estructurado.
        print(f"Error: El archivo CSV '{nombre_archivo}' no tiene el formato esperado o contiene valores no numéricos.")
        return None, None
    return matriz, etiquetas

def busqueda_ramificacion_poda(matriz_principal, matriz_heuristica, etiquetas, inicio, objetivo, Hmax):
    """
    Implementa una búsqueda de Ramificación y Poda con Subestimación (Branch and Bound).

    Características principales de esta implementación:
    1.  **Visualización Completa**: El algoritmo está modificado para no detenerse al podar una rama.
        En su lugar, la sigue explorando para poder generar un árbol de búsqueda visual completo.
        Las ramas "podadas" se marcan como tal, pero su exploración continúa.
    2.  **Sin Optimización de Ruta a Nodo**: Se ha eliminado la optimización 'best_g' que evitaría
        explorar caminos más caros a un nodo ya visitado. Esto se hace para asegurar que el árbol
        visual contenga absolutamente todas las rutas posibles que no formen ciclos.
    3.  **Primera Solución**: El algoritmo devuelve la primera ruta válida que encuentra hacia el objetivo,
        pero no se detiene en ese momento, sino que continúa hasta agotar la cola de prioridad para
        completar el árbol.
    4.  **Parada en Objetivo**: Una vez que una rama específica alcanza el nodo objetivo, esa rama
        deja de expandirse.

    Args:
        matriz_principal (list): Matriz de adyacencia con los costos reales (g).
        matriz_heuristica (list): Matriz con los valores heurísticos (h) entre nodos.
        etiquetas (list): Lista de nombres de los nodos.
        inicio (str): Nodo inicial.
        objetivo (str): Nodo objetivo.
        Hmax (int): Límite superior de costo (f) para la poda.

    Returns:
        tuple: (ruta_final, costo_final, arbol_de_busqueda).
    """
    try:
        idx_inicio = etiquetas.index(inicio)
        idx_objetivo = etiquetas.index(objetivo)
    except ValueError as e:
        print(f"Error: Nodo '{e.args[0].split()[0]}' no encontrado en las etiquetas.")
        return None, float('inf'), None

    # Cola de prioridad para gestionar los nodos a expandir. Ordena por el costo 'f'.
    pq = PriorityQueue()
    # Grafo de NetworkX para construir el árbol de búsqueda visual.
    tree_graph = nx.DiGraph() if LIBS_AVAILABLE else None

    # --- Inicialización del Nodo Raíz ---
    g_inicial = 0  # Costo del camino desde el inicio hasta el nodo actual.
    h_inicial = matriz_heuristica[idx_inicio][idx_objetivo] # Costo heurístico estimado hasta el objetivo.
    f_inicial = g_inicial + h_inicial # Costo total estimado (g + h).
    
    ruta_inicial = [idx_inicio]
    nodo_id_inicial = etiquetas[idx_inicio]
    
    # La tupla en la cola de prioridad contiene:
    # (costo_f, indice_nodo, costo_g, ruta_actual, fue_padre_podado)
    # 'fue_padre_podado' es un booleano para propagar el estado de poda a los descendientes.
    pq.put((f_inicial, idx_inicio, g_inicial, ruta_inicial, False))

    if tree_graph is not None:
        # Añade el nodo raíz al grafo visual. Se guarda el texto completo para usos futuros,
        # aunque la visualización final solo muestre el nombre del nodo.
        label_text = f"{etiquetas[idx_inicio]}\ng={g_inicial} h={h_inicial} f={f_inicial}"
        tree_graph.add_node(nodo_id_inicial, label_text=label_text, pruned=False)

    # Variables para almacenar la primera solución válida encontrada.
    primera_ruta_encontrada = None
    costo_primera_ruta = float('inf')

    print("\n" + "="*50)
    print("--- Iniciando Búsqueda (Modo Visualización Total) ---")
    print("="*50)
    print(f"Inicio: {inicio}, Objetivo: {objetivo}, Límite Hmax = {Hmax}\n")

    # --- Bucle Principal de Búsqueda ---
    # El bucle se ejecuta hasta que no queden nodos por explorar en la cola de prioridad.
    while not pq.empty():
        f, nodo_actual_idx, g, ruta, padre_fue_podado = pq.get()
        
        nodo_actual_etiqueta = etiquetas[nodo_actual_idx]
        # El ID único de un nodo en el árbol visual es la ruta completa hasta él.
        nodo_actual_id = '->'.join([etiquetas[i] for i in ruta])
        h_actual = matriz_heuristica[nodo_actual_idx][idx_objetivo]

        # Si el nodo padre fue podado, este nodo también se marca como podado.
        # Esto asegura la propagación del estado de poda a través de las ramas.
        if padre_fue_podado and tree_graph is not None:
            tree_graph.nodes[nodo_actual_id]['pruned'] = True

        # --- Bloque de Información en Terminal ---
        print(f"\n--------------------------------------------------")
        print(f"Ruta Actual: {nodo_actual_id}")
        print(f"-- Expandiendo Nodo: [{nodo_actual_etiqueta}]")
        print(f"   Dist Recorrida (g)    = {g}")
        print(f"   Dist Por Recorrer (h) = {h_actual}")
        print(f"   Total (f)             = {f}")
        if padre_fue_podado:
            print("   (Rama previamente marcada como podada)")

        # --- Verificación de Nodo Objetivo ---
        if nodo_actual_idx == idx_objetivo:
            # Si es la primera vez que llegamos al objetivo por una ruta válida...
            if not padre_fue_podado and primera_ruta_encontrada is None:
                print("\n✅ Nodo Objetivo Encontrado")
                print(f"   Costo (Dist Recorrida): {g}")
                primera_ruta_encontrada = [etiquetas[i] for i in ruta]
                costo_primera_ruta = g
            
            # Se detiene la expansión de ESTA rama para no generar hijos desde el objetivo.
            print(f"   -> Rama finalizada en el objetivo. No se expande más.")
            continue

        # --- Expansión de Nodos Vecinos (Hijos) ---
        for vecino_idx, peso in enumerate(matriz_principal[nodo_actual_idx]):
            if peso > 0:  # Si existe un camino al vecino
                vecino_etiqueta = etiquetas[vecino_idx]
                
                print(f"\n    -> Evaluando hijo: [{vecino_etiqueta}]")

                # 1. Detección de Ciclos
                if vecino_idx in ruta:
                    print(f"       DECISIÓN: Rechazado (El nodo ya fue visitado en esta ruta)")
                    continue

                # 2. Cálculo de Costos para el Hijo
                g_hijo = g + peso
                h_hijo = matriz_heuristica[vecino_idx][idx_objetivo]
                f_hijo = g_hijo + h_hijo
                
                print(f"       - Peso del camino:       {peso}")
                print(f"       - Dist Recorrida (g):    {g_hijo}")
                print(f"       - Dist Por Recorrer (h): {h_hijo}")
                print(f"       - Total (f):             {f_hijo}")

                # 3. Creación del Nodo Hijo para el Grafo Visual
                nueva_ruta = ruta + [vecino_idx]
                hijo_id = '->'.join([etiquetas[i] for i in nueva_ruta])
                label_text = f"{vecino_etiqueta}\ng={g_hijo} h={h_hijo} f={f_hijo}"

                # Un hijo se considera podado si su padre ya lo estaba, o si su costo 'f' supera Hmax.
                hijo_sera_podado = padre_fue_podado or (f_hijo > Hmax)

                if tree_graph is not None:
                    tree_graph.add_node(hijo_id, label_text=label_text, pruned=hijo_sera_podado)
                    tree_graph.add_edge(nodo_actual_id, hijo_id)

                # 4. Lógica de Poda (solo para información en terminal)
                # Si el costo supera Hmax y no venía ya de una rama podada, se informa.
                if f_hijo > Hmax and not padre_fue_podado:
                    print(f"       DECISIÓN: Podado (Total={f_hijo} > H={Hmax}), continuamos para visualización completa del arbol.")
                
                # 5. Añadir a la Cola de Prioridad
                # El nodo se añade a la cola de todas formas para asegurar la visualización completa.
                print(f"       DECISIÓN: Aceptado")
                pq.put((f_hijo, vecino_idx, g_hijo, nueva_ruta, hijo_sera_podado))
                
    print("\n" + "="*50)
    print("--- Búsqueda Finalizada ---")
    print("="*50)
    return primera_ruta_encontrada, costo_primera_ruta, tree_graph

def dibujar_arbol_busqueda(tree_graph, ruta_final=None, titulo="Arbol - Ramificación y Poda con Subestimacion"):
    """
    Dibuja el árbol de búsqueda completo generado por el algoritmo.

    Utiliza `matplotlib` y `networkx`. La disposición de los nodos es jerárquica.

    Args:
        tree_graph (networkx.DiGraph): El grafo del árbol a dibujar.
        ruta_final (list, optional): La lista de nodos de la ruta final para resaltarla.
        titulo (str, optional): Título del gráfico.
    """
    # Si las librerías no están disponibles o el árbol está vacío, no hace nada.
    if tree_graph is None or len(tree_graph) == 0:
        print("No hay arbol de búsqueda para dibujar (faltan librerías o arbol vacío)")
        return

    # --- Función Auxiliar para Layout Jerárquico ---
    # Esta función calcula las posiciones (x, y) de cada nodo para que el árbol
    # se dibuje de arriba hacia abajo de forma ordenada, sin necesidad de 'pygraphviz'.
    def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.4, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = hierarchy_pos(G, root=child, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    # Se busca el nodo raíz del árbol (el que no tiene nodos padre).
    posibles_raices = [n for n in tree_graph.nodes if tree_graph.in_degree(n) == 0]
    root = posibles_raices[0] if posibles_raices else list(tree_graph.nodes)[0]

    # Se calculan las posiciones de todos los nodos.
    pos = hierarchy_pos(tree_graph, root=root, width=2.5, vert_gap=0.3)

    # --- Verificación de Seguridad para Colores ---
    # Este bucle asegura que si un nodo padre está podado, todos sus descendientes
    # también se marquen como podados para la visualización. Es una salvaguarda para
    # garantizar la coherencia visual del árbol.
    for node in list(tree_graph.nodes):
        try:
            ancestors = nx.ancestors(tree_graph, node)
            for ancestor in ancestors:
                if tree_graph.nodes[ancestor].get('pruned', False):
                    tree_graph.nodes[node]['pruned'] = True
                    break
        except nx.NetworkXError:
            pass # Ignorar si el nodo no se encuentra por alguna razón

    # --- Configuración de Estilos de Nodos y Etiquetas ---
    node_colors = []
    node_borders = []
    labels = {}

    # Función para comprobar si un nodo (identificado por su ruta 'A->B->C')
    # pertenece a la ruta final encontrada.
    def pertenece_a_ruta(nodo):
        if not ruta_final:
            return False
        # Genera todas las sub-rutas posibles de la ruta final (ej: 'A', 'A->B', 'A->B->C')
        posibles = ['->'.join(ruta_final[:i+1]) for i in range(len(ruta_final))]
        return nodo in posibles

    # Itera sobre cada nodo para asignarle un color y una etiqueta.
    for node, data in tree_graph.nodes(data=True):
        # Extrae solo el nombre del nodo (ej: 'A') del texto completo guardado.
        label = data.get('label_text', node).split('\n')[0]
        
        if data.get('pruned', False):
            # Nodos podados: color rojo y símbolo ❌.
            label += "✖️"
            node_colors.append('#ffb3b3')  # Rojo claro
            node_borders.append('darkred')
        elif pertenece_a_ruta(node):
            # Nodos de la ruta final: color verde.
            node_colors.append('#7CFC00')  # Verde brillante
            node_borders.append('green')
        else:
            # Otros nodos explorados: color azul.
            node_colors.append('#ADD8E6')  # Azul celeste
            node_borders.append('gray')
        labels[node] = label

    # --- Dibujo del Grafo con Matplotlib ---
    plt.figure(figsize=(14, 9))

    # Dibuja los nodos con sus colores y bordes.
    nx.draw_networkx_nodes(
        tree_graph, pos,
        node_color=node_colors,
        node_size=1800,
        edgecolors=node_borders,
        linewidths=1.8
    )

    # Dibuja las aristas (flechas).
    nx.draw_networkx_edges(
        tree_graph, pos,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=14
    )

    # Dibuja las etiquetas dentro de los nodos.
    nx.draw_networkx_labels(
        tree_graph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold'
    )

    plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
    plt.axis('off')  # Oculta los ejes x, y.
    plt.tight_layout()
    plt.show()

# --- Bloque Principal de Ejecución ---
# Este código solo se ejecuta cuando el script es llamado directamente.
if __name__ == "__main__":
    # Construye las rutas a los archivos CSV basándose en la ubicación del script.
    direccion = os.path.dirname(os.path.abspath(__file__))
    nombre_matriz_principal = os.path.join(direccion, 'matriz_de_grafo.csv')
    nombre_matriz_intermedia = os.path.join(direccion, 'matriz_intermedia.csv')
    
    # Carga la matriz de costos reales.
    matriz_principal, etiquetas = cargar_matriz(nombre_matriz_principal)
    
    if not matriz_principal:
        print(f"\n❗ Error: No se pudo cargar la matriz principal '{nombre_matriz_principal}', Terminando")
    else:
        print("\n🆗 --- Matriz Principal Cargada Correctamente ---")
        print("Nodos encontrados:", etiquetas)
        print("-" * 35)

        # Carga la matriz heurística.
        matriz_heuristica, etiquetas_heuristica = cargar_matriz(nombre_matriz_intermedia)
        
        if not matriz_heuristica:
            # Si no hay archivo de heurística, se crea una matriz de ceros (h=0 para todo).
            print("\n⚠️  No se encontro 'matriz_intermedia.csv'. Se usara heuristica h=0 por defecto.")
            num_nodos = len(etiquetas)
            matriz_heuristica = np.zeros((num_nodos, num_nodos), dtype=int).tolist()
        else:
            print("\n🆗 --- Matriz Intermedia Cargada Correctamente ---")
            # Valida que ambas matrices tengan los mismos nodos.
            if etiquetas != etiquetas_heuristica:
                print("❗ Los nodos de las matrices no coinciden, Terminando")
                exit()
        print("-" * 35)

        # Bucle para solicitar al usuario el valor de Hmax.
        while True:
            try:
                HMAX = int(input("Ingrese el valor de H (entero mayor que 0): "))
                if HMAX > 0:
                    break
                else:
                    print("\n⚠️  El valor negativo invalido. Ingrese un numero positivo\n")
            except ValueError:
                print("\n⚠️  Valor decimal invalido. Ingrese un numero entero\n")

        # Nodos de inicio y fin (hardcodeados).
        INICIO = 'A'
        OBJETIVO = 'Z'

        if INICIO not in etiquetas or OBJETIVO not in etiquetas:
            print(f"❗ El nodo de inicio '{INICIO}' o de objetivo '{OBJETIVO}' no se encuentran en el grafo.")
        else:
            # Llama a la función principal de búsqueda.
            ruta_final, costo_final, arbol_busqueda = busqueda_ramificacion_poda(
                matriz_principal, matriz_heuristica, etiquetas, INICIO, OBJETIVO, HMAX
            )

            # Imprime el resultado final si se encontró una ruta.
            if ruta_final:
                print("\n\n--- Resultado Final ---")
                print(f"Ruta hacia Nodo Objetivo: {' -> '.join(ruta_final)}")
                print(f"Costo total : {costo_final}")
                print("-" * 35)
            else:
                print("\n❗ No se encontró una ruta al objetivo que cumpla con las restricciones.")

            # Llama a la función de dibujo si las librerías están disponibles.
            if LIBS_AVAILABLE:
                dibujar_arbol_busqueda(arbol_busqueda, ruta_final)
            else:
                print("\nPara visualizar el árbol de búsqueda, instale 'networkx' y 'matplotlib'.")