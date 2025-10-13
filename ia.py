import csv
import os
import numpy as np
from queue import PriorityQueue

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

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
    Realiza una búsqueda de ramificación y poda. Construye un árbol de visualización completo
    explorando incluso las ramas marcadas como podadas. Para la visualización, explora todas
    las rutas sin optimización de costos a nodos ya visitados.
    Devuelve la primera ruta válida encontrada y detiene la expansión de una rama al llegar al objetivo.
    """
    try:
        idx_inicio = etiquetas.index(inicio)
        idx_objetivo = etiquetas.index(objetivo)
    except ValueError as e:
        print(f"Error: Nodo '{e.args[0].split()[0]}' no encontrado en las etiquetas.")
        return None, float('inf'), None

    pq = PriorityQueue()
    tree_graph = nx.DiGraph() if LIBS_AVAILABLE else None

    g_inicial = 0
    h_inicial = matriz_heuristica[idx_inicio][idx_objetivo]
    f_inicial = g_inicial + h_inicial
    
    ruta_inicial = [idx_inicio]
    nodo_id_inicial = etiquetas[idx_inicio]
    
    # La cola ahora guarda: (f, idx, g, ruta, padre_fue_podado)
    pq.put((f_inicial, idx_inicio, g_inicial, ruta_inicial, False))

    if tree_graph is not None:
        label_text = f"{etiquetas[idx_inicio]}\ng={g_inicial} h={h_inicial} f={f_inicial}"
        tree_graph.add_node(nodo_id_inicial, label_text=label_text, pruned=False)

    primera_ruta_encontrada = None
    costo_primera_ruta = float('inf')

    print("\n--- Iniciando Búsqueda de Ramificación y Poda (Visualización Total) ---")
    print(f"Inicio: {inicio}, Objetivo: {objetivo}, Límite Hmax = {Hmax}\n")

    while not pq.empty():
        f, nodo_actual_idx, g, ruta, padre_fue_podado = pq.get()
        
        nodo_actual_etiqueta = etiquetas[nodo_actual_idx]
        nodo_actual_id = '->'.join([etiquetas[i] for i in ruta])
        h_actual = matriz_heuristica[nodo_actual_idx][idx_objetivo]

        if padre_fue_podado and tree_graph is not None:
            tree_graph.nodes[nodo_actual_id]['pruned'] = True

        print(f"\n-- Expandir: {nodo_actual_etiqueta} (g={g}, h={h_actual}, f={f})")
        if padre_fue_podado:
            print("    (Rama previamente podada, continuando para visualización)")

        # Si encontramos el objetivo en una rama válida...
        if nodo_actual_idx == idx_objetivo:
            if not padre_fue_podado and primera_ruta_encontrada is None:
                print("\n✅ Primera ruta válida al objetivo encontrada (costo g={g})")
                primera_ruta_encontrada = [etiquetas[i] for i in ruta]
                costo_primera_ruta = g
            
            # Detener la expansión de ESTA rama, sea válida o no.
            print(f"    -> Rama finalizada en el objetivo. No se expande más.")
            continue

        for vecino_idx, peso in enumerate(matriz_principal[nodo_actual_idx]):
            if peso > 0:
                vecino_etiqueta = etiquetas[vecino_idx]
                
                if vecino_idx in ruta:
                    print(f"\n  Hijo: {vecino_etiqueta}. DECISIÓN: CICLO")
                    continue

                g_hijo = g + peso
                h_hijo = matriz_heuristica[vecino_idx][idx_objetivo]
                f_hijo = g_hijo + h_hijo
                
                print(f"\n  Nodo: {nodo_actual_etiqueta}, Nodo hijo: {vecino_etiqueta}, Peso: {peso}, Dist_recorrida: {g_hijo}, Dist_por_recorrer: {h_hijo}, Total: {f_hijo}")

                nueva_ruta = ruta + [vecino_idx]
                hijo_id = '->'.join([etiquetas[i] for i in nueva_ruta])
                label_text = f"{vecino_etiqueta}\ng={g_hijo} h={h_hijo} f={f_hijo}"

                hijo_sera_podado = padre_fue_podado or (f_hijo > Hmax)

                if tree_graph is not None:
                    tree_graph.add_node(hijo_id, label_text=label_text, pruned=hijo_sera_podado)
                    tree_graph.add_edge(nodo_actual_id, hijo_id)

                if f_hijo > Hmax and not padre_fue_podado:
                    print(f"    DECISIÓN: MARCADO COMO PODADO (f_hijo={f_hijo} > Hmax={Hmax}), pero se sigue explorando para visualización.")
                
                print(f"    DECISIÓN: ENCOLADO")
                pq.put((f_hijo, vecino_idx, g_hijo, nueva_ruta, hijo_sera_podado))
                
    print("\n--- Búsqueda Finalizada ---")
    return primera_ruta_encontrada, costo_primera_ruta, tree_graph

def dibujar_arbol_busqueda(tree_graph, ruta_final=None, titulo="Arbol - Ramificación y Poda con Subestimacion"):
    import matplotlib.pyplot as plt
    import networkx as nx

    if tree_graph is None or len(tree_graph) == 0:
        print("No hay arbol de búsqueda para dibujar (faltan librerías o arbol vacío)")
        return

    # === Función auxiliar para obtener posiciones jerárquicas ===
    def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.4, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        """
        Calcula posiciones jerárquicas para nodos de un árbol en layout vertical.
        No requiere pygraphviz.
        """
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

    # === Determinar nodo raíz (el primero que no tiene padres) ===
    posibles_raices = [n for n in tree_graph.nodes if tree_graph.in_degree(n) == 0]
    root = posibles_raices[0] if posibles_raices else list(tree_graph.nodes)[0]

    pos = hierarchy_pos(tree_graph, root=root, width=2.5, vert_gap=0.3)

    # === Configurar colores, etiquetas y bordes ===
    node_colors = []
    node_borders = []
    labels = {}

    def pertenece_a_ruta(nodo):
        """Comprueba si el nodo pertenece a la ruta final (por prefijo de 'A->B->C' etc.)"""
        if not ruta_final:
            return False
        posibles = ['->'.join(ruta_final[:i+1]) for i in range(len(ruta_final))]
        return nodo in posibles

    for node, data in tree_graph.nodes(data=True):
        label = data.get('label_text', node)
        if data.get('pruned', False):
            label += " ❌"
            node_colors.append('#ffb3b3')  # rojo claro para podadas
            node_borders.append('darkred')
        elif pertenece_a_ruta(node):
            node_colors.append('#7CFC00')  # verde brillante para ruta final
            node_borders.append('green')
        else:
            node_colors.append('#ADD8E6')  # azul celeste normal
            node_borders.append('gray')
        labels[node] = label

    # === Dibujar todo el árbol (sin ocultar ramas podadas) ===
    plt.figure(figsize=(14, 9))

    # Dibuja los nodos con colores
    nx.draw_networkx_nodes(
        tree_graph, pos,
        node_color=node_colors,
        node_size=1800,
        edgecolors=node_borders,
        linewidths=1.8
    )

    # Dibuja las aristas (todas)
    nx.draw_networkx_edges(
        tree_graph, pos,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=14
    )

    # Dibuja etiquetas
    nx.draw_networkx_labels(
        tree_graph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold'
    )

    plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    direccion = os.path.dirname(os.path.abspath(__file__))
    nombre_matriz_principal = os.path.join(direccion, 'matriz_de_grafo.csv')
    nombre_matriz_intermedia = os.path.join(direccion, 'matriz_intermedia.csv')
    
    matriz_principal, etiquetas = cargar_matriz(nombre_matriz_principal)
    
    if not matriz_principal:
        print(f"\n❗ Error: No se pudo cargar la matriz principal '{nombre_matriz_principal}', Terminando")
    else:
        print("\n🆗 --- Matriz Principal Cargada Correctamente ---")
        print("Nodos encontrados:", etiquetas)
        print("-" * 35)

        matriz_heuristica, etiquetas_heuristica = cargar_matriz(nombre_matriz_intermedia)
        
        if not matriz_heuristica:
            print("\n⚠️  No se encontro 'matriz_intermedia.csv'. Se usara heuristica h=0 por defecto.")
            num_nodos = len(etiquetas)
            matriz_heuristica = np.zeros((num_nodos, num_nodos), dtype=int).tolist()
        else:
            print("\n🆗 --- Matriz Intermedia Cargada Correctamente ---")
            if etiquetas != etiquetas_heuristica:
                print("❗ Los nodos de las matrices no coinciden, Terminando")
                exit()
        print("-" * 35)

        while True:
            try:
                HMAX = int(input("Ingrese el valor de H (entero mayor que 0): "))
                if HMAX > 0:
                    break
                else:
                    print("\n⚠️  El valor negativo invalido. Ingrese un numero positivo\n")
            except ValueError:
                print("\n⚠️  Valor decimal invalido. Ingrese un numero entero\n")

        INICIO = 'A'
        OBJETIVO = 'Z'

        if INICIO not in etiquetas or OBJETIVO not in etiquetas:
            print(f"❗ El nodo de inicio '{INICIO}' o de objetivo '{OBJETIVO}' no se encuentran en el grafo.")
        else:
            ruta_final, costo_final, arbol_busqueda = busqueda_ramificacion_poda(
                matriz_principal, matriz_heuristica, etiquetas, INICIO, OBJETIVO, HMAX
            )

            if ruta_final:
                print("\n\n--- Resultado Final ---")
                print(f"Mejor ruta calculada: {' -> '.join(ruta_final)}")
                print(f"Costo total : {costo_final}")
                print("-" * 35)
            else:
                print("\n❗ No se encontró una ruta al objetivo que cumpla con las restricciones.")

            if LIBS_AVAILABLE:
                dibujar_arbol_busqueda(arbol_busqueda, ruta_final)
            else:
                print("\nPara visualizar el árbol de búsqueda, instale 'networkx' y 'matplotlib'.")
                print("Para un layout jerárquico, instale también 'pygraphviz'.")
