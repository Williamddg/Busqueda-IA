import csv
import os
import matplotlib.pyplot as plt
import networkx as nx


def cargar_matriz(nombre_archivo):
    matriz = []
    try:
        with open(nombre_archivo, mode='r', newline='', encoding='utf-8') as archivo_csv:
            lector_csv = csv.reader(archivo_csv, delimiter=';')
            for fila in lector_csv:
                if not fila:
                    break
                fila_numerica = [int(valor) for valor in fila]
                matriz.append(fila_numerica)
    except FileNotFoundError:
        print(f"Error, archivo '{nombre_archivo}' no encontrado")
        return None
    except ValueError:
        print("Error: el archivo contiene valores no numéricos")
        return None
    return matriz


def matriz_a_grafo(matriz):
    grafo = {}
    nodos = [chr(65 + i) for i in range(len(matriz))]  # A, B, C, ...
    for i in range(len(matriz)):
        origen = nodos[i]
        grafo[origen] = []
        for j in range(len(matriz[i])):
            if matriz[i][j] > 0:  # hay arista
                grafo[origen].append((nodos[j], matriz[i][j]))
    return grafo, nodos


def ramificacion_y_poda(grafo, inicio, objetivo, limite):
    frontera = [(0, [inicio])]
    mejor_camino = None
    mejor_costo = float("inf")

    explorados = []
    podados = []

    while frontera:
        costo_actual, camino = frontera.pop(0)
        nodo_actual = camino[-1]

        if costo_actual >= limite:
            podados.append((camino, costo_actual))
            continue

        explorados.append((camino, costo_actual))

        if nodo_actual == objetivo:
            if costo_actual < mejor_costo:
                mejor_costo = costo_actual
                mejor_camino = camino
            continue

        for vecino, costo in grafo.get(nodo_actual, []):
            if vecino not in camino:
                nuevo_costo = costo_actual + costo
                nuevo_camino = camino + [vecino]
                frontera.append((nuevo_costo, nuevo_camino))

    return mejor_camino, (mejor_costo if mejor_camino else None), explorados, podados


if __name__ == "__main__":
    # Cargar la matriz desde el CSV
    direccion = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = os.path.join(direccion, 'matriz_de_grafo.csv')

    matriz = cargar_matriz(nombre_archivo)
    if matriz is None:
        exit()

    # Convertir la matriz a grafo
    grafo, nodos = matriz_a_grafo(matriz)

    print("Nodos disponibles:", nodos)
    inicio = input("Ingrese el nodo inicial: ").upper()
    objetivo = input("Ingrese el nodo objetivo: ").upper()
    limite = int(input("Ingrese el límite heurístico (valor máximo permitido): "))

    mejor_camino, costo, explorados, podados = ramificacion_y_poda(
        grafo, inicio, objetivo, limite
    )

    # Dibujar el grafo
    G = nx.DiGraph()
    for nodo, vecinos in grafo.items():
        for v, c in vecinos:
            G.add_edge(nodo, v, weight=c)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

    nx.draw(
        G, pos, with_labels=True,
        node_size=800, node_color="lightblue",
        font_size=10, font_weight="bold"
    )
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    # Resaltar explorados (verde)
    for camino_e, _ in explorados:
        edges = [(camino_e[i], camino_e[i+1]) for i in range(len(camino_e)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="green", width=2)

    # Resaltar podados (rojo)
    for camino_p, _ in podados:
        edges = [(camino_p[i], camino_p[i+1]) for i in range(len(camino_p)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="red", style="dashed")

    # Mejor camino (azul)
    if mejor_camino:
        edges = [(mejor_camino[i], mejor_camino[i+1]) for i in range(len(mejor_camino)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="blue", width=3)
        print("\n Mejor camino dentro del límite:", " -> ".join(mejor_camino))
        print("Costo total:", costo)
    else:
        print("\n No se encontró un camino dentro del límite dado.")

    plt.title("(verde=explorado, rojo=podado, azul=mejor camino)")
    plt.show()

