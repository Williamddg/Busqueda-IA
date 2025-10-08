import csv
import os

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
        print("Error,el  archivo contiene valores no numericos y no puede ser leido")
        return None
    return matriz

def imprimir_matriz(matriz,archivo):
    if not matriz:
        print("Matriz vacia o no se pudo leer")
        return
    print(f"Matriz cargada desde '{archivo}':")
    for fila in matriz:
        print(fila)

if __name__ == "__main__":
    direccion = os.path.dirname(os.path.abspath(__file__))
    nombre_archivo = os.path.join(direccion, 'matriz_de_grafo.csv')
    
    matriz = cargar_matriz(nombre_archivo)
    
    if matriz is not None:
        imprimir_matriz(matriz, nombre_archivo)