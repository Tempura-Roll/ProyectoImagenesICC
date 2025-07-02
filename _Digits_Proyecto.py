from collections import Counter
from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ===========================
# Función para calcular promedios
# ===========================
def calcular_promedios(images, targets):
    avg_images = []
    for d in range(10):
        avg_images.append(images[targets == d].mean(axis=0))
    return np.array(avg_images)


# ===========================
# Función para visualizar una imagen
# ===========================
def plot_image(img, title=""):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# ===========================
# Clasificación por distancias a todos los dígitos (Versión 1)
# ===========================
def clasificar_version_1(nuevo, data, targets, k=3):
    distancias = cdist([nuevo.flatten()], data)[0]
    indices_cercanos = np.argsort(distancias)[:k]
    return indices_cercanos, targets[indices_cercanos]


# ===========================
# Clasificación por distancia a promedios (Versión 2)
# ===========================
def clasificar_version_2(nuevo, avg_images):
    dist_prom = cdist([nuevo.flatten()], avg_images.reshape(10, -1))[0]
    return np.argmin(dist_prom)


# ===========================
# Clasificar por mayoría entre los 3 más cercanos
# ===========================
def clasificar_por_mayoria(targets_cercanos):
    conteo = Counter(targets_cercanos)
    mas_comun, freq = conteo.most_common(1)[0]
    if freq >= 2:
        return mas_comun
    else:
        # Por ahora, tomamos el más cercano
        return targets_cercanos[0]


# ===========================
# Evaluar accuracy en test set
# ===========================
def evaluar_modelos(X_train, y_train, X_test, y_test, avg_images):
    X_train_flat = X_train.reshape(len(X_train), -1)
    y_pred_v1, y_pred_v2 = [], []
    for img in X_test:
        _, cercanos = clasificar_version_1(img, X_train_flat, y_train)
        y_pred_v1.append(clasificar_por_mayoria(cercanos))
        y_pred_v2.append(clasificar_version_2(img, avg_images))
    acc_v1 = accuracy_score(y_test, y_pred_v1)
    acc_v2 = accuracy_score(y_test, y_pred_v2)
    return acc_v1, acc_v2, y_pred_v1, y_pred_v2


# ===========================
# Leer un dígito desde un archivo
# ===========================
def leer_imagen_con_cv2(ruta):
    miMatriz = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if miMatriz is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta}")

    imagen_pequena = cv2.resize(miMatriz, (8, 8))

    # Invertir colores
    imagen_pequena = 255 - imagen_pequena

    # Escalar a rango 0-16
    imagen_normalizada = (imagen_pequena / 255.0) * 16
    imagen_final = imagen_normalizada.astype(np.float32)

    return imagen_final


# ===========================
# Guardar reportes de presición
# ===========================


def guardar_reportes_txt(nombre_archivo, acc_v1, acc_v2, y_test, y_pred_v1, y_pred_v2):
    with open(nombre_archivo, "w") as f:
        f.write(f"Accuracy version 1 (3 vecinos): {acc_v1:.4f}\n")
        f.write(f"Accuracy version 2 (promedios): {acc_v2:.4f}\n\n")

        f.write("=== Reporte clasificación versión 1 ===\n")
        f.write(classification_report(y_test, y_pred_v1, digits=4))
        f.write("\n=== Reporte clasificación versión 2 ===\n")
        f.write(classification_report(y_test, y_pred_v2, digits=4))


# ===========================
# Programa principal
# ===========================
if __name__ == "__main__":
    digits = load_digits()
    images = digits.images
    data = digits.data
    targets = digits.target

    # Calcular promedios
    avg_images = calcular_promedios(images, targets)

    # Mostrar promedios
    if input("Desea ver las imagenes promedio? (y/n): ") == "y":
        for d in range(10):
            plot_image(avg_images[d], title=f'Promedio dígito {d}')

    # Separar entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        data.reshape(-1, 8, 8), targets, test_size=0.2, random_state=42
    )
    X_train_flat = X_train.reshape(len(X_train), -1)

    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
    print(f"Tamaño del conjunto de prueba: {len(X_test)}")

    # Evaluar los dos métodos automáticamente
    acc_v1, acc_v2, y_pred_v1, y_pred_v2 = evaluar_modelos(X_train, y_train, X_test, y_test, avg_images)
    print(f"Accuracy versión 1 (3 más cercanos): {acc_v1:.2f}")
    print(f"Accuracy versión 2 (distancia a promedios): {acc_v2:.2f}")

    print("\n=== Reporte clasificación versión 1 ===")
    print(classification_report(y_test, y_pred_v1, digits=4))

    print("=== Reporte clasificación versión 2 ===")
    print(classification_report(y_test, y_pred_v2, digits=4))

    # Mostrar aciertos por dígito
    totales = {d: 0 for d in range(10)}
    aciertos_v1 = {d: 0 for d in range(10)}
    aciertos_v2 = {d: 0 for d in range(10)}
    for i, img in enumerate(X_test):
        totales[y_test[i]] += 1
        if y_pred_v1[i] == y_test[i]:
            aciertos_v1[y_test[i]] += 1
        if y_pred_v2[i] == y_test[i]:
            aciertos_v2[y_test[i]] += 1

    print("\nAciertos por dígito (versión 1):", {k: f"{aciertos_v1[k]}/{totales[k]}" for k in totales})
    print("Aciertos por dígito (versión 2):", {k: f"{aciertos_v2[k]}/{totales[k]}" for k in totales})

    # ================================
    # Análisis del inciso h
    # ================================
    print(f"\nCantidad de imágenes clasificadas en los experimentos: {len(X_test)}")

    print("\nPorcentaje de aciertos por dígito:")
    mejor_v1, peor_v1 = None, None
    mejor_v2, peor_v2 = None, None
    max_v1, min_v1 = -1, 101
    max_v2, min_v2 = -1, 101

    for d in range(10):
        porc_v1 = 100 * aciertos_v1[d] / totales[d]
        porc_v2 = 100 * aciertos_v2[d] / totales[d]
        print(f"Dígito {d}: v1 = {porc_v1:.1f}%, v2 = {porc_v2:.1f}%")

    # Identificar mejor y peor por versión
        if porc_v1 > max_v1:
            max_v1 = porc_v1
            mejor_v1 = d
        if porc_v1 < min_v1:
            min_v1 = porc_v1
            peor_v1 = d

        if porc_v2 > max_v2:
            max_v2 = porc_v2
            mejor_v2 = d
        if porc_v2 < min_v2:
            min_v2 = porc_v2
            peor_v2 = d

    print("\nResumen de desempeño por dígito:")
    print(f"Versión 1 acierta mejor en el dígito {mejor_v1} ({max_v1:.1f}%) y peor en el {peor_v1} ({min_v1:.1f}%)")
    print(f"Versión 2 acierta mejor en el dígito {mejor_v2} ({max_v2:.1f}%) y peor en el {peor_v2} ({min_v2:.1f}%)")


    guardar_reportes_txt("reporte_clasificacion.txt", acc_v1, acc_v2, y_test, y_pred_v1, y_pred_v2)
    print("\nReporte guardado en 'reporte_clasificacion.txt'")


    # =================================
    # Leer imagen externa usando OpenCV
    # =================================

    print("\nSeleccione una opción para probar la IA:")
    print("1. Usar imagen externa")
    print("2. Usar un elemento aleatorio del test automático")

    opcion = input("Ingrese 1 o 2: ").strip()
    randnum = randint(0, len(X_test) - 1)
    if opcion == "1":
        try:
            number = input("Ingrese el número de la imagen(0 - 9): ")
            nuevo = leer_imagen_con_cv2(f"image-test-{number}.png")
            plot_image(nuevo, title="Imagen cargada desde archivo externo")
        except Exception as e:
            print(f"\nError al leer la imagen externa: {e}")
            nuevo = X_test[randnum]
            plot_image(nuevo, title=f"Se usó imagen de prueba automática (etiqueta real: {y_test[randnum]})")
    else:
        nuevo = X_test[randnum]
        plot_image(nuevo, title=f"Imagen automática (etiqueta real: {y_test[randnum]})")

    indices_cercanos, targets_cercanos = clasificar_version_1(nuevo, X_train_flat, y_train)
    print("\nIndices de los 3 más cercanos:", indices_cercanos)
    print("Targets de los 3 más cercanos:", targets_cercanos)

    clasificacion_v1_usuario = clasificar_por_mayoria(targets_cercanos)
    print(f"Soy la inteligencia artificial (versión 1), y he detectado que el dígito es {clasificacion_v1_usuario}")

    clasificacion_v2_usuario = clasificar_version_2(nuevo, avg_images)
    print(f"Soy la inteligencia artificial (versión 2), y he detectado que el dígito es {clasificacion_v2_usuario}")
