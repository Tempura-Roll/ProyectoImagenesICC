import cv2

miMatriz = cv2.imread("miNumero.png", cv2.IMREAD_GRAYSCALE)
imagen_pequena = cv2.resize(miMatriz, (8,8))

#invertimos la escala de colores:
i = 0
while i<8:
    j = 0
    while j<8:
        imagen_pequena[i][j] = 255 - imagen_pequena[i][j]
        j=j+1
    i=i+1

#ahora aplicamos el marillo
i = 0
while i<8:
    j = 0
    while j<8:
        imagen_pequena[i][j] = imagen_pequena[i][j]/255*16
        j=j+1
    i=i+1

print(imagen_pequena)