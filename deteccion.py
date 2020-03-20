import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# Abre la imagen definida por el usuario
archivo = input('Nombre de la imagen a procesar: ')
while not os.path.isfile(archivo):
  archivo = input('No se encontro el archivo.\nVuelve a teclear el nombre: ')
  pass
input_img = cv.imread(archivo,0)
# Crea un objeto tipo FAST con parametros por defecto
fast = cv.FastFeatureDetector_create()
# Detecta los rasgos
kp = fast.detect(input_img,None)
#Dibuja los rasgos detectados sobre la imagen original
output_img = cv.drawKeypoints(input_img, kp, None, color=(255,0,0))
# Muestra la imagen con los rasgos detectados
plt.imshow(output_img),plt.show()
