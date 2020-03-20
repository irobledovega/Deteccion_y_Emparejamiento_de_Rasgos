import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
input_img = cv.imread('prueba1.jpg',0)
# Crea un objeto tipo FAST con parametros por defecto
fast = cv.FastFeatureDetector_create()
# Detecta los rasgos
kp = fast.detect(input_img,None)
#Dibuja los rasgos detectados sobre la imagen original
output_img = cv.drawKeypoints(input_img, kp, None, color=(255,0,0))
# Muestra la imagen con los rasgos detectados
plt.imshow(output_img),plt.show()
