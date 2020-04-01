import numpy as np
import cv2 as cv
import os.path
# from matplotlib import pyplot as plt

# Define variables globales
input_img = np.zeros((480,640,3), np.uint8)
metodo = '1'

# Funcion que se ejecuta al mover la barra de deslizamiento
def detectar(x):
  t=cv.getTrackbarPos('Threshold','Deteccion de Rasgos')

  # Crea un objeto de acuerdo al metodo seleccionado
  # y llama a la funcion para detectar los rasgos
  if metodo=='1':
    gftt = cv.GFTTDetector_create(0, 0.01, t, 3)
    kp = gftt.detect(input_img,None)
  elif metodo=='2':
    # Crea un objeto tipo FAST
    fast = cv.FastFeatureDetector_create(t)
    # Detecta los rasgos
    kp = fast.detect(input_img,None)
  elif metodo=='3':
    # Crea un objeto tipo AGAST
    agast = cv.AgastFeatureDetector_create(t)
    # Detecta los rasgos
    kp = agast.detect(input_img,None)
  else:
    return
  
  #Dibuja los rasgos detectados sobre la imagen original
  output_img = cv.drawKeypoints(input_img, kp, None, color=(255,0,0))
  cv.imshow('Deteccion de Rasgos',output_img)


# Abre la imagen definida por el usuario
archivo = input('Nombre de la imagen a procesar: ')
while not os.path.isfile(archivo):
  archivo = input('No se encontro el archivo.\nVuelve a teclear el nombre: ')
  pass
input_img = cv.imread(archivo,0)

#Permite al usuario seleccionar el metodo de deteccion de rasgos
print('\nMetodos de Deteccion de Rasgos:\n(1)GFTT\n(2)FAST\n(3)AGAST\n(4)SIFT\n(5)SURF\n')
metodo = input('Selecciona el metodo: ')

#Crea una ventana y muestra la imagen
cv.namedWindow('Deteccion de Rasgos', cv.WINDOW_NORMAL)
cv.createTrackbar('Threshold','Deteccion de Rasgos',10,100,detectar)
cv.imshow('Deteccion de Rasgos',input_img)

#Ciclo que espera un evento
while(1):
    #Espera la tecla ESC para salir del ciclo
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

# Destruye la ventana antes de terminar la ejecucion
cv.destroyAllWindows()
