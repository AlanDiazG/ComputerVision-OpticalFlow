import cv2
import numpy as np
import matplotlib.pyplot as plt

# Importando el video
vc = cv2.VideoCapture("dance.mp4")

#Crear medios para guardar el video del Optic Flow RGB
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Generando el archivo donde se guardará el video 
out = cv2.VideoWriter('NuestroOFResultados/NuestroOpticalFlow.mp4', fourcc, 20.0, (1000, 562))

"""Leyendo el primer frame y sacando medidas"""
_, first_frame = vc.read() # Leer el primer frame

# Reescalando imagen y obteniendo el primer frame
resize_dim = 1000 #Ancho total
max_dim = max(first_frame.shape) 
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
# Convierte el primer cuadro a escala de grises
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

"""Declaracion de variables varias"""
umbral=50 #Umbral de deteccion de movimiento
mask = np.zeros_like(first_frame) #Matriz de zeros para alojar O.F.
mask[..., 1] = 255 # Nos deshacemos de la saturacion de la imagen poniendo al maximo
output_folder2="NuestroOFResultados/" 
i=0
flow2 = np.zeros((562, 1000))
flow3 = np.zeros((562, 1000))
kernel=np.ones((11,11),np.uint8)

while(vc.isOpened() and _==True):
    # Obtenemos los siguientes frames
    _, frame = vc.read()
    if frame is None:
        print("Fin del video.")
        break

    # Redimension y filtrado
    gray = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray,None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #Filtrado de fotograma para eliminar ruido
    gray = cv2.GaussianBlur(gray,(3,3),0)


    """Calculo del Optic Flow"""
    # Calculamos el flujo optico a traves de la diferencia de pixeles entre ambos frames
    # A traves de un umbral
    diff = abs(prev_gray-gray) #Diferencia de fotogramas
    cv2.imshow('GRIS-ORIGINAL', gray) #Desplegar
    condicion= diff> umbral/255 #Condicion para marcar los pixeles que cumplen con el umbral
    condicion2= diff<= umbral/255 # No cumplen
    diff[condicion] = 255 #Comprobando en cada celda del arreglo
    diff[condicion2] = 0 
    diff = diff.astype(np.uint8) #Conversion a imagen
    cv2.imshow('OF+Original', diff) #Desplegar

    prev_gray=gray #Actualizar el frame
    
    """Despliegue Optic Flow"""
    factor_atenuacion = 0.1  # Ajusta este valor según tus preferencias
    frame_oscuro = frame.astype(float) * factor_atenuacion #Calculo de atenuacion por celda
    frame = cv2.resize(frame_oscuro.astype(np.uint8), None, fx=scale, fy=scale) # Redimension    
    diff2=diff 
    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) #Cambio a 3 canales
    # Añadido y despliegue
    dense_flow = cv2.addWeighted(frame, 1,diff, 2, 0) #Sobreponiendo frame original y OF
    cv2.imshow("Dense optical flow", dense_flow) #Despliegue
    

    """Generacion MEI imagen"""
    if 50 <= i <= 80: #Seleccionando el los frames del MEI
        flow2=diff2+flow2 #Acumulacion de los frames
        imagen_escalada = cv2.normalize(flow2, None, 0, 255, cv2.NORM_MINMAX) 
        # Convertir la imagen a tipo uint8
        imagen_uint8 = imagen_escalada.astype(np.uint8)
        # Mostrar la imagen con cv2
        cv2.imshow('MEI', imagen_uint8)
        cv2.imwrite('NuestroOFResultados/MEI.png',imagen_uint8)

    """Generacion MHI"""
    if 50 <= i <= 80:
        magnitude2=cv2.normalize(diff2, None, 0, 255, cv2.NORM_MINMAX)
        
        # Aplica la condición a cada elemento
        condicion = magnitude2 >= 18
        condicion2 = magnitude2 < 18

        # Establece a 255 solo los elementos que cumplen la condición
        magnitude2[condicion] = 255        
        magnitude2[condicion2] = 0

        flow3=magnitude2+flow3*.8 #Atenuacion de los frames viejos
        
        imagen_escalada2 = cv2.normalize(flow3, None, 0, 255, cv2.NORM_MINMAX)
        # Convertir la imagen a tipo uint8
        imagen_uint82 = imagen_escalada2.astype(np.uint8)
        
        # Mostrar la imagen con cv2
        cv2.imshow('MHI', imagen_uint82)
        cv2.imwrite('NuestroOFResultados/MHI.png',imagen_uint82) 


    out.write(dense_flow) #Creacion de video de imagen original + optic flow rgb
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



"""Fuera de la lectura del video"""
"""Iterando el Optic Flow para volver blancos todos los niveles de gris
   con un umbral para eliminar el ruido"""
for x in range(562):
    for y in range(1000):
        if imagen_escalada[x,y] > 0:
            imagen_escalada[x,y] = 255 #Asignando a blanco
            
        else:
            imagen_escalada[x,y] = 0 #Asignando a negro

cv2.imwrite('NuestroOFResultados/MEI binaria2.png',imagen_escalada)       



"""Contorno del MEI/SOBEL"""
sobel=cv2.Sobel(imagen_escalada, ddepth=cv2.CV_64F,dx=1,dy=1,ksize=5)
cv2.imwrite('NuestroOFResultados/ContornoDelMEI.png',sobel) 


vc.release()
cv2.destroyAllWindows()