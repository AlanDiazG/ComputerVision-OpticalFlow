import cv2
import numpy as np
import matplotlib.pyplot as plt

def flujo_to_image(flujo):
    # Convertir el flujo a valores entre 0 y 255 para su visualización
    magnitud = np.sqrt(np.sum(flujo**2, axis=-1))
    magnitud_normalizada = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)

    # Crear una imagen en tonos de gris
    imagen_flujo = np.zeros_like(flujo)
    imagen_flujo[..., 0] = magnitud_normalizada
    
    return magnitud_normalizada


# Importando el video original2
vc = cv2.VideoCapture("dance.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Generando el archivo donde se guardará el video 
out = cv2.VideoWriter('dense_optical_flow_output.mp4', fourcc, 20.0, (1000, 562))

"""Leyendo el primer frame y sacando medidas"""
# Read first frame
_, first_frame = vc.read() #Leyendo primer frame
# Redimenzionando fotograma
resize_dim = 1000
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
# Convierte el primer cuadro a escala de grises
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Creando mascara
mask = np.zeros_like(first_frame)
# Poniendo la saturacion en el valor maximo
mask[..., 1] = 255
i=0

"""Generando variables varias"""
output_folder="frames_opticflow/" #Carpeta para fotogramas
kernel=np.ones((11,11),np.uint8) #Kernel de limpieza
mei_accumulator = np.zeros_like(prev_gray, dtype=np.float32)
mei_accumulator2 = np.zeros_like(prev_gray, dtype=np.float32)
flow2 = np.zeros((562, 1000)) #Matriz para el MEI
flow3 = np.zeros((562, 1000)) #Matriz para el MHI


while(vc.isOpened() and _==True): #Bucle mientras el video tenga contenido
    # Leyendo cada frame
    _, frame = vc.read()
    
    # Redimencionando el frame y convirtiendo a escala de grises 
    gray = cv2.resize(frame, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('gray',gray)     #Desplegando imagen original en gris
    """Metodo de Farneback"""
    # Calcular el flujo optico usando el metodo de Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, 
                                        levels = 3, winsize = 15, 
                                        iterations = 3, poly_n = 5, 
                                        poly_sigma = 1.2, flags = 0)
    
    filename = f'{output_folder}opticflow_{i}.png' #Guardando cada fotograma como imagen
    cv2.imwrite(filename, flujo_to_image(flow))

    """Calculo del OF RGB"""
    # Calcular la magnitud y angulo de cada vector del Flujo Optico
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Ajustando el color del pixel dependiendo de la direccion
    mask[..., 0] = angle * 180 / np.pi / 2
    
    # Normalizando la imagen
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convirtiendo el valor HSV a RGB
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    # Ajustar la intensidad de la imagen original para hacerla más oscura
    factor_atenuacion = 0.1  # Ajusta este valor según tus preferencias
    frame_oscuro = frame.astype(float) * factor_atenuacion
    
    # Redimensionando frame
    frame = cv2.resize(frame_oscuro.astype(np.uint8), None, fx=scale, fy=scale)
    
    # Desplegando imagenes/video
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    cv2.imshow("Dense optical flow", dense_flow)
    
    """Generacion MEI imagen"""
    if 50 <= i <= 80:
        flow2=magnitude+flow2 # Acumulando el valor del OF 
        imagen_escalada = cv2.normalize(flow2, None, 0, 255, cv2.NORM_MINMAX) #Normalizando
        
        # Convertir la imagen a tipo uint8
        imagen_uint8 = imagen_escalada.astype(np.uint8)
        
        # Mostrar y guardar la imagen con cv2
        cv2.imshow('Imagen', imagen_uint8)
        cv2.imwrite('MEI.png',imagen_uint8)

    """Generacion MHI"""
    if 50 <= i <= 80:
        magnitude2=cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Aplica la condición a cada elemento
        condicion = magnitude2 >= 18
        condicion2 = magnitude2 < 18
        # Establece a 255 solo los elementos que cumplen la condición
        magnitude2[condicion] = 255        
        magnitude2[condicion2] = 0

        #Acumulacion de los arreglos 
        flow3=magnitude2+flow3*.8
        imagen_escalada2 = cv2.normalize(flow3, None, 0, 255, cv2.NORM_MINMAX)

        # Convertir el arreglo a imagen
        imagen_uint82 = imagen_escalada2.astype(np.uint8)

        # Mostrar la imagen con cv2
        cv2.imshow('Imagen2', imagen_uint82)
        cv2.imwrite('MHI.png',imagen_uint82)    

    out.write(dense_flow) #Creacion de video de imagen original + optic flow rgb
    prev_gray = gray # Actualizando frame
    i=i+1
    
    # El programa terminar si se aprieta q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


"""Fuera del bucle"""

"""Limpieza del MEI""" 
imagen_erodida = cv2.erode(imagen_escalada, kernel, iterations=2) #Erosion del pixel
imagen_escalada = cv2.dilate(imagen_erodida, kernel, iterations=2) #Dilatacion del pixel
for x in range(562): #Altura
    for y in range(1000): #Ancho
        if imagen_escalada[x,y] >= 23: #Conversion de pixeles mayores a 23 a blanco
            imagen_escalada[x,y] = 255 
            
        else: #El resto se hace 0
            imagen_escalada[x,y] = 0

cv2.imwrite('MEI binaria2.png',imagen_escalada) # Desplegando MEI



"""Contorno del MEI/SOBEL"""
sobel=cv2.Sobel(imagen_escalada, ddepth=cv2.CV_64F,dx=1,dy=1,ksize=5)
cv2.imwrite('ContornoDelMEI.png',sobel)       

#Liberando el video y cerrando pestañas
vc.release()
cv2.destroyAllWindows()