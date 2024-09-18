import os
import cv2
from constants import *

def flip_images_in_directory(input_directory, output_directory=None):
    # Si no se especifica un directorio de salida, usar el mismo que el de entrada
    if output_directory is None:
        output_directory = input_directory
    
    # Recorrer todos los archivos en el directorio de entrada
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg'):
            # Leer la imagen
            filepath = os.path.join(input_directory, filename)
            image = cv2.imread(filepath)

            # Aplicar el flip horizontal
            flipped_image = cv2.flip(image, 1)

            # Guardar la imagen volteada en el directorio de salida
            output_filepath = os.path.join(output_directory, filename)
            cv2.imwrite(output_filepath, flipped_image)

            print(f"Imagen {filename} procesada y guardada en {output_filepath}")

if __name__ == "__main__":
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]

    for word_id in word_ids:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        if os.path.isdir(word_path):
            print(f'Normalizando frames para "{word_id}"...')
            process_directory(word_path, MODEL_FRAMES)
            
    # Definir el directorio de imágenes
    input_dir = "ruta/al/directorio/de/entrada"
    output_dir = None

    # Aplicar el flip a todas las imágenes
    flip_images_in_directory(input_dir, output_dir)
