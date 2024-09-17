import cv2
import numpy as np
import os
import shutil
from constants import *

def adjust_frames_in_directory(directory, target_count=15):
    """
    Ajusta la cantidad de frames directamente en el directorio.
    Si hay más de target_count frames, se eliminan los sobrantes.
    Luego, se renombran todos los frames de forma secuencial.

    Parameters:
    - directory: El directorio donde están los frames.
    - target_count: El número deseado de frames.
    """
    # Obtener todos los archivos de imágenes (.jpg) en el directorio
    frame_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    num_frames = len(frame_files)
    
    if num_frames > target_count:
        # Si hay más frames de los necesarios, seleccionamos los que se deben conservar
        indices_to_keep = sorted(np.linspace(0, num_frames - 1, target_count, dtype=int))
        
        # Convertimos a un set para fácil eliminación de los que no están
        indices_to_keep = set(indices_to_keep)
        
        # Eliminar los frames sobrantes
        for i, filename in enumerate(frame_files):
            if i not in indices_to_keep:
                os.remove(os.path.join(directory, filename))

    # Ahora renombrar los frames restantes en orden secuencial (frame_01.jpg, frame_02.jpg, ...)
    remaining_frames = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    for i, filename in enumerate(remaining_frames, start=1):
        new_filename = os.path.join(directory, f'frame_{i:02}.jpg')
        old_filename = os.path.join(directory, filename)
        os.rename(old_filename, new_filename)

def process_directory(word_directory, target_frame_count=15):
    """
    Procesa cada subdirectorio dentro de un directorio de palabras.
    """
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            adjust_frames_in_directory(sample_directory, target_frame_count)

if __name__ == "__main__":
    # Generar para todas las palabras en el directorio raíz
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]
    
    for word_id in word_ids:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        if os.path.isdir(word_path):
            print(f'Normalizando frames para "{word_id}"...')
            process_directory(word_path, MODEL_FRAMES)
