import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from func import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime

def rename_folders(path):
    # Obtener todas las carpetas en el directorio especificado
    existing_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # Filtrar las carpetas que contienen 'sample_' y tienen una cadena de fecha de 18 caracteres
    date_folders = [d for d in existing_folders if 'sample_' in d and len(d.split('_')[-1]) == 18]
    date_folders.sort()
    
    # Renombrar las carpetas secuencialmente
    for index, folder in enumerate(date_folders, start=1):
        new_folder_name = f"sample{str(index).zfill(2)}"
        old_folder_path = os.path.join(path, folder)
        new_folder_path = os.path.join(path, new_folder_name)
        
        # Renombrar la carpeta
        os.rename(old_folder_path, new_folder_path)
        print(f"Renamed '{old_folder_path}' to '{new_folder_path}'")

if __name__ == "__main__":
    # Cambia esta ruta a la ubicación donde tienes las carpetas guardadas
    word_path = "C:/Users/ENOS/Desktop/modelo_lstm_lsp/hola"  # Ajusta esta ruta según sea necesario
    
    # Llamar a la función para renombrar las carpetas
    rename_folders(word_path)

