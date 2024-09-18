import os
import cv2
import numpy as np
from func import *
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

def rename_folders(path):
    # Obtener todas las carpetas en el directorio especificado
    existing_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    # Filtrar las carpetas que contienen 'sample_' y tienen una cadena de fecha de 18 caracteres
    #date_folders = [d for d in existing_folders if 'sample_' in d and len(d.split('_')[-1]) == 18]
    date_folders = [d for d in existing_folders]
    date_folders.sort()
    
    # Renombrar las carpetas secuencialmente
    for index, folder in enumerate(date_folders, start=251):
        new_folder_name = f"sample{str(index).zfill(2)}"
        old_folder_path = os.path.join(path, folder)
        new_folder_path = os.path.join(path, new_folder_name)
        
        # Renombrar la carpeta
        os.rename(old_folder_path, new_folder_path)
        print(f"Renamed '{old_folder_path}' to '{new_folder_path}'")

if __name__ == "__main__":
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]
    
    for word_id in word_ids:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        if os.path.isdir(word_path):
            print(f'Renombrando para "{word_id}"...')
            rename_folders(word_path)
    

