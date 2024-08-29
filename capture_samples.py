import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime

def get_next_sample_number(path):
    existing_samples = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    sample_numbers = []
    for d in existing_samples:
        if d.startswith('sample'):
            try:
                sample_number = int(d.split('_')[0].replace('sample', ''))
                sample_numbers.append(sample_number)
            except ValueError:
                continue
    sample_numbers.sort()
    next_sample_number = 1
    for number in sample_numbers:
        if number == next_sample_number:
            next_sample_number += 1
        else:
            break
    return next_sample_number

def capture_samples(path, margin_frame=2, min_cant_frames=5, delay_frames=3):
    '''
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    `path` ruta de la carpeta de la palabra \n
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
    `min_cant_frames` cantidad de frames minimos para cada muestra \n
    `delay_frames` cantidad de frames que espera antes de detener la captura después de no detectar manos
    '''
    create_folder(path)
    
    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.flip(frame,1)

            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: - (margin_frame + delay_frames)]
                    #today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    #output_folder = os.path.join(path, f"sample_{today}")
                    next_sample_number = get_next_sample_number(path)
                    output_folder = os.path.join(path, f"sample{str(next_sample_number).zfill(2)}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                
                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "gracias"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
