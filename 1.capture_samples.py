import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from func import *
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

def capture_samples(path, margin_frame=2, min_cant_frames=5, delete_frames=3):
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
    
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_model, \
        vision.HandLandmarker.create_from_options(hand_options) as hand_model:
        video = cv2.VideoCapture(video_source)
        
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.flip(frame,1)

            if not ret:
                break
            
            image = frame.copy()
            pose_result, hand_result = mediapipe_detection(frame, pose_model, hand_model)
            
            if there_hand(hand_result):
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    frames = frames[:-delete_frames]
                    next_sample_number = get_next_sample_number(path)
                    output_folder = os.path.join(path, f"sample{str(next_sample_number).zfill(2)}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    print(f"sample{str(next_sample_number).zfill(2)}")

                frames, count_frame = [], 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            annotated_image = draw_landmarks_on_image(image, pose_result, hand_result)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}', annotated_image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "hola"
    print(word_name)
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
