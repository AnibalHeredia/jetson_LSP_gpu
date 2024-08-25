import json
import os
import cv2
import platform
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import pandas as pd
from typing import NamedTuple
from constants import *

# Detectar el sistema operativo
if platform.system() == 'Linux':
    print("init Linux")
    os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
    print("LD_PRELOAD ha sido configurado correctamente.")
    delegate = python.BaseOptions.Delegate.GPU
    video_source = "/dev/video0"
elif platform.system() == 'Windows':
    print("init Windows")
    delegate = python.BaseOptions.Delegate.CPU
    video_source = 1
else:
    raise Exception("Sistema operativo no soportado")


# DRAW LANDMARKS
def draw_landmarks_on_image(rgb_image, pose_result, hand_result):
    annotated_image = np.copy(rgb_image)
    
    if pose_result:
        for pose_landmarks in pose_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z) for landmark in pose_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp.solutions.drawing_utils.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
             )
    
    if hand_result:
        hand_landmarks_list = hand_result.hand_landmarks
        for hand_landmarks in hand_landmarks_list:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z) for landmark in hand_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=4), 
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                ) 
    #annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image

# CREATE KEYPOINTS
def extract_keypoints(pose_results, hands_results):
    hand_info_map = {} 
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    pose= np.zeros(33*4)
    
    #pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]  for landmark in (pose_results.pose_landmarks[0] if pose_results.pose_landmarks else [])]).flatten() if pose_results else np.zeros(33*4)
    
    if pose_results and pose_results.pose_landmarks:
        pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in (pose_results.pose_landmarks[0])]).flatten()
    
    if hands_results and hands_results.handedness:
        for i, hand_info_list in enumerate(hands_results.handedness):
            if hand_info_list:
                hand_info = hand_info_list[0]
                hand_label = hand_info.category_name.lower()
                hand_info_map[i] = hand_label

        hand_landmarks_list = hands_results.hand_landmarks
        
        for index, label in hand_info_map.items():
            if index < len(hand_landmarks_list):     
                keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks_list[index]]).flatten()
                if label == 'left':
                    lh = keypoints
                elif label == 'right':
                    rh = keypoints
    # 21*3*2 + 33*4 = 258 keypoints
    keypoints = np.concatenate([pose, lh, rh])

    # Verificación de longitud
    if len(keypoints) != 258:
        print(f"Error: La cantidad de keypoints detectados es {len(keypoints)}, debería ser 258.")
        return np.zeros(258)
    else:
        return keypoints

# MODEL
base_options_pose = python.BaseOptions(model_asset_path=pose_model_path, delegate=delegate)
base_options_hand = python.BaseOptions(model_asset_path=hand_model_path, delegate=delegate)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options_pose,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    output_segmentation_masks=False
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=num_hands,
    min_hand_detection_confidence=min_hand_detection_confidence,
    min_hand_presence_confidence=min_hand_presence_confidence,
    min_tracking_confidence=min_tracking_confidence
)

def mediapipe_detection(image, pose_landmarker, hand_landmarker):
    # Convertir la imagen a formato RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Ejecutar detección de pose y mano
    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)
    
    return pose_result , hand_result


def get_keypoints(pose_model,hand_model, path):
    #OBTENER KEYPOINTS DE LA MUESTRA
    # Retorna la secuencia de keypoints de la muestra

    kp_seq = np.array([])
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        frame = cv2.imread(img_path)
        pose_result, hand_result = mediapipe_detection(frame, pose_model,hand_model)
        kp_frame = extract_keypoints(pose_result,hand_result)
        kp_seq = np.concatenate([kp_seq, [kp_frame]] if kp_seq.size > 0 else [[kp_frame]])
    return kp_seq

def insert_keypoints_sequence(df, n_sample:int, kp_seq):
    '''
    ### INSERTA LOS KEYPOINTS DE LA MUESTRA AL DATAFRAME
    Retorna el mismo DataFrame pero con los keypoints de la muestra agregados
    '''
    for frame, keypoints in enumerate(kp_seq):
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': [keypoints]}
        df_keypoints = pd.DataFrame(data)
        df = pd.concat([df, df_keypoints])
    
    return df

# GENERAL
def create_folder(path):
    '''
    ### CREAR CARPETA SI NO EXISTE
    Si ya existe, no hace nada.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def get_word_ids(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        return data.get('word_ids')

# TRAINING MODEL
def get_sequences_and_labels(words_id):
    sequences, labels = [], []
    
    for word_index, word_id in enumerate(words_id):
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        data = pd.read_hdf(hdf_path, key='data')
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = [fila['keypoints'] for _, fila in df_sample.iterrows()]
            sequences.append(seq_keypoints)
            labels.append(word_index)
                    
    return sequences, labels

def there_hand(results: NamedTuple) -> bool:
    return results.hand_landmarks