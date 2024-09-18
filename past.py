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


def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)

    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints


def adjust_frames(frames, target_count=15):
    num_frames = len(frames)
    
    if num_frames > target_count:
        # Reducir el número de frames de manera uniforme
        indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
        adjusted_frames = [frames[int(i)] for i in indices]
        
        # Asegurarse de que no haya frames repetidos debido a la selección de índices
        unique_indices = np.unique(indices)
        adjusted_frames = [frames[i] for i in unique_indices]
        
    elif num_frames < target_count:
        # Si hay menos frames de los necesarios, copiar y distribuir
        indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
        adjusted_frames = [frames[int(i)] for i in indices]
        
    else:
        # Si ya tenemos el número correcto de frames, no hacer nada
        adjusted_frames = frames

    return adjusted_frames

def interpolate_frames(frames, target_frame_count=15):
    current_frame_count = len(frames)
    if current_frame_count == target_frame_count:
        return frames
    
    indices = np.linspace(0, current_frame_count - 1, target_frame_count)
    interpolated_frames = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        interpolated_frame = cv2.addWeighted(frames[lower_idx], 1 - weight, frames[upper_idx], weight, 0)
        interpolated_frames.append(interpolated_frame)
    
    return interpolated_frames

def normalize_frames(frames, target_frame_count=15):
    current_frame_count = len(frames)
    if current_frame_count < target_frame_count:
        return interpolate_frames(frames, target_frame_count)
    elif current_frame_count > target_frame_count:
        step = current_frame_count / target_frame_count
        indices = np.arange(0, current_frame_count, step).astype(int)[:target_frame_count]
        return [frames[i] for i in indices]
    else:
        return frames


# CREATE KEYPOINTS
def extract_keypoints(pose_result, hand_result):
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]  for landmark in (pose_result.pose_landmarks[0] if pose_result.pose_landmarks else [])]).flatten() if pose_result else np.zeros(33*4)
    if hand_result.handedness:
        if hand_result.handedness[0][0].category_name == 'Right':
            rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_result.hand_landmarks[0]]).flatten()
        else:
            rh = np.zeros(21 * 3)

        if hand_result.handedness[0][0].category_name == 'Left' :#hand_result.handedness[0][0].category_name == 'Left':
            lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_result.hand_landmarks[0]]).flatten()
        else:
            lh = np.zeros(21 * 3)
    else:
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)

    #lh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_result.hand_landmarks[0]]).flatten() if hand_result.handedness[0][0].category_name == 'Right' else np.zeros(21*3)
    #rh = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_result.hand_landmarks[0]]).flatten() if hand_result.handedness[1][0].category_name == 'Left' else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

    
# CREATE KEYPOINTS
def extract_keypoints(pose_result, hand_result):
    pose_keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                               for landmark in (pose_result.pose_landmarks[0] if pose_result.pose_landmarks else [])]).flatten() if pose_result else np.zeros(33*4)
    
    hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] 
                               for hand in (hand_result.hand_landmarks if hand_result else []) 
                               for landmark in hand]).flatten() if hand_result else np.zeros(21*3*2)  # Suponiendo dos manos
    if not pose_keypoints.any():
        pose_keypoints = np.zeros(33*4)
    
    if not hand_keypoints.any():
        hand_keypoints = np.zeros(21*3*2)
    else:
        if len(hand_keypoints)<(21*3*2):
            hand_keypoints = np.concatenate([hand_keypoints, np.zeros(21*3)])

    return np.concatenate([pose_keypoints, hand_keypoints])


def extract_keypoints(pose_result, hand_result):
    hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] 
                               for hand in (hand_result.left_hand_landmarks.landmark if hand_result else []) 
                               for landmark in hand]).flatten() if hand_result else np.zeros(21*3)  # Suponiendo dos manos

    return np.concatenate([hand_keypoints])


def extract_keypoints(pose_result, hand_result):
    # Extraer keypoints de la pose
    if pose_result:
        pose_landmarks = pose_result.pose_landmarks
        pose_keypoints = np.array([
            [landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks[0].landmark
        ]).flatten()
    else:
        pose_keypoints = np.zeros(33 * 4)  # 33 puntos * 4 (x, y, z, visibility)

    # Extraer keypoints de la mano
    if hand_result:
        hand_landmarks = hand_result.hand_landmarks
        left_hand_keypoints = np.array([
            [landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[0].landmark
        ]).flatten() if len(hand_landmarks) > 0 else np.zeros(21 * 3)  # 21 puntos * 3 (x, y, z)
        
        right_hand_keypoints = np.array([
            [landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks[1].landmark
        ]).flatten() if len(hand_landmarks) > 1 else np.zeros(21 * 3)  # 21 puntos * 3 (x, y, z)
        
    else:
        left_hand_keypoints = np.zeros(21 * 3)  # 21 puntos * 3 (x, y, z)
        right_hand_keypoints = np.zeros(21 * 3)  # 21 puntos * 3 (x, y, z)

    # Concatenar todos los keypoints
    return np.concatenate([pose_keypoints, left_hand_keypoints, right_hand_keypoints])

def extract_keypoints(pose_result, hand_result):
    pose_keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                               for landmark in (pose_result.pose_landmarks[0] if pose_result.pose_landmarks else [])]).flatten() if pose_result else np.zeros(33*4)
    
    hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] 
                               for hand in (hand_result.hand_landmarks if hand_result else []) 
                               for landmark in hand]).flatten() if hand_result else np.zeros(21*3*2)  # Suponiendo dos manos

    return np.concatenate([pose_keypoints, hand_keypoints])

'''
# CREATE KEYPOINTS
def extract_keypoints(pose_result, hand_result):
    pose_keypoints = np.zeros(33*4)
    hand_keypoints = np.zeros(21*3*2)
    pose_keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_result.pose_landmarks[0]]).flatten() if pose_result.pose_landmarks else np.zeros(33*4)
    hand_keypoints = np.array([[res.x, res.y, res.z] for res in hand_result.hand_landmarks]).flatten() if hand_result.hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose_keypoints, hand_keypoints])
'''

def extract_keypoints(pose_result, hand_result):
    pose_keypoints = np.zeros(33*4)
    hand_keypoints = np.zeros(21*3*2)
    pose_keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                               for landmark in (pose_result.pose_landmarks[0] if pose_result.pose_landmarks else [])]).flatten() if pose_result else np.zeros(33*4)
    
    hand_keypoints = np.array([[landmark.x, landmark.y, landmark.z] 
                               for hand in (hand_result.hand_landmarks if hand_result else []) 
                               for landmark in hand]).flatten() if hand_result else np.zeros(21*3*2)  # Suponiendo dos manos

    return np.concatenate([pose_keypoints, hand_keypoints])