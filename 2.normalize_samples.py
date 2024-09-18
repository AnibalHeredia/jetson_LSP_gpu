import cv2
import numpy as np
import re
import os
import shutil
from constants import *

def normalize_frames(frames, target_count=15):
    num_frames = len(frames)
    
    if  num_frames != target_count:
        indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
        adjusted_frames = [frames[int(i)] for i in indices]
    else:
        adjusted_frames = frames

    return adjusted_frames

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def read_frames_from_directory(directory):
    frame_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    sorted_files = sorted(frame_files, key=extract_number)
    frames = []
    
    for filename in sorted_files:
        frame = cv2.imread(os.path.join(directory, filename))
        frames.append(frame)
    return frames

def process_directory(word_directory, target_frame_count=15):
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            frames = read_frames_from_directory(sample_directory)
            normalized_frames = normalize_frames(frames, target_frame_count)
            clear_directory(sample_directory)
            save_normalized_frames(sample_directory, normalized_frames)

def save_normalized_frames(directory, frames):
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(directory, f'frame_{i:02}.jpg'), frame)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

if __name__ == "__main__":
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]
    
    for word_id in word_ids:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        if os.path.isdir(word_path):
            print(f'Normalizando frames para "{word_id}"...')
            process_directory(word_path, MODEL_FRAMES)
    