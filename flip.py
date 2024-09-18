import os
import cv2
import re
from constants import *
import shutil

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def read_frames_from_directory(directory):
    frame_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    sorted_files = sorted(frame_files, key=extract_number)
    frames = []
    
    for filename in sorted_files:
        frame = cv2.imread(os.path.join(directory, filename))
        frame = cv2.flip(frame, 1)
        frames.append(frame)
    return frames

def save_normalized_frames(directory, frames):
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(directory, f'{i:02}.jpg'), frame)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree

def flip_images_in_directory(input_directory):
    for filename in os.listdir(input_directory):
        sample_directory = os.path.join(input_directory, filename)
        if os.path.isdir(input_directory):
            frames = read_frames_from_directory(sample_directory)
            clear_directory(sample_directory)
            save_normalized_frames(sample_directory, frames)

if __name__ == "__main__":
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]

    for word_id in word_ids:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        if os.path.isdir(word_path):
            print(f'flip frame para "{word_id}"...')
            flip_images_in_directory(word_path)
