from flask import Flask, request
import numpy as np
import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerResult, FaceLandmarkerOptions, HandLandmarker, HandLandmarkerResult, HandLandmarkerOptions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections, face_mesh
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

model_face_path = "tasks/face_landmarker.task"
model_hand_path = "tasks/hand_landmarker.task"

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_hand_path, delegate=BaseOptions.Delegate.GPU),
    num_hands=2,
)

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_face_path, delegate=BaseOptions.Delegate.GPU),
    num_faces=1,
)

face_landmarker = FaceLandmarker.create_from_options(face_options)
hand_landmarker = HandLandmarker.create_from_options(hand_options)

app = Flask(__name__)

def draw_keypoints(image, detection_result):
    if type(detection_result) is HandLandmarkerResult:
        landmarks_list = detection_result.hand_landmarks
        for idx in range(len(landmarks_list)):
            landmarks = landmarks_list[idx]
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
            ])

            draw_landmarks(
                image,
                landmarks_proto,
                hands_connections.HAND_CONNECTIONS,
                DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
            )

    if type(detection_result) is FaceLandmarkerResult:
        landmarks_list = detection_result.face_landmarks
        for idx in range(len(landmarks_list)):
            landmarks = landmarks_list[idx]
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
            ])

            draw_landmarks(
                image,
                landmarks_proto,
                face_mesh.FACEMESH_CONTOURS, # FACEMESH_TESSELATION (otro)
                DrawingSpec(color=(80, 180, 100), thickness=1, circle_radius=1),
                DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
            )
           
def process_kp(image):
    mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    face_results = face_landmarker.detect(mp_image)
    hands_results = hand_landmarker.detect(mp_image)

    draw_keypoints(image, face_results)
    draw_keypoints(image, hands_results)

    return image

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_kp = process_kp(image)
    image_bytes = cv2.imencode('.jpg', image_kp)[1].tobytes()
    return image_bytes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)