import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

pose_model_path = "tasks/pose_landmarker_full.task"
hand_model_path = "tasks/hand_landmarker.task"
video_source = "/dev/video0"

num_poses = 1  # Detectar solo una pose para mejorar el rendimiento
num_hands = 2
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5
min_hand_detection_confidence = 0.5
min_hand_presence_confidence = 0.5

to_window = None
last_timestamp_ms = 0

combined_results = {
    "pose": None,
    "hand": None
}

def draw_landmarks_on_image(rgb_image, pose_result, hand_result):
    annotated_image = np.copy(rgb_image)
    
    if pose_result:
        pose_landmarks_list = pose_result.pose_landmarks
        for pose_landmarks in pose_landmarks_list:
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
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    
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
                mp.solutions.drawing_styles.get_default_hand_landmarks_style())
    
    return annotated_image

def print_result(pose_result: vision.PoseLandmarkerResult, hand_result: vision.HandLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    global combined_results

    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    
    if pose_result:
        combined_results["pose"] = pose_result
    if hand_result:
        combined_results["hand"] = hand_result
    
    to_window = cv2.cvtColor(
        draw_landmarks_on_image(output_image.numpy_view(), combined_results["pose"], combined_results["hand"]), cv2.COLOR_RGB2BGR)

base_options_pose = python.BaseOptions(model_asset_path=pose_model_path, delegate=python.BaseOptions.Delegate.GPU)
base_options_hand = python.BaseOptions(model_asset_path=hand_model_path, delegate=python.BaseOptions.Delegate.GPU)

pose_options = vision.PoseLandmarkerOptions(
    base_options=base_options_pose,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    output_segmentation_masks=False,
    result_callback=lambda result, image, timestamp: print_result(result, combined_results["hand"], image, timestamp)
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=num_hands,
    min_hand_detection_confidence=min_hand_detection_confidence,
    min_hand_presence_confidence=min_hand_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    result_callback=lambda result, image, timestamp: print_result(combined_results["pose"], result, image, timestamp)
)

with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
    
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break

        # Reducir la resolución de la imagen
        small_image = cv2.resize(image, (640, 480))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        pose_landmarker.detect_async(mp_image, timestamp_ms)
        hand_landmarker.detect_async(mp_image, timestamp_ms)

        if to_window is not None:
            cv2.imshow("MediaPipe Pose and Hand Landmarks", to_window)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
