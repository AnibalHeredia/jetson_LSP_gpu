import cv2
from mediapipe import Image, ImageFormat
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker ,PoseLandmarkerOptions , PoseLandmarkerResult, HandLandmarker, HandLandmarkerResult, HandLandmarkerOptions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections , pose
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from utils import CvFpsCalc

cvFpsCalc = CvFpsCalc(buffer_len=10)
model_hand_path = "tasks/hand_landmarker.task"
model_pose_path = "tasks/pose_landmarker_full.task"
to_window = None
last_timestamp_ms = 0

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

    if type(detection_result) is PoseLandmarkerResult:
        landmarks_list = detection_result.pose_landmarks
        for idx in range(len(landmarks_list)):
            landmarks = landmarks_list[idx]
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
            ])

            draw_landmarks(
                image,
                landmarks_proto,
                pose.POSE_CONNECTIONS,
                DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    # print("pose landmarker result: {}".format(detection_result))
    to_window = cv2.cvtColor(
        draw_keypoints(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)
    
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_hand_path, delegate=BaseOptions.Delegate.GPU),
    num_hands=2,
    min_hand_detection_confidence = 0.5,
    min_hand_presence_confidence = 0.5,
    min_tracking_confidence = 0.5,
)

'''
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_pose_path, delegate=BaseOptions.Delegate.GPU),
    num_poses=4,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
)
'''

pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_pose_path, delegate=BaseOptions.Delegate.GPU),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=4,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
    result_callback=print_result
)

with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
    vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            display_fps = cvFpsCalc.get()
            success, image = cap.read()
            if not success:
                print("Image capture failed.")
                break

            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            pose_results = pose_landmarker.detect_async(mp_image, timestamp_ms)
            #pose_results = pose_landmarker.detect(mp_image)
            hands_results = hand_landmarker.detect(mp_image)

            draw_keypoints(image, pose_results)
            draw_keypoints(image, hands_results)
            cv2.putText(image, "FPS:" + str(display_fps), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('LSP', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()