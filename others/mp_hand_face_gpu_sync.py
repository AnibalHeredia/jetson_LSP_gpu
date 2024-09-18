from constants import *
from func import *

with vision.PoseLandmarker.create_from_options(pose_options) as pose_model, \
     vision.HandLandmarker.create_from_options(hand_options) as hand_model:
    
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image,1)
        if not success:
            print("Image capture failed.")
            break

        small_image = cv2.resize(image, (640, 480))
        pose_result, hand_result = mediapipe_detection(small_image, pose_model, hand_model)
        keypoints = extract_keypoints(pose_result, hand_result)
        annotated_image = draw_landmarks_on_image(small_image, pose_result, hand_result)
        cv2.imshow("LSTM LSP", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
