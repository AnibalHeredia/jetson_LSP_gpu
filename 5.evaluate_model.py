import cv2
from keras.models import load_model
from func import *
from constants import *
#from tensorflow.keras.preprocessing.sequence import pad_sequences

def normalize_keypoints(keypoints, target_keypoints=15):
    num_keypoints = len(keypoints)
    if  num_keypoints != target_keypoints:
        indices = np.linspace(0, num_keypoints - 1, target_keypoints, dtype=int)
        adjusted_keypoints = [keypoints[int(i)] for i in indices]
    else:
        adjusted_keypoints = keypoints

    return adjusted_keypoints
    
def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model = load_model(MODEL_PATH)
    count_frame = 0
    fix_frames = 0
    recording = False
    
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_model, \
        vision.HandLandmarker.create_from_options(hand_options) as hand_model:
        video = cv2.VideoCapture(video_source)
        
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.flip(frame,1)
            window_name ='Traductor LSP'
            #cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
            #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            if not ret: 
                print("Image capture failed.")
                break

            pose_result, hand_result = mediapipe_detection(frame, pose_model, hand_model)
            
            # TODO: colocar un máximo de frames para cada seña,
            # es decir, que traduzca incluso cuando hay mano si se llega a ese máximo.
            if there_hand(hand_result) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(pose_result, hand_result)
                    kp_seq.append(kp_frame)
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                    
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                    if res[np.argmax(res)] > threshold:
                        word_id = word_ids[np.argmax(res)].split('-')[0]
                        
                        sent = words_text.get(word_id)
                        sentence.insert(0, sent)
                
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
            
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                annotated_image = draw_landmarks_on_image(frame, pose_result, hand_result)
                cv2.imshow(window_name, annotated_image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        video.release()
        cv2.destroyAllWindows()
        return sentence
    
if __name__ == "__main__":
    evaluate_model()