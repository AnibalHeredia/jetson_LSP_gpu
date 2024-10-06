from keras.models import load_model
from func import *
from constants import *


def evaluate_model(src=None, threshold=0.8, margin_frame=2, delete_frames=3):
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model = load_model(MODEL_PATH)
    count_frame = 0
    
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_model, \
        vision.HandLandmarker.create_from_options(hand_options) as hand_model:
        video = cv2.VideoCapture(src or video_source)
        
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.flip(frame, 1)
            window_name ='Traductor LSP'

            if not ret: 
                print("Image capture failed.", flush=True)
                break

            pose_result, hand_result = mediapipe_detection(frame, pose_model, hand_model)
            
            # TODO: colocar un máximo de frames para cada seña,
            # es decir, que traduzca incluso cuando hay mano si se llega a ese máximo.
            if there_hand(hand_result):
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(pose_result, hand_result)
                    kp_seq.append(kp_frame)
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    kp_seq = kp_seq[: -delete_frames]
                    #save_frames(kp_seq, evaluate_path)
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                    print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)", flush=True)
                    
                    if res[np.argmax(res)] > threshold:
                        word_id = word_ids[np.argmax(res)].split('-')[0]
                        sent = words_text.get(word_id)
                        sentence.insert(0, sent)
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
    evaluate_path = os.path.join(ROOT_PATH, EVALUATE_PATH)
    create_folder(evaluate_path)
    evaluate_model()
