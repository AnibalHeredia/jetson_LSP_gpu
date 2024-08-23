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