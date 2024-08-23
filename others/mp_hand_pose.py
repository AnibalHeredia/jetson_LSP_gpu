import cv2
import mediapipe as mp

# Inicializar los modelos de MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurar la captura de video
cap = cv2.VideoCapture(1)

# Inicializar los modelos con los parámetros de confianza
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
     
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Procesar las detecciones de pose y manos
        pose_results = pose.process(image)
        hands_results = hands.process(image)
        
        # Convertir la imagen de nuevo a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dibujar los puntos clave de pose
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Dibujar los puntos clave de las manos
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Mostrar la imagen procesada
        cv2.imshow('Pose and Hands Detection', image)
        
        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
