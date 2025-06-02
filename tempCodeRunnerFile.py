import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize webcam and MediaPipe
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip for natural mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Index finger tip
            x1 = int(landmarks[8].x * screen_width)
            y1 = int(landmarks[8].y * screen_height)

            # Thumb tip
            x2 = int(landmarks[4].x * screen_width)
            y2 = int(landmarks[4].y * screen_height)

            pyautogui.moveTo(x1, y1)

            # Left click gesture (index-thumb pinch)
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < 40:
                pyautogui.click()
                pyautogui.sleep(0.3)  # Delay to avoid multiple clicks

    cv2.imshow("Gesture Controlled Mouse", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
