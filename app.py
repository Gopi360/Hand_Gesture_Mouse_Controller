import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_frame)
    hands = result.multi_hand_landmarks
    frame_h, frame_w, _ = frame.shape

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Index finger
            index_x = int(landmarks[8].x * frame_w)
            index_y = int(landmarks[8].y * frame_h)
            cv2.circle(frame, (index_x, index_y), 8, (0, 0, 255), -1)
            screen_x = screen_w / frame_w * index_x
            screen_y = screen_h / frame_h * index_y
            pyautogui.moveTo(screen_x, screen_y)

            # Thumb
            thumb_x = int(landmarks[4].x * frame_w)
            thumb_y = int(landmarks[4].y * frame_h)
            cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), -1)

            # Middle finger
            mid_x = int(landmarks[12].x * frame_w)
            mid_y = int(landmarks[12].y * frame_h)
            cv2.circle(frame, (mid_x, mid_y), 8, (255, 0, 0), -1)

            # Ring finger
            ring_x = int(landmarks[16].x * frame_w)
            ring_y = int(landmarks[16].y * frame_h)
            cv2.circle(frame, (ring_x, ring_y), 8, (255, 0, 0), -1)

            # Distance for left click (middle + thumb)
            dist_mid_thumb = math.hypot(mid_x - thumb_x, mid_y - thumb_y)
            # print(dist_index_thumb)
            if dist_mid_thumb < 20:
                pyautogui.click()
                pyautogui.sleep(0.5)

            # Distance for right click (ring + thumb)
            dist_ring_thumb = math.hypot(ring_x - thumb_x, ring_y - thumb_y)
            if dist_ring_thumb < 20:
                pyautogui.rightClick()
                pyautogui.sleep(1)

    cv2.imshow("Hand Controlled Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()