import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize the webcam and Mediapipe FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
screen_w, screen_h = pyautogui.size()

# Parameters
blink_threshold = 2  # Number of frames to confirm a blink
blink_duration_threshold = 0.1  # seconds for a single blink
smoothing_factor = 0.5  # Smoothing factor for cursor movement
prev_x, prev_y = 0, 0

def get_eye_aspect_ratio(eye):
    # Calculate the vertical distance between the top and bottom eyelids
    v_dist_1 = np.linalg.norm(np.array((eye[1].x, eye[1].y)) - np.array((eye[5].x, eye[5].y)))
    v_dist_2 = np.linalg.norm(np.array((eye[2].x, eye[2].y)) - np.array((eye[4].x, eye[4].y)))
    # Calculate the horizontal distance between the outer corners of the eyes
    h_dist = np.linalg.norm(np.array((eye[0].x, eye[0].y)) - np.array((eye[3].x, eye[3].y)))
    # Calculate the eye aspect ratio
    ear = (v_dist_1 + v_dist_2) / (2.0 * h_dist)
    return ear

def smooth_cursor_movement(curr_x, curr_y, prev_x, prev_y, smoothing_factor):
    return prev_x + (curr_x - prev_x) * smoothing_factor, prev_y + (curr_y - prev_y) * smoothing_factor

left_eye_blink_counter = 0
right_eye_blink_counter = 0
left_eye_blink_start_time = 0
right_eye_blink_start_time = 0

while True:
    success, frame = cam.read()
    if not success:
        print("Failed to capture frame from camera")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Get eye position
        left_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        right_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        
        left_ear = get_eye_aspect_ratio(left_eye)
        right_ear = get_eye_aspect_ratio(right_eye)
        
        # Draw eye landmarks
        for landmark in left_eye + right_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Get the center of the eyes to move the cursor
        left_center = np.mean([(landmark.x * frame_w, landmark.y * frame_h) for landmark in left_eye], axis=0)
        right_center = np.mean([(landmark.x * frame_w, landmark.y * frame_h) for landmark in right_eye], axis=0)
        center_x, center_y = (left_center[0] + right_center[0]) / 2, (left_center[1] + right_center[1]) / 2
        
        screen_x = screen_w * (center_x / frame_w)
        screen_y = screen_h * (center_y / frame_h)
        
        # Smooth the cursor movement
        curr_x, curr_y = smooth_cursor_movement(screen_x, screen_y, prev_x, prev_y, smoothing_factor)
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y
        
        # Detect left eye blink
        if left_ear < 0.2:  # Adjust threshold as needed
            if left_eye_blink_counter == 0:
                left_eye_blink_start_time = time.time()
            left_eye_blink_counter += 1
        else:
            if left_eye_blink_counter >= blink_threshold and (time.time() - left_eye_blink_start_time) >= blink_duration_threshold:
                # Check if right eye was not blinking
                if right_eye_blink_counter < blink_threshold:
                    pyautogui.click()
                    print("Single Click")
            left_eye_blink_counter = 0
        
        # Detect right eye blink
        if right_ear < 0.2:  # Adjust threshold as needed
            if right_eye_blink_counter == 0:
                right_eye_blink_start_time = time.time()
            right_eye_blink_counter += 1
        else:
            if right_eye_blink_counter >= blink_threshold and (time.time() - right_eye_blink_start_time) >= blink_duration_threshold:
                # Check if left eye was not blinking
                if left_eye_blink_counter < blink_threshold:
                    pyautogui.rightClick()
                    print("Right Click")
            right_eye_blink_counter = 0
        
        # Detect both eyes blink for double click
        if left_eye_blink_counter >= blink_threshold and right_eye_blink_counter >= blink_threshold:
            if (time.time() - left_eye_blink_start_time) >= blink_duration_threshold and (time.time() - right_eye_blink_start_time) >= blink_duration_threshold:
                pyautogui.doubleClick()
                print("Double Click")
                left_eye_blink_counter = 0
                right_eye_blink_counter = 0
    
    cv2.imshow('Eye Controlled Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()