import cv2
import mediapipe as mp
import csv
import os
import argparse
import time
import numpy as np

# Set up MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    """Normalize the landmarks relative to the wrist (landmark 0)"""
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    
    normalized = []
    for lm in landmarks:
        normalized.append(lm.x - base_x)
        normalized.append(lm.y - base_y)
    
    return normalized

def collect_data(gesture_name, num_samples=500):
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    filename = f"data/{gesture_name}.csv"
    
    # Initialize the CSV file
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if new file (42 columns for 21 X, Y hand landmarks)
        if not file_exists:
            header = []
            for i in range(21):
                header.extend([f"lm_{i}_x", f"lm_{i}_y"])
            header.append("label")
            writer.writerow(header)

    cap = cv2.VideoCapture(0)
    
    print(f"Collecting data for gesture: {gesture_name}")
    print("Press 's' to start collecting.")
    print("Press 'q' to quit.")
    
    # Wait for the user to press 's' to start
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        cv2.putText(image, f"Press 's' to start saving {gesture_name} gesture", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Data Collection', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print("Starting collection...")
            time.sleep(1) # small pause
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
            
    # Start capturing
    count = 0
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        
        while cap.isOpened() and count < num_samples:
            success, image = cap.read()
            if not success:
                break
            
            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract and normalize coordinates
                    coords = normalize_landmarks(hand_landmarks.landmark)
                    
                    # Append label
                    coords.append(gesture_name)
                    
                    # Save to CSV
                    with open(filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(coords)
                    
                    count += 1
                        
            # Display stats
            cv2.putText(image, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Samples: {count}/{num_samples}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Hand Gesture Data Collection', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete! Saved {count} samples to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hand gesture dataset")
    parser.add_argument("--gesture", type=str, required=True, help="Name of the gesture (e.g., ThumbsUp, Peace, Stop)")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to collect")
    args = parser.parse_args()
    
    collect_data(args.gesture, args.samples)