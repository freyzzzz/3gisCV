import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import os
import firebase_admin
from firebase_admin import credentials, storage
import requests  
import sys


cred = credentials.Certificate(
    "C:\\Users")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'three.appspot.com'
})
counter = 3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Constants
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
MOUTH_OUTER_LIP = [61, 291, 81, 13, 311, 402, 14, 178]

EAR_THRESHOLD = 0.235
MAR_THRESHOLD = 0.5
RECORD_DURATION = 5
API_URL = ""  


# Function definitions
def calculate_ear(eye):
    """Calculate Eye Aspect Ratio (EAR)."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def calculate_mar(mouth):
    """Calculate Mouth Aspect Ratio (MAR)."""
    A = distance.euclidean(mouth[2], mouth[6])
    B = distance.euclidean(mouth[3], mouth[5])
    C = distance.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)


def upload_to_firebase(video_path):
    """Upload video to Firebase and return public URL."""
    bucket = storage.bucket()
    blob = bucket.blob(f'videos/{os.path.basename(video_path)}')
    blob.upload_from_filename(video_path)
    blob.make_public()
    return blob.public_url


def send_report_to_server(vehicle_id, driver_id, job_id, report_type, time, location, media, status):
    """Send the report to the server."""
    payload = {
        "vehicle_id": vehicle_id,
        "driver_id": driver_id,
        "job_id": job_id,
        "type": report_type,
        "time": time,
        "location": location,
        "media": media,
        "status": status,
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print("Report sent successfully:", response.json())
    except requests.RequestException as e:
        print(f"Failed to send report: {e}")


# Main program
cap = cv2.VideoCapture(0)

while cap.isOpened():
    frame_count = 0
    closed_eyes_frames = 0
    start_time = time.time()

    video_filename = f'recording_{int(start_time)}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(video_filename, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while time.time() - start_time < RECORD_DURATION:
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Skiped frames for optimization
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                             int(face_landmarks.landmark[i].y * image.shape[0])) for i in LEFT_EYE_LANDMARKS]
                right_eye = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                              int(face_landmarks.landmark[i].y * image.shape[0])) for i in RIGHT_EYE_LANDMARKS]
                mouth = [(int(face_landmarks.landmark[i].x * image.shape[1]),
                          int(face_landmarks.landmark[i].y * image.shape[0])) for i in MOUTH_OUTER_LIP]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                mar = calculate_mar(mouth)

                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    closed_eyes_frames += 1

                cv2.putText(image, f"EAR: {left_ear:.2f}, {right_ear:.2f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"MAR: {mar:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        out.write(image)
        cv2.imshow('Eye Tracking and Recording', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    out.release()
    closed_eyes_percentage = (closed_eyes_frames / frame_count) * 100

    if closed_eyes_percentage > 10:
        video_url = upload_to_firebase(video_filename)
        counter += 1
        send_report_to_server(
            vehicle_id= counter,
            driver_id= counter,  
            job_id= counter,  
            report_type="other",
            time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time)),
            location="Sample Location",  
            media=video_url,
            status="high"
        )
    else:
        os.remove(video_filename)
        print(f"{video_filename} deleted due to low eye closure percentage.")

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
