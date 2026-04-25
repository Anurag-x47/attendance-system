import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist

# Path to the downloaded model
MODEL_PATH = 'face_landmarker.task'

# Try face_recognition
HAS_FR = False
try:
    import face_recognition
    HAS_FR = True
except ImportError:
    pass

DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings_v3.pickle"

# --- BLINK DETECTION CONSTANTS ---
EYE_AR_THRESH = 0.22

def calculate_ear(eye_landmarks):
    # eye_landmarks is a list of (x, y) tuples
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

class FaceAI:
    def __init__(self):
        self.known_encodings = []
        self.known_ids = []
        self.load_encodings()
        
        # Initialize MediaPipe Tasks Face Landmarker
        if os.path.exists(MODEL_PATH):
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1)
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe Tasks FaceLandmarker initialized.")
        else:
            self.detector = None
            print(f"Warning: {MODEL_PATH} not found. Anti-spoofing disabled.")

    def load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_ids = data["ids"]

    def train(self):
        if not HAS_FR: return False
        encodings = []
        ids = []
        if not os.path.exists(DATASET_PATH): return False
        
        for student_id in os.listdir(DATASET_PATH):
            student_dir = os.path.join(DATASET_PATH, student_id)
            if not os.path.isdir(student_dir): continue
            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                image = face_recognition.load_image_file(img_path)
                face_boxes = face_recognition.face_locations(image)
                if face_boxes:
                    img_enc = face_recognition.face_encodings(image, face_boxes)[0]
                    encodings.append(img_enc)
                    ids.append(student_id)
        
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": encodings, "ids": ids}, f)
        self.load_encodings()
        return True

    def get_liveness_metrics(self, frame):
        """Returns (blinked, head_moved, student_id, confidence)"""
        if self.detector is None:
            return False, False, None, 0

        # Convert OpenCV BGR to RGB and then to mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        blinked = False
        head_moved = False
        
        if not detection_result.face_landmarks:
            return False, False, None, 0

        # Landmark data
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape
        
        # Left eye landmarks (indices are same as FaceMesh)
        # 33, 160, 158, 133, 153, 144
        l_idx = [33, 160, 158, 133, 153, 144]
        left_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in l_idx]
        
        ear = calculate_ear(left_eye)
        if ear < EYE_AR_THRESH:
            blinked = True

        # Head Pose (simplified yaw check)
        nose = landmarks[4]
        left_center = landmarks[33]
        right_center = landmarks[263]
        
        rel_pos = (nose.x - left_center.x) / (right_center.x - left_center.x) if (right_center.x - left_center.x) != 0 else 0.5
        if rel_pos < 0.35 or rel_pos > 0.65:
            head_moved = True

        # Identification
        student_id = None
        confidence = 0
        if HAS_FR:
            # Optimize: use the same RGB frame but resized
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            # Find faces
            face_locations = face_recognition.face_locations(small_frame)
            if face_locations:
                face_encs = face_recognition.face_encodings(small_frame, face_locations)
                for enc in face_encs:
                    distances = face_recognition.face_distance(self.known_encodings, enc)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        if min_dist < 0.5: # Identification Threshold
                            idx = np.argmin(distances)
                            student_id = self.known_ids[idx]
                            confidence = (1 - min_dist) * 100

        return blinked, head_moved, student_id, confidence

    def register_images(self, student_id, frames):
        s_path = os.path.join(DATASET_PATH, student_id)
        if not os.path.exists(s_path): os.makedirs(s_path)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(s_path, f"cap_{i}.jpg"), f)
        self.train()
        return True
