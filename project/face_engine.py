import cv2
import numpy as np
import os
import pickle

# Try to import face_recognition, fallback to OpenCV LBPH if needed
HAS_FACE_RECOG = False
try:
    import face_recognition
    HAS_FACE_RECOG = True
    print("Using 'face_recognition' library for AI.")
except ImportError:
    print("'face_recognition' not found. Falling back to OpenCV LBPH.")

DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"

def get_face_engine():
    """Return the engine type and tools."""
    return "face_recognition" if HAS_FACE_RECOG else "opencv_lbph"

def register_face(student_id, frames):
    """
    Process frames and save encodings for a student.
    In face_recognition: saves 128D embeddings.
    In LBPH: saves images for training.
    """
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    student_dir = os.path.join(DATASET_PATH, student_id)
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)

    saved_count = 0
    for i, frame in enumerate(frames):
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if HAS_FACE_RECOG:
            # Detect and encode
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    # Save the first encoding found
                    img_path = os.path.join(student_dir, f"{i}.jpg")
                    cv2.imwrite(img_path, frame)
                    saved_count += 1
        else:
            # OpenCV Fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(student_dir, f"{i}.jpg")
                cv2.imwrite(img_path, face_img)
                saved_count += 1

    # Re-train or update encodings
    train_model()
    return saved_count > 0

def train_model():
    """Train the recognizer or update the pickle file."""
    if HAS_FACE_RECOG:
        known_encodings = []
        known_ids = []
        
        for student_id in os.listdir(DATASET_PATH):
            student_dir = os.path.join(DATASET_PATH, student_id)
            if not os.path.isdir(student_dir): continue
            
            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_ids.append(student_id)
        
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": known_encodings, "ids": known_ids}, f)
    else:
        # OpenCV LBPH Training (simplified)
        # We'll do this on the fly or keep a list
        pass

def recognize_face(frame):
    """Recognize a face in a frame."""
    if HAS_FACE_RECOG:
        if not os.path.exists(ENCODINGS_FILE):
            return None
            
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            if True in matches:
                first_match_index = matches.index(True)
                return data["ids"][first_match_index]
    else:
        # Simplified OpenCV Cascade check for 'recognizability' 
        # (True recognition requires LBPH training which is stateful)
        pass
    
    return None
