import pymongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

# MongoDB Connection
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = client["enhanced_attendance_db"]
    students_col = db["students"]
    attendance_col = db["attendance"]
    admins_col = db["admins"]
    unknown_logs_col = db["unknown_logs"]
    
    # Check connection
    client.server_info()
    
    # Create default admin if not exists
    if admins_col.count_documents({}) == 0:
        admins_col.insert_one({
            "username": "admin",
            "password": generate_password_hash("admin123")
        })
    
    print("MongoDB connected successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db = None

# --- AUTH OPERATIONS ---

def verify_admin(username, password):
    admin = admins_col.find_one({"username": username})
    if admin and check_password_hash(admin["password"], password):
        return True, admin
    return False, None

def verify_student(student_id, password):
    student = students_col.find_one({"student_id": student_id})
    if student and check_password_hash(student["password"], password):
        return True, student
    return False, None

# --- STUDENT OPERATIONS ---

def add_student(student_id, name, password, student_class, qr_path):
    if students_col.find_one({"student_id": student_id}):
        return False, "Student ID already exists."
    
    student_data = {
        "student_id": student_id,
        "name": name,
        "password": generate_password_hash(password),
        "class": student_class,
        "images": [],
        "qr_code_path": qr_path,
        "created_at": datetime.now()
    }
    students_col.insert_one(student_data)
    return True, "Student added with QR Code. Now proceed to face registration."

def get_all_students():
    return list(students_col.find({}, {"password": 0}))

def delete_student(student_id):
    students_col.delete_one({"student_id": student_id})
    attendance_col.delete_many({"student_id": student_id})
    # File cleanup handled in app.py or module
    return True

# --- ATTENDANCE OPERATIONS ---

def log_attendance(student_id, confidence=0, liveness=True):
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Check double attendance
    if attendance_col.find_one({"student_id": student_id, "date": today}):
        return False, "Attendance already marked for today."
    
    student = students_col.find_one({"student_id": student_id})
    if not student: return False, "Student not found."
    
    attendance_data = {
        "student_id": student_id,
        "name": student["name"],
        "class": student["class"],
        "date": today,
        "timestamp": datetime.now(),
        "status": "Present",
        "confidence": round(confidence, 2),
        "liveness_verified": liveness
    }
    attendance_col.insert_one(attendance_data)
    return True, f"Success! Attendance marked for {student['name']}."

def log_unknown_attempt():
    unknown_logs_col.insert_one({
        "timestamp": datetime.now(),
        "date": datetime.now().strftime("%Y-%m-%d")
    })

# --- ANALYTICS ---

def get_admin_stats():
    total_students = students_col.count_documents({})
    today = datetime.now().strftime("%Y-%m-%d")
    present_today = attendance_col.count_documents({"date": today})
    
    # Simple monthly trend (last 30 days)
    return {
        "total_students": total_students,
        "present_today": present_today,
        "attendance_rate": round((present_today/total_students*100), 1) if total_students > 0 else 0
    }

def get_student_history(student_id):
    history = list(attendance_col.find({"student_id": student_id}).sort("timestamp", -1))
    total_days = len(history)
    # Mocking a target days for percentage (e.g., 30 days)
    percentage = round((total_days / 30 * 100), 1) if total_days < 30 else 100.0
    return history, percentage
