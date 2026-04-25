from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash, session
import cv2
import os
import qrcode
import database as db
from face_engine import FaceAI
from datetime import datetime
from functools import wraps

try:
    import cv2
except:
    cv2 = None

app = Flask(__name__)
app.secret_key = "attendance_secret_key"
face_ai = FaceAI()

@app.context_processor
def inject_now():
    return {'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def generate_qr(student_id):
    qr_dir = os.path.join(app.static_folder, 'qrcodes')
    if not os.path.exists(qr_dir):
        os.makedirs(qr_dir)
    
    qr_filename = f"{student_id}_qr.png"
    qr_path = os.path.join(qr_dir, qr_filename)
    
    # Generate QR with student_id
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(student_id)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(qr_path)
    return f"qrcodes/{qr_filename}"

# --- SECURITY MIDDLEWARE ---

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            flash("Admin access required.", "error")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'student':
            flash("Student login required.", "error")
            return redirect(url_for('student_login'))
        return f(*args, **kwargs)
    return decorated_function

# --- GENERIC ROUTES ---

@app.route('/')
def index():
    return redirect(url_for('student_login'))

# --- ADMIN ROUTES ---

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        user = request.form.get('username')
        pw = request.form.get('password')
        success, admin = db.verify_admin(user, pw)
        if success:
            session['role'] = 'admin'
            session['user'] = 'Administrator'
            return redirect(url_for('admin_dashboard'))
        flash("Invalid Admin Credentials", "error")
    return render_template('admin/login.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    stats = db.get_admin_stats()
    students = db.get_all_students()
    return render_template('admin/dashboard.html', stats=stats, students=students)

@app.route('/admin/add_student', methods=['POST'])
@admin_required
def add_student():
    s_id = request.form.get('student_id')
    name = request.form.get('name')
    pw = request.form.get('password')
    s_class = request.form.get('class')
    
    # NEW: Generate QR Path
    qr_path = generate_qr(s_id)
    
    success, msg = db.add_student(s_id, name, pw, s_class, qr_path)
    flash(msg, "success" if success else "error")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/qrs')
@admin_required
def admin_qrs():
    students = db.get_all_students()
    return render_template('admin/qrs.html', students=students)

@app.route('/admin/delete_student/<s_id>')
@admin_required
def delete_student(s_id):
    db.delete_student(s_id)
    flash(f"Student {s_id} deleted successfully.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/register_face/<s_id>')
@admin_required
def register_face(s_id):
    return render_template('admin/register_face.html', student_id=s_id)

# --- STUDENT ROUTES ---

@app.route('/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        s_id = request.form.get('student_id')
        pw = request.form.get('password')
        success, student = db.verify_student(s_id, pw)
        if success:
            session['role'] = 'student'
            session['user'] = student['name']
            session['id'] = student['student_id']
            return redirect(url_for('student_dashboard'))
        flash("Invalid Student Credentials", "error")
    return render_template('student/login.html')

@app.route('/student/dashboard')
@student_required
def student_dashboard():
    history, perc = db.get_student_history(session['id'])
    return render_template('student/dashboard.html', history=history, percentage=perc)

# --- ATTENDANCE ENGINE ROUTES ---

@app.route('/attendance')
def attendance_page():
    return render_template('attendance_flow.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # AJAX call to check liveness and ID
    # In a real app, we'd stream, but here we'll simulate the check
    cam = cv2.VideoCapture(0)
    success, frame = cam.read()
    cam.release()
    
    if not success:
        return jsonify({"status": "error", "message": "Camera not accessible."})

    blink, head, s_id, conf = face_ai.get_liveness_metrics(frame)
    
    return jsonify({
        "blink": blink,
        "head": head,
        "student_id": s_id,
        "confidence": conf
    })

@app.route('/mark_success', methods=['POST'])
def mark_success():
    s_id = request.json.get('student_id')
    conf = request.json.get('confidence')
    success, msg = db.log_attendance(s_id, confidence=conf, liveness=True)
    return jsonify({"status": "success" if success else "error", "message": msg})

# --- QR SCANNER ROUTES ---

@app.route('/qr_scanner')
def qr_scanner():
    return render_template('qr_scanner.html')

@app.route('/process_qr', methods=['POST'])
def process_qr():
    s_id = request.json.get('student_id')
    # QR doesn't have confidence or liveness the same way face does
    success, msg = db.log_attendance(s_id, confidence=100.0, liveness=True)
    return jsonify({"status": "success" if success else "error", "message": msg})

# --- VIDEO STREAMING ENGINE ---

def gen_frames():
    """Video streaming generator function."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
@admin_required
def capture_face():
    """Enrolls a student by capturing 5 face samples."""
    student_id = request.json.get('student_id')
    camera = cv2.VideoCapture(0)
    frames = []
    for _ in range(5):
        success, frame = camera.read()
        if success:
            frames.append(frame)
    camera.release()
    
    if len(frames) > 0:
        face_ai.register_images(student_id, frames)
        return jsonify({"status": "success", "message": "Face registration successful!"})
    return jsonify({"status": "error", "message": "Could not capture frames from camera."})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('dataset'): os.makedirs('dataset')
    app.run(debug=True, port=5000)
