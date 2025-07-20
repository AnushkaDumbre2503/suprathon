# import os
# import zipfile

# # -------------------------
# # Project Structure
# # -------------------------
# folders = [
#     "ai_model",
#     "utils",
#     "templates",
#     "static",
#     "uploads/videos",
#     "uploads/reports"
# ]

# files = {
#     "app.py": """
# from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager, login_user, login_required, logout_user, current_user
# import os
# from datetime import datetime
# from ai_model.analysis import analyze_video
# from utils.pdf_generator import generate_pdf
# from models import db, User, InterviewVideo, ShortlistedCandidate

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret123'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['UPLOAD_FOLDER'] = 'uploads/videos'
# app.config['REPORT_FOLDER'] = 'uploads/reports'

# db.init_app(app)
# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user(user_id))

# @app.route('/')
# def home():
#     return render_template('base.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         role = request.form['role']
#         if User.query.filter_by(username=username).first():
#             flash('Username already exists.')
#             return redirect(url_for('signup'))
#         user = User(username=username, password=password, role=role)
#         db.session.add(user)
#         db.session.commit()
#         flash('Signup successful.')
#         return redirect(url_for('login'))
#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         user = User.query.filter_by(username=username, password=password).first()
#         if user:
#             login_user(user)
#             if user.role == 'admin':
#                 return redirect(url_for('admin_dashboard'))
#             return redirect(url_for('user_dashboard'))
#         else:
#             flash('Invalid credentials')
#     return render_template('login.html')

# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('home'))

# @app.route('/user')
# @login_required
# def user_dashboard():
#     if current_user.role != 'user':
#         return redirect(url_for('login'))
#     videos = InterviewVideo.query.filter_by(user_id=current_user.id).all()
#     return render_template('user_dashboard.html', videos=videos)

# @app.route('/admin')
# @login_required
# def admin_dashboard():
#     if current_user.role != 'admin':
#         return redirect(url_for('login'))
#     candidates = InterviewVideo.query.all()
#     return render_template('hr_dashboard.html', candidates=candidates)

# @app.route('/submit-video', methods=['POST'])
# @login_required
# def submit_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video part'}), 400
#     video = request.files['video']
#     filename = f"{current_user.username}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.webm"
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     video.save(filepath)

#     # AI Analysis
#     rating, report = analyze_video(filepath)
#     pdf_name = filename.replace('.webm', '.pdf')
#     pdf_path = os.path.join(app.config['REPORT_FOLDER'], pdf_name)
#     generate_pdf(report, pdf_path)

#     new_video = InterviewVideo(user_id=current_user.id, video_filename=filename, report_pdf=pdf_name, rating=rating)
#     db.session.add(new_video)
#     db.session.commit()
#     return jsonify({'success': True, 'video': filename})

# @app.route('/shortlist/<int:video_id>')
# @login_required
# def shortlist(video_id):
#     if current_user.role != 'admin':
#         return redirect(url_for('login'))
#     video = InterviewVideo.query.get(video_id)
#     if video:
#         sc = ShortlistedCandidate(user_id=video.user_id, post='Interview', rating=video.rating, status='shortlisted')
#         db.session.add(sc)
#         db.session.commit()
#         flash('Candidate shortlisted.')
#     return redirect(url_for('admin_dashboard'))

# @app.route('/hire/<int:candidate_id>')
# @login_required
# def hire(candidate_id):
#     if current_user.role != 'admin':
#         return redirect(url_for('login'))
#     candidate = ShortlistedCandidate.query.get(candidate_id)
#     if candidate:
#         candidate.status = 'hired'
#         db.session.commit()
#         flash('Candidate hired.')
#     return redirect(url_for('admin_dashboard'))

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)
# """,

#     "models.py": """
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import UserMixin

# db = SQLAlchemy()

# class User(UserMixin, db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(100), unique=True, nullable=False)
#     password = db.Column(db.String(200), nullable=False)
#     role = db.Column(db.String(10), nullable=False)  # 'user' or 'admin'

# class InterviewVideo(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     video_filename = db.Column(db.String(200), nullable=False)
#     report_pdf = db.Column(db.String(200), nullable=True)
#     rating = db.Column(db.Integer, nullable=True)

# class ShortlistedCandidate(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     post = db.Column(db.String(100), nullable=False)
#     rating = db.Column(db.Integer, nullable=True)
#     status = db.Column(db.String(20), default='shortlisted')
# """,

#     "ai_model/analysis.py": """
# from pydub import AudioSegment
# import cv2

# def extract_audio(video_path, audio_path):
#     AudioSegment.converter = "ffmpeg"
#     audio = AudioSegment.from_file(video_path)
#     audio.export(audio_path, format="wav")

# def detect_filler_words(transcription):
#     return []  # Placeholder

# def calculate_speech_rate(transcription):
#     return []  # Placeholder

# def analyze_sentiment(transcription):
#     return [{'start': 0, 'end': 5, 'sentiment': 'POSITIVE', 'score': 0.9}]

# def analyze_body_language(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
#     emotions = []
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % (fps * 3) == 0:
#             emotions.append({'time': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 'emotion': 'neutral'})
#         frame_count += 1
#     cap.release()
#     return emotions

# def generate_report(transcription, filler_words, speech_rate, sentiment, body_language):
#     return {'transcription': 'Demo text', 'filler_words': filler_words, 'speech_rate': speech_rate,
#             'sentiment': sentiment, 'body_language': body_language, 'suggestions': []}

# def analyze_video(video_path):
#     transcription = {'text': 'Sample interview text', 'segments': []}
#     filler = detect_filler_words(transcription)
#     speech = calculate_speech_rate(transcription)
#     sentiment = analyze_sentiment(transcription)
#     body = analyze_body_language(video_path)
#     report = generate_report(transcription, filler, speech, sentiment, body)
#     return 85, str(report)
# """,

#     "utils/pdf_generator.py": """
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas

# def generate_pdf(report_text, pdf_path):
#     c = canvas.Canvas(pdf_path, pagesize=letter)
#     c.setFont("Helvetica", 12)
#     c.drawString(100, 750, "AI Analysis Report")
#     y = 720
#     for line in report_text.split('\\n'):
#         c.drawString(100, y, line)
#         y -= 20
#     c.save()
# """,

#     "templates/base.html": "<h1>Welcome to AI Interview Portal</h1><a href='/login'>Login</a> | <a href='/signup'>Signup</a>",
#     "templates/signup.html": "<form method='POST'>Username: <input name='username'> Password: <input name='password'> Role: <select name='role'><option value='user'>User</option><option value='admin'>Admin</option></select> <button type='submit'>Signup</button></form>",
#     "templates/login.html": "<form method='POST'>Username: <input name='username'> Password: <input name='password'> <button type='submit'>Login</button></form>",
#     "templates/user_dashboard.html": "<h1>User Dashboard</h1> <form action='/submit-video' method='POST' enctype='multipart/form-data'><input type='file' name='video'><button type='submit'>Upload</button></form>",
#     "templates/hr_dashboard.html": "<h1>HR Dashboard</h1> {% for c in candidates %} <p>{{ c.video_filename }} - <a href='/uploads/reports/{{ c.report_pdf }}'>View Report</a> <a href='/shortlist/{{ c.id }}'>Shortlist</a></p> {% endfor %}",
#     "requirements.txt": """
# flask
# flask_sqlalchemy
# flask_login
# werkzeug
# reportlab
# pydub
# ffmpeg-python
# opencv-python
# """
# }

# # -------------------------
# # Create directories
# # -------------------------
# for folder in folders:
#     os.makedirs(folder, exist_ok=True)

# # -------------------------
# # Create files
# # -------------------------
# for file, content in files.items():
#     with open(file, 'w', encoding='utf-8') as f:
#         f.write(content.strip())

# # -------------------------
# # Create ZIP
# # -------------------------
# zip_name = "interview_portal.zip"
# with zipfile.ZipFile(zip_name, 'w') as zipf:
#     for folder in folders:
#         for root, dirs, filenames in os.walk(folder):
#             for filename in filenames:
#                 path = os.path.join(root, filename)
#                 zipf.write(path)
#     for file in files.keys():
#         zipf.write(file)

# print(f"Project generated and zipped as {zip_name}")
