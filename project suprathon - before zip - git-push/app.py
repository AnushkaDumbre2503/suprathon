from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, Response
from models import db, User, Report
from ai_model.analysis import analyze_video
from flask_migrate import Migrate
import os
import random
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import logging
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['REPORT_FOLDER'] = 'Uploads/reports'

# Create upload and report folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

db.init_app(app)
migrate = Migrate(app, db)

with app.app_context():
    db.create_all()

def generate_pdf(report_data, user_id, post_applied):
    """
    Generate a PDF report from analysis data.
    """
    filename = f"report_user_{user_id}_{post_applied.replace(' ', '_')}.pdf"
    filepath = os.path.join(app.config['REPORT_FOLDER'], filename)
    
    try:
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        story.append(Paragraph("Interview Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Add user and post info
        story.append(Paragraph(f"User ID: {user_id}", styles['Normal']))
        story.append(Paragraph(f"Post Applied: {post_applied}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Add overall rating
        story.append(Paragraph(f"Overall Rating: {report_data['overall_performance']['star_rating']}/5", styles['Normal']))
        story.append(Spacer(1, 12))

        # Add transcription
        story.append(Paragraph("Transcription:", styles['Heading2']))
        story.append(Paragraph(report_data['transcription']['text'], styles['Normal']))
        story.append(Spacer(1, 12))

        # Add analysis details
        story.append(Paragraph("Analysis Details:", styles['Heading2']))
        story.append(Paragraph(f"Sentiment Score: {report_data['emotional_delivery']['overall_tone']['score']}/5", styles['Normal']))
        story.append(Paragraph(f"Body Language Score: {round((report_data['physical_presence']['posture_confidence']['score'] + report_data['physical_presence']['eye_contact']['score']) * 10)}/100", styles['Normal']))
        story.append(Spacer(1, 12))

        # Add improvement timelines (deduplicated by recommendation)
        story.append(Paragraph("Improvement Timelines:", styles['Heading2']))
        timelines = report_data['improvement_timelines']
        all_timelines = [
            item for sublist in [timelines.get('filler_words', []), timelines.get('pauses', []), timelines.get('sentiment', [])]
            for item in sublist if 'recommendation' in item and 'start' in item
        ]
        unique_timelines = {}
        for item in all_timelines:
            rec = item['recommendation']
            if rec not in unique_timelines or item['start'] < unique_timelines[rec]['start']:
                unique_timelines[rec] = item
        timeline_text = "\n".join(
            f"{item['recommendation']} (at {item['start']}s)" for item in unique_timelines.values()
        )
        story.append(Paragraph(timeline_text or "No specific improvements identified.", styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        logger.info(f"PDF generated successfully at {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to generate PDF: {str(e)}")
        raise

@app.route('/')
def landing():
    return render_template("index.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        if User.query.filter_by(username=username).first():
            flash("Username already exists.")
            return redirect(url_for('signup'))
        user = User(username=username, password=password, role=role)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if not user:
            flash("Invalid credentials.")
            return redirect(url_for('login'))
        session['user_id'] = user.id
        session['role'] = user.role
        return redirect(url_for('hr_dashboard' if user.role == 'hr' else 'interviewee_dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/interviewee')
def interviewee_dashboard():
    if session.get('role') != 'interviewee':
        return redirect(url_for('login'))
    questions = [
        "Tell us about a challenging project you worked on and how you solved it.",
        "Where do you see yourself in the next 5 years?"
    ]
    session['interview_questions'] = questions
    return render_template(
        'interviewee_dashboard.html',
        questions=questions
    )

def gen_frames():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    if session.get('role') != 'interviewee':
        return '', 403
    if 'interview_questions' not in session:
        flash("Please start the interview first to generate questions.")
        return redirect(url_for('interviewee_dashboard'))

    try:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No video file provided"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # Generate unique filename
        timestamp = time.strftime('%Y%m%d%H%M%S')
        video_filename = f"user_{session['user_id']}_interview_{timestamp}.webm"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        # Save the uploaded video
        video_file.save(video_path)
        logger.info(f"Video saved to {video_path}")

        # Verify file exists
        if not os.path.exists(video_path):
            return jsonify({"status": "error", "message": "Failed to save video file"}), 500

        # Analyze the video
        questions = session['interview_questions']
        post_applied = request.form.get('post_applied', 'Interview')  # Get post from form
        report_data = analyze_video(video_path, questions=questions)

        # Deduplicate improvement timelines
        timelines = report_data['improvement_timelines']
        all_timelines = [
            item for sublist in [timelines.get('filler_words', []), timelines.get('pauses', []), timelines.get('sentiment', [])]
            for item in sublist if 'recommendation' in item and 'start' in item
        ]
        unique_timelines = {}
        for item in all_timelines:
            rec = item['recommendation']
            if rec not in unique_timelines or item['start'] < unique_timelines[rec]['start']:
                unique_timelines[rec] = item
        report_data['improvement_timelines'] = {
            'filler_words': [item for item in timelines.get('filler_words', []) if item['recommendation'] in unique_timelines],
            'pauses': [item for item in timelines.get('pauses', []) if item['recommendation'] in unique_timelines],
            'sentiment': [item for item in timelines.get('sentiment', []) if item['recommendation'] in unique_timelines]
        }

        pdf_path = generate_pdf(report_data, session['user_id'], post_applied)

        # Store analysis in database
        analysis_text = (
            f"Sentiment: {report_data['emotional_delivery']['overall_tone']['score']}/5\n"
            f"Body Language: {round((report_data['physical_presence']['posture_confidence']['score'] + report_data['physical_presence']['eye_contact']['score']) * 10)}/100\n"
            f"Recommendations: {', '.join(rec.get('recommendation', '') for rec in (report_data['improvement_timelines'].get('filler_words', []) + report_data['improvement_timelines'].get('pauses', []) + report_data['improvement_timelines'].get('sentiment', [])) if 'recommendation' in rec)}"
        )
        report = Report(
            user_id=session['user_id'],
            post=post_applied,
            score=report_data['overall_performance']['star_rating'],
            transcription=report_data['transcription']['text'],
            analysis=analysis_text,
            pdf_path=pdf_path
        )
        db.session.add(report)
        db.session.commit()

        # Clean up the saved video file
        try:
            os.remove(video_path)
            logger.info(f"Cleaned up video file {video_path}")
        except Exception as e:
            logger.warning(f"Failed to delete video file {video_path}: {e}")

        # Return full report data
        return jsonify({
            "status": "ok",
            "report": report_data
        })
    except Exception as e:
        logger.error(f"Error during interview processing: {str(e)}")
        flash(f"Error processing interview: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/hr')
def hr_dashboard():
    if session.get('role') != 'hr':
        return redirect(url_for('login'))
    post = request.args.get('post')
    min_score = request.args.get('min_score', 0, type=float)
    query = Report.query
    if post:
        query = query.filter(Report.post.ilike(f"%{post}%"))
    query = query.filter(Report.score >= min_score)

    reports = query.all()
    return render_template('hr_dashboard.html', reports=reports)

@app.route('/shortlist/<int:report_id>', methods=['POST'])
def shortlist(report_id):
    if session.get('role') != 'hr':
        return '', 403
    try:
        report = Report.query.get_or_404(report_id)
        report.shortlisted = True
        db.session.commit()
        logger.info(f"Shortlisted report ID {report_id}")
        return redirect(url_for('shortlisted'))
    except Exception as e:
        logger.error(f"Error shortlisting report {report_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/remove/<int:report_id>', methods=['POST'])
def remove(report_id):
    if session.get('role') != 'hr':
        return '', 403
    try:
        report = Report.query.get_or_404(report_id)
        db.session.delete(report)
        db.session.commit()
        logger.info(f"Removed report ID {report_id} from database")
        return redirect(url_for('hr_dashboard'))
    except Exception as e:
        logger.error(f"Error removing report {report_id}: {str(e)}")
        flash(f"Error removing candidate: {str(e)}")
        return redirect(url_for('hr_dashboard')), 500

@app.route('/shortlisted')
def shortlisted():
    if session.get('role') != 'hr':
        return redirect(url_for('login'))
    try:
        reports = Report.query.filter_by(shortlisted=True).all()
        logger.info(f"Loaded {len(reports)} shortlisted reports")
        return render_template('shortlisted.html', reports=reports)
    except Exception as e:
        logger.error(f"Error loading shortlisted page: {str(e)}")
        flash(f"Error loading shortlisted candidates: {str(e)}")
        return redirect(url_for('hr_dashboard'))

@app.route('/hire/<int:report_id>')
def hire(report_id):
    if session.get('role') != 'hr':
        return '', 403
    try:
        report = Report.query.get_or_404(report_id)
        report.hired = True
        db.session.commit()
        logger.info(f"Hired report ID {report_id}")
        return redirect(url_for('shortlisted'))
    except Exception as e:
        logger.error(f"Error hiring report {report_id}: {str(e)}")
        flash(f"Error hiring candidate: {str(e)}")
        return redirect(url_for('shortlisted'))

@app.route('/report_pdf/<int:report_id>')
def serve_pdf(report_id):
    if session.get('role') != 'hr':
        return '', 403
    try:
        report = Report.query.get_or_404(report_id)
        if not report.pdf_path or not os.path.exists(report.pdf_path):
            flash("PDF report not found.")
            return redirect(url_for('hr_dashboard'))
        logger.info(f"Serving PDF for report ID {report_id}")
        return send_file(report.pdf_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error serving PDF for report {report_id}: {str(e)}")
        flash(f"Error downloading PDF: {str(e)}")
        return redirect(url_for('hr_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)