from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Float, nullable=False)
    transcription = db.Column(db.Text, nullable=False)
    analysis = db.Column(db.Text, nullable=False)
    shortlisted = db.Column(db.Boolean, default=False)
    pdf_path = db.Column(db.String(200), nullable=True)
    hired = db.Column(db.Boolean, default=False)  # New column
    user = db.relationship('User', backref=db.backref('reports', lazy=True))