import os
import whisper
from transformers import pipeline
import cv2
import numpy as np
from pydub import AudioSegment
import ffmpeg
from datetime import datetime
import mediapipe as mp
import noisereduce as nr
import librosa
import re
from collections import Counter
import logging
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload folder
UPLOAD_FOLDER = os.path.join('static', 'Uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Enhanced model loading
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

try:
    sentiment_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    sentiment_pipeline = None

# MediaPipe initialization
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# Load face detection models
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except Exception as e:
    logger.error(f"Failed to load OpenCV cascades: {e}")
    face_cascade = None
    eye_cascade = None

def analyze_video(video_path, questions):
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_video_{timestamp}.mp4")
    audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{timestamp}.wav")
    
    try:
        logger.info(f"Validating video file")
        validation_result = validate_video_file(video_path)
        if not validation_result['valid']:
            logger.error(f"Video validation failed: {validation_result['error']}")
            raise ValueError(validation_result['error'])

        logger.info("Converting video to MP4")
        try:
            (
                ffmpeg
                .input(video_path)
                .output(temp_path, vcodec='libx264', acodec='aac', format='mp4', crf=23, preset='fast')
                .overwrite_output()
                .run(quiet=True)
            )
            video_path = temp_path
            logger.info(f"Video converted to {video_path}")
        except Exception as e:
            logger.warning(f"Video conversion failed: {e}, proceeding with original file")

        logger.info(f"Extracting audio to {audio_path}")
        extract_audio_enhanced(video_path, audio_path)

        logger.info("Transcribing audio")
        if whisper_model is None:
            raise ValueError("Whisper model not available")
        transcription = whisper_model.transcribe(
            audio_path,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0
        )

        logger.info("Analyzing filler words")
        filler_words = detect_filler_words_enhanced(transcription)

        logger.info("Calculating speech rate")
        speech_rate = calculate_speech_rate_enhanced(transcription)

        logger.info("Analyzing sentiment")
        sentiment = analyze_sentiment_enhanced(transcription)

        logger.info("Analyzing body language")
        body_language = analyze_body_language_enhanced(video_path, validation_result)

        logger.info("Analyzing audio quality")
        audio_quality = analyze_audio_quality(audio_path)

        logger.info("Generating report")
        report = generate_report_enhanced(
            transcription, filler_words, speech_rate,
            sentiment, body_language, audio_quality
        )
        return report

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise
    finally:
        for file_path in [video_path, audio_path, temp_path]:
            try:
                if os.path.exists(file_path) and file_path != video_path:
                    os.remove(file_path)
                    logger.info(f"Removed file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")

def validate_video_file(video_path):
    """Validate video file integrity with detailed error reporting and fallback"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video file with OpenCV")
            return {'valid': False, 'error': 'Cannot open video file'}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            logger.error("Cannot read video frames")
            return {'valid': False, 'error': 'Cannot read video frames'}
        
        if frame_count <= 0 or fps <= 0:
            logger.warning(f"Invalid frame_count ({frame_count}) or fps ({fps}), attempting FFmpeg validation")
            try:
                probe = ffmpeg.probe(video_path)
                logger.info(f"FFmpeg probe: {probe}")
                duration = float(probe['format'].get('duration', 0))
                if duration <= 0:
                    logger.warning("FFmpeg reported invalid duration, using fallback values")
                    frame_count = 30
                    fps = 30
                else:
                    frame_count = max(1, int(duration * max(fps, 30)))
                    fps = max(fps, 30)
            except Exception as e:
                logger.warning(f"FFmpeg validation failed: {str(e)}, checking file existence and size")
                if not os.path.exists(video_path):
                    return {'valid': False, 'error': 'Video file does not exist'}
                file_size = os.path.getsize(video_path)
                if file_size < 1000:
                    return {'valid': False, 'error': 'Video file is too small or corrupted'}
                logger.info(f"File exists, size: {file_size} bytes, using fallback values")
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                cap.release()
                if frame_count <= 0:
                    frame_count = 30
                fps = 30
                logger.info(f"Fallback frame count: {frame_count}, fps: {fps}")
        
        return {'valid': True, 'frame_count': frame_count, 'fps': fps, 'width': width, 'height': height}
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {'valid': False, 'error': f'Validation error: {str(e)}'}

def extract_audio_enhanced(video_path, audio_path):
    """Enhanced audio extraction with better preprocessing"""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.normalize()
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.max(np.abs(samples))
        reduced_noise = nr.reduce_noise(y=samples, sr=16000, prop_decrease=0.8)
        reduced_noise = (reduced_noise * 32767).astype(np.int16)
        processed_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1
        )
        processed_audio.export(audio_path, format="wav")
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        raise

def detect_filler_words_enhanced(transcription):
    """Enhanced filler word detection with deduplication and case normalization"""
    fillers = {
        'basic': ['um', 'uh', 'er', 'ah', 'hmm', 'well'],
        'discourse': ['like', 'you know', 'i mean', 'sort of', 'kind of'],
        'hesitation': ['basically', 'actually', 'literally', 'really', 'just'],
        'repetitive': ['so', 'but', 'yeah', 'okay', 'right']
    }
    professional_terms = {
        'technical': ['machine learning', 'data science', 'software development', 
                     'artificial intelligence', 'deep learning', 'neural network'],
        'business': ['team collaboration', 'project management', 'stakeholder engagement',
                    'strategic planning', 'market analysis', 'client relations']
    }
    all_fillers = []
    for category in fillers.values():
        all_fillers.extend(category)
    
    occurrences = []
    seen = set()
    total_words = 0
    word_sequences = []
    
    for segment in transcription['segments']:
        text = segment['text'].lower().strip()
        words = re.findall(r'\b\w+\b', text)
        total_words += len(words)
        for i in range(len(words)):
            for length in [2, 3]:
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    word_sequences.append(phrase)
            if 'words' in segment:
                for word_info in segment['words']:
                    word = word_info['word'].lower().strip()
                    key = (word, word_info['start'], word_info['end'])
                    if word in all_fillers and key not in seen:
                        is_meaningful = any(term in text for term_list in professional_terms.values() 
                                          for term in term_list if word in term.lower())
                        if not is_meaningful:
                            occurrences.append({
                                'word': word_info['word'],
                                'start': word_info['start'],
                                'end': word_info['end'],
                                'confidence': word_info.get('confidence', 1.0),
                                'category': next((cat for cat, words in fillers.items() if word in words), 'other')
                            })
                            seen.add(key)
    
    phrase_counts = Counter(word_sequences)
    repetitive_phrases = []
    for phrase, count in phrase_counts.most_common():
        if count >= 2 and len(phrase.split()) >= 2:
            is_professional = any(prof_phrase in phrase.lower() 
                                for term_list in professional_terms.values() 
                                for prof_phrase in term_list)
            if not is_professional and phrase not in all_fillers:
                repetitive_phrases.append({
                    'phrase': phrase,
                    'count': count,
                    'severity': 'medium' if count >= 4 else 'low'
                })
    
    filler_count = len(occurrences)
    repetitive_count = sum(item['count'] - 1 for item in repetitive_phrases)
    total_issues = filler_count + repetitive_count
    filler_ratio = total_issues / total_words if total_words > 0 else 0
    base_score = 5
    if filler_ratio > 0.18:
        base_score = 2
    elif filler_ratio > 0.12:
        base_score = 3
    elif filler_ratio > 0.06:
        base_score = 4
    category_penalties = {
        'basic': 0.15,
        'discourse': 0.1,
        'hesitation': 0.05,
        'repetitive': 0.05
    }
    category_counts = Counter(occ['category'] for occ in occurrences)
    penalty = sum(count * category_penalties.get(cat, 0.05) for cat, count in category_counts.items())
    final_score = max(2, round(base_score - penalty))
    return {
        'occurrences': occurrences,
        'repetitive_phrases': repetitive_phrases,
        'total_issues': total_issues,
        'filler_count': filler_count,
        'repetitive_count': repetitive_count,
        'ratio': round(filler_ratio * 100, 2),
        'score': final_score,
        'category_breakdown': dict(category_counts)
    }

def calculate_speech_rate_enhanced(transcription):
    """Enhanced speech rate calculation with relaxed thresholds and pause timelines"""
    segment_rates = []
    pauses = []
    total_words = 0
    total_duration = 0
    last_end = 0
    silence_threshold = 0.5
    
    for i, segment in enumerate(transcription['segments']):
        words = len(segment['text'].split())
        start = segment['start']
        end = segment['end']
        duration = end - start
        if duration < 0.5:
            continue
        if i > 0:
            pause_duration = start - last_end
            if pause_duration > silence_threshold:
                pauses.append({
                    'duration': pause_duration,
                    'type': 'long' if pause_duration > 2.5 else 'medium' if pause_duration > 1.5 else 'short',
                    'start': last_end,
                    'end': start
                })
        if duration > 0:
            rate = words / (duration / 60)
            segment_rates.append({
                'start': start,
                'end': end,
                'rate': rate,
                'words': words,
                'duration': duration
            })
            total_words += words
            total_duration += duration
        last_end = end
    
    if total_duration == 0:
        logger.warning("No valid speech segments found")
        return {
            'error': 'No valid speech segments found',
            'score': 3,
            'pauses': []
        }
    
    avg_rate = total_words / (total_duration / 60)
    rates_only = [seg['rate'] for seg in segment_rates]
    rate_std = np.std(rates_only) if len(rates_only) >= 2 else 0
    pause_counts = Counter(p['type'] for p in pauses)
    total_pause_time = sum(p['duration'] for p in pauses)
    optimal_min, optimal_max = 110, 190
    acceptable_min, acceptable_max = 90, 210
    if optimal_min <= avg_rate <= optimal_max:
        rate_score = 5
    elif acceptable_min <= avg_rate <= acceptable_max:
        rate_score = 4
    elif 70 <= avg_rate <= 230:
        rate_score = 3
    else:
        rate_score = 2
    consistency_score = 5
    if rate_std > 55:
        consistency_score = 3
    elif rate_std > 35:
        consistency_score = 4
    pause_score = 5
    excessive_pauses = pause_counts.get('long', 0)
    total_pauses = len(pauses)
    if excessive_pauses > 5 or total_pauses > 25:
        pause_score = 3
    elif excessive_pauses > 3 or total_pauses > 15:
        pause_score = 4
    final_score = (rate_score * 0.6 + consistency_score * 0.2 + pause_score * 0.2)
    return {
        'average_rate': round(avg_rate, 1),
        'rate_std': round(rate_std, 1),
        'segment_rates': segment_rates,
        'pauses': pauses,
        'pause_counts': dict(pause_counts),
        'total_pause_time': round(total_pause_time, 2),
        'score': max(2, round(final_score))
    }

def analyze_sentiment_enhanced(transcription):
    """Enhanced sentiment analysis with stricter penalties for negative tones including sadness"""
    if sentiment_pipeline is None:
        logger.warning("Sentiment analysis not available")
        return {'score': 3, 'positive_ratio': 50, 'neutral_ratio': 30, 'sadness_ratio': 20, 'negative_ratio': 0, 'segment_emotions': []}
    
    segment_emotions = []
    total_segments = 0
    positive_segments = 0
    neutral_segments = 0
    sadness_segments = 0
    negative_segments = 0
    
    for segment in transcription['segments']:
        text = segment['text'].strip()
        start = segment['start']
        end = segment['end']
        if len(text) < 10:
            continue
        total_segments += 1
        try:
            emotions = sentiment_pipeline(text)
            dominant_emotion = max(emotions[0], key=lambda x: x['score'])
            segment_emotions.append({
                'text': text,
                'start': start,
                'end': end,
                'dominant_emotion': dominant_emotion
            })
            label = dominant_emotion['label']
            score = dominant_emotion['score']
            if label in ['joy', 'confidence', 'optimism'] and score > 0.5:
                positive_segments += 1
            elif label == 'neutral' and score > 0.5:
                neutral_segments += 1
            elif label == 'sadness' and score > 0.5:
                sadness_segments += 1
            elif label in ['anger', 'fear', 'disgust'] and score > 0.5:
                negative_segments += 1
        except Exception as e:
            logger.warning(f"Emotion analysis failed for segment: {e}")
            continue
    
    if total_segments == 0:
        logger.warning("No valid segments for sentiment analysis")
        return {'score': 3, 'positive_ratio': 50, 'neutral_ratio': 30, 'sadness_ratio': 20, 'negative_ratio': 0, 'segment_emotions': []}
    
    positive_ratio = (positive_segments / total_segments) * 100 if total_segments > 0 else 50
    neutral_ratio = (neutral_segments / total_segments) * 100 if total_segments > 0 else 30
    sadness_ratio = (sadness_segments / total_segments) * 100 if total_segments > 0 else 20
    negative_ratio = (negative_segments / total_segments) * 100 if total_segments > 0 else 0
    
    score = 3
    if positive_ratio >= 50 or (positive_ratio + neutral_ratio) >= 70:
        score = 5
    elif neutral_ratio >= 50 and positive_ratio >= 20:
        score = 4
    elif positive_ratio >= 30 or (positive_ratio + neutral_ratio) >= 50:
        score = 4
    if negative_ratio >= 20 or sadness_ratio >= 20:
        score = 1
    elif negative_ratio >= 10 or sadness_ratio >= 10:
        score = 2
    
    return {
        'positive_ratio': round(positive_ratio, 1),
        'neutral_ratio': round(neutral_ratio, 1),
        'sadness_ratio': round(sadness_ratio, 1),
        'negative_ratio': round(negative_ratio, 1),
        'score': score,
        'segment_emotions': segment_emotions
    }

def analyze_body_language_enhanced(video_path, validation_result):
    """Enhanced body language analysis with relaxed thresholds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file")
        return {'confidence': {'score': 3}, 'eye_contact': {'score': 3}}
    
    fps = max(1, int(validation_result.get('fps', 24)))
    total_frames = max(1, int(validation_result.get('frame_count', 1)))
    
    frame_count = 0
    sampled_frames = 0
    pose_detections = 0
    face_detections = 0
    eye_contact_frames = 0
    sample_interval = fps
    
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            sampled_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = mp_pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                pose_detections += 1
            face_results = mp_face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                face_detections += 1
                for face_landmarks in face_results.multi_face_landmarks:
                    eye_center_x = (face_landmarks.landmark[468].x + face_landmarks.landmark[473].x) / 2
                    gaze_offset = abs(eye_center_x - 0.5)
                    if gaze_offset < 0.2:
                        eye_contact_frames += 1
                    break
        frame_count += 1
    cap.release()
    
    if sampled_frames == 0:
        logger.warning("No frames could be processed")
        return {'confidence': {'score': 3}, 'eye_contact': {'score': 3}}
    
    pose_detection_ratio = pose_detections / sampled_frames if sampled_frames > 0 else 0
    eye_contact_ratio = eye_contact_frames / sampled_frames if sampled_frames > 0 else 0
    confidence_score = 5
    if pose_detection_ratio < 0.75:
        confidence_score = 4
    elif pose_detection_ratio < 0.55:
        confidence_score = 3
    eye_contact_score = 5
    if eye_contact_ratio < 0.65:
        eye_contact_score = 4
    elif eye_contact_ratio < 0.45:
        eye_contact_score = 3
    return {
        'confidence': {
            'pose_detection_ratio': round(pose_detection_ratio * 100, 1),
            'score': confidence_score
        },
        'eye_contact': {
            'eye_contact_ratio': round(eye_contact_ratio * 100, 1),
            'score': eye_contact_score
        }
    }

def analyze_audio_quality(audio_path):
    """Enhanced audio quality analysis with relaxed thresholds"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        rms_energy = librosa.feature.rms(y=y)[0]
        signal_power = np.mean(y**2)
        noise_samples = np.concatenate([y[:int(0.5*sr)], y[-int(0.5*sr):]])
        noise_power = np.mean(noise_samples**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        quality_score = 5
        if snr < 7:
            quality_score = 3
        elif snr < 10:
            quality_score = 4
        return {
            'snr': round(snr, 2),
            'score': quality_score
        }
    except Exception as e:
        logger.error(f"Audio quality analysis failed: {e}")
        return {'score': 3, 'snr': 15}

def generate_report_enhanced(transcription, filler_words, speech_rate, sentiment, body_language, audio_quality):
    """Generate report with balanced scoring and improvement timelines"""
    weights = {
        'filler_words': 0.1,
        'speech_rate': 0.3,
        'sentiment': 0.2,
        'confidence': 0.2,
        'eye_contact': 0.1,
        'audio_quality': 0.1
    }

    scores = {
        'filler_words': filler_words.get('score', 3),
        'speech_rate': speech_rate.get('score', 3),
        'sentiment': sentiment.get('score', 3),
        'confidence': body_language.get('confidence', {}).get('score', 3),
        'eye_contact': body_language.get('eye_contact', {}).get('score', 3),
        'audio_quality': audio_quality.get('score', 3)
    }

    weighted_score = sum(scores[category] * weights[category] for category in weights)
    score_distribution = list(scores.values())
    min_score = min(score_distribution)
    high_scores = sum(1 for score in score_distribution if score >= 4)

    if sentiment.get('positive_ratio', 0) == 0 and sentiment.get('sadness_ratio', 100) >= 80:
        star_rating = 1
    elif min_score <= 1:
        star_rating = 1
    elif weighted_score >= 4.2 and min_score >= 3:
        star_rating = 5
    elif weighted_score >= 3.3 and min_score >= 3:
        star_rating = 4
    elif weighted_score >= 2.5:
        star_rating = 3
    elif weighted_score >= 2.0:
        star_rating = 2
    else:
        star_rating = 1

    strengths = [f"{category.replace('_', ' ').title()} ({score}/5)" for category, score in scores.items() if score >= 4]
    areas_for_improvement = [f"{category.replace('_', ' ').title()} ({score}/5)" for category, score in scores.items() if score <= 3]

    improvement_timelines = {
        'filler_words': [],
        'pauses': [],
        'sentiment': []
    }

    if filler_words.get('score', 3) <= 3:
        grouped_fillers = {}
        for occ in filler_words.get('occurrences', []):
            key = (occ['word'].lower(), occ['start'], occ['end'])
            if key not in grouped_fillers:
                grouped_fillers[key] = {
                    'word': occ['word'],
                    'start': occ['start'],
                    'end': occ['end'],
                    'count': 0
                }
            grouped_fillers[key]['count'] += 1
        improvement_timelines['filler_words'] = [
            {
                'start': data['start'],
                'end': data['end'],
                'word': data['word'],
                'count': data['count'],
                'recommendation': f"Avoid using '{data['word']}' as a filler (used {data['count']} time{'s' if data['count'] > 1 else ''}); try pausing instead."
            } for data in grouped_fillers.values()
        ]

    if speech_rate.get('score', 3) <= 3:
        improvement_timelines['pauses'] = [
            {
                'start': pause['start'],
                'end': pause['end'],
                'duration': pause['duration'],
                'recommendation': f"Reduce {pause['type']} pause ({pause['duration']:.1f}s); transition smoothly."
            } for pause in speech_rate.get('pauses', []) if pause['type'] in ['long', 'medium']
        ]

    if sentiment.get('score', 3) <= 3:
        improvement_timelines['sentiment'] = [
            {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'emotion': seg['dominant_emotion']['label'],
                'recommendation': (
                    f"Your authenticity is powerful; consider balancing with positive reflections." 
                    if seg['dominant_emotion']['label'] == 'sadness' 
                    else f"Replace negative phrasing '{seg['text']}' with constructive language."
                )
            } for seg in sentiment.get('segment_emotions', [])
            if seg['dominant_emotion']['label'] in ['anger', 'fear', 'disgust', 'sadness']
        ]

    return {
        'overall_performance': {
            'star_rating': star_rating,
            'weighted_score': round(weighted_score, 2),
            'score_distribution': scores,
            'strengths': strengths,
            'areas_for_improvement': areas_for_improvement
        },
        'transcription': {
            'text': transcription.get('text', ''),
            'language': transcription.get('language', 'unknown'),
            'duration': sum(seg.get('end', 0) - seg.get('start', 0) for seg in transcription.get('segments', []))
        },
        'speech_fluency': {
            'filler_words': {
                'ratio_percentage': filler_words.get('ratio', 0),
                'score': filler_words.get('score', 3)
            },
            'speech_rate': {
                'average_wpm': speech_rate.get('average_rate', 150),
                'score': speech_rate.get('score', 3)
            }
        },
        'emotional_delivery': {
            'overall_tone': {
                'positive_ratio': sentiment.get('positive_ratio', 50),
                'neutral_ratio': sentiment.get('neutral_ratio', 30),
                'sadness_ratio': sentiment.get('sadness_ratio', 20),
                'negative_ratio': sentiment.get('negative_ratio', 0),
                'score': sentiment.get('score', 3)
            }
        },
        'physical_presence': {
            'posture_confidence': {
                'stability_metrics': {
                    'pose_detection_ratio': body_language.get('confidence', {}).get('pose_detection_ratio', 0)
                },
                'score': body_language.get('confidence', {}).get('score', 3)
            },
            'eye_contact': {
                'contact_metrics': {
                    'eye_contact_ratio': body_language.get('eye_contact', {}).get('eye_contact_ratio', 0)
                },
                'score': body_language.get('eye_contact', {}).get('score', 3)
            }
        },
        'technical_quality': {
            'audio': {
                'quality_metrics': {
                    'signal_to_noise_ratio': audio_quality.get('snr', 15)
                },
                'score': audio_quality.get('score', 3)
            }
        },
        'improvement_timelines': improvement_timelines
    }