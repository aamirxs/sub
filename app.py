from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import os
import whisper
import tempfile
from datetime import timedelta
import json
from pathlib import Path
import ffmpeg
from werkzeug.utils import secure_filename
import uuid
from threading import Thread
import math
from threading import Timer
import subprocess
from pathlib import Path
import logging
import traceback

# Set up logging with more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024  # 2000MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['STATIC_FOLDER'] = 'static'
app.static_folder = app.config['STATIC_FOLDER']
app.secret_key = 'your-secret-key-here'  # Required for session

# Ensure required directories exist
for directory in [app.config['UPLOAD_FOLDER'], app.static_folder]:
    os.makedirs(directory, exist_ok=True)

# Global task tracking
tasks = {}

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean'
    
}

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not accessible")
        return False

def validate_video_file(file_path):
    """Validate if the file is a proper video file using ffmpeg."""
    try:
        # Try to get video metadata
        probe = ffmpeg.probe(file_path)
        
        # Check if there's at least one video stream
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        if not video_streams:
            logger.error(f"No video streams found in file: {file_path}")
            return False, "No video content found in file"
            
        # Check if there's at least one audio stream
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if not audio_streams:
            logger.error(f"No audio streams found in file: {file_path}")
            return False, "No audio content found in file"
            
        return True, None
    except ffmpeg.Error as e:
        error_message = str(e.stderr.decode()) if e.stderr else "Unknown FFmpeg error"
        logger.error(f"FFmpeg error while validating file: {error_message}")
        return False, f"Invalid video file: {error_message}"
    except Exception as e:
        logger.error(f"Error validating video file: {str(e)}")
        return False, f"Error validating video file: {str(e)}"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def format_ass_time(seconds):
    """Convert seconds to ASS time format (h:mm:ss.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    centiseconds = int((seconds % 1) * 100)
    seconds = int(seconds)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

def get_video_metadata(video_path):
    """Get video metadata using ffmpeg."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return {
            'width': int(video_info['width']),
            'height': int(video_info['height']),
            'duration': float(probe['format']['duration']),
            'format': probe['format']['format_name']
        }
    except Exception as e:
        return None

def get_time_estimate(file_size):
    """
    Estimate processing time based on file size.
    Uses a basic calculation assuming ~100MB per minute processing speed
    with some overhead for initialization.
    """
    # Base time for initialization (in minutes)
    base_time = 0.5
    
    # Convert file size to MB
    size_in_mb = file_size / (1024 * 1024)
    
    # Calculate processing time
    # Assuming ~100MB per minute processing speed
    processing_time = (size_in_mb / 100) + base_time
    
    # Round up to nearest 0.5 minute
    processing_time = math.ceil(processing_time * 2) / 2
    
    return processing_time

def detect_audio_language(audio_path):
    """Detect the language of the audio using Whisper."""
    try:
        # Load the base model for quick language detection
        model = whisper.load_model("base")
        # Use only first 30 seconds for language detection
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Detect language
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        return detected_lang
    except Exception as e:
        logger.error(f"Error in language detection: {str(e)}")
        return "en"  # Default to English if detection fails

@app.route('/')
def index():
    return render_template('index.html', languages={'auto': 'Auto Detect'} | SUPPORTED_LANGUAGES)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Starting file upload process")
        
        # Check if ffmpeg is installed
        if not check_ffmpeg_installed():
            logger.error("FFmpeg is not installed")
            return jsonify({'error': 'FFmpeg is not installed. Please install FFmpeg to process videos.'}), 500

        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': f'File type not allowed. Supported types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400

        try:
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{str(uuid.uuid4())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            logger.info(f"Saving file to: {filepath}")
            
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the file
            file.save(filepath)
            logger.info("File saved successfully")
            
            # Validate the video file
            logger.info("Validating video file")
            is_valid, error_message = validate_video_file(filepath)
            if not is_valid:
                logger.error(f"Video validation failed: {error_message}")
                os.remove(filepath)
                return jsonify({'error': error_message}), 400

            # Create a new task
            task_id = str(uuid.uuid4())
            language = request.form.get('language', 'auto')
            
            logger.info(f"Creating task {task_id} for file {filename}")
            
            tasks[task_id] = {
                'status': 'starting',
                'progress': 0,
                'filename': filename,
                'filepath': filepath,
                'detected_language': None,
                'error': None,
                'output_files': {}
            }

            # Start processing in background
            thread = Thread(target=process_video_task, args=(task_id, filepath, language))
            thread.daemon = True
            thread.start()
            
            # Calculate processing time estimate
            file_size = os.path.getsize(filepath)
            estimate_minutes = get_time_estimate(file_size)
            time_str = f"{int(estimate_minutes)} minutes" if estimate_minutes >= 1 else "less than a minute"
            
            logger.info(f"File upload successful. Task ID: {task_id}")
            
            return jsonify({
                'task_id': task_id,
                'filename': filename,
                'estimate_minutes': estimate_minutes,
                'estimate_readable': time_str
            })
            
        except Exception as e:
            logger.error(f"Error during file processing: {str(e)}")
            logger.error(traceback.format_exc())
            # Clean up file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up file: {str(cleanup_error)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/estimate_time', methods=['POST'])
def estimate_time():
    if 'file_size' not in request.json:
        return jsonify({'error': 'No file size provided'}), 400
        
    file_size = request.json['file_size']
    estimate = get_time_estimate(file_size)
    
    # Convert to a more readable format
    if estimate < 1:
        time_str = f"{int(estimate * 60)} seconds"
    elif estimate == 1:
        time_str = "1 minute"
    elif estimate.is_integer():
        time_str = f"{int(estimate)} minutes"
    else:
        minutes = int(estimate)
        seconds = int((estimate - minutes) * 60)
        time_str = f"{minutes} minutes {seconds} seconds"
    
    return jsonify({
        'estimate_minutes': estimate,
        'estimate_readable': time_str
    })

def process_video_task(task_id, video_path, language=None):
    """Background task to process video and generate subtitles."""
    try:
        logger.info(f"Starting video processing for task {task_id}")
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['progress'] = 10
        
        # Load the model
        logger.info("Loading Whisper model")
        model = whisper.load_model("base")
        tasks[task_id]['progress'] = 20

        # If language is not specified, detect it
        if not language or language == 'auto':
            logger.info("Starting language detection")
            tasks[task_id]['status'] = 'detecting_language'
            detected_lang = detect_audio_language(video_path)
            language = detected_lang
            tasks[task_id]['detected_language'] = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang.upper())
            logger.info(f"Detected language: {detected_lang}")
        
        tasks[task_id]['progress'] = 30
        tasks[task_id]['status'] = 'transcribing'
        
        logger.info("Starting transcription")
        # Transcribe the video in original language
        result = model.transcribe(
            video_path,
            language=language,
            task="transcribe"
        )
        
        logger.info("Transcription completed")
        tasks[task_id]['progress'] = 60

        # If the detected language is not English, translate to English
        if language != 'en':
            logger.info("Starting translation to English")
            tasks[task_id]['status'] = 'translating'
            
            # Translate using Whisper's translate task
            translation = model.transcribe(
                video_path,
                language=language,  # Source language
                task="translate"    # This will translate to English
            )
            
            # Store both original and translated subtitles
            result_original = result
            result = translation
            tasks[task_id]['has_translation'] = True
            logger.info("Translation completed")
        else:
            tasks[task_id]['has_translation'] = False
            result_original = result

        tasks[task_id]['progress'] = 80
        
        # Generate subtitle files
        logger.info("Generating subtitle files")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)

        formats = {
            'srt': generate_srt,
            'vtt': generate_vtt,
            'ass': generate_ass
        }

        # Generate subtitles in both languages if translation exists
        for format_name, generator_func in formats.items():
            try:
                # Generate English subtitles
                output_path = os.path.join(output_dir, f"{base_name}.en.{format_name}")
                logger.info(f"Generating English {format_name} subtitle: {output_path}")
                generator_func(result['segments'], output_path)
                tasks[task_id]['output_files'][f'en_{format_name}'] = output_path

                # Generate original language subtitles if different from English
                if language != 'en':
                    output_path_orig = os.path.join(output_dir, f"{base_name}.{language}.{format_name}")
                    logger.info(f"Generating {language} {format_name} subtitle: {output_path_orig}")
                    generator_func(result_original['segments'], output_path_orig)
                    tasks[task_id]['output_files'][f'{language}_{format_name}'] = output_path_orig

            except Exception as e:
                logger.error(f"Error generating {format_name} subtitle: {str(e)}")
                tasks[task_id]['error'] = f"Error generating {format_name} subtitle"

        # Update task status
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        logger.info(f"Task {task_id} completed successfully")
        
        # Clean up input video
        try:
            os.remove(video_path)
            logger.info(f"Cleaned up input video: {video_path}")
        except Exception as e:
            logger.error(f"Error cleaning up input video: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in process_video_task: {str(e)}")
        logger.error(traceback.format_exc())
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        # Clean up on error
        try:
            os.remove(video_path)
        except:
            pass

@app.route('/task/<task_id>/status')
def get_task_status(task_id):
    """Get the status of a background task."""
    try:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
            
        task = tasks[task_id]
        return jsonify({
            'status': task.get('status', 'unknown'),
            'progress': task.get('progress', 0),
            'error': task.get('error'),
            'detected_language': task.get('detected_language'),
            'has_translation': task.get('has_translation', False),
            'output_files': task.get('output_files', {})
        })
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<task_id>/<format>')
def download_subtitle(task_id, format):
    """Download generated subtitle file."""
    try:
        if task_id not in tasks:
            return jsonify({'error': 'Task not found'}), 404
            
        task = tasks[task_id]
        if task['status'] != 'completed':
            return jsonify({'error': 'Subtitles not ready yet'}), 400
            
        if format not in task.get('output_files', {}):
            return jsonify({'error': f'No {format} subtitle available'}), 404
            
        subtitle_path = task['output_files'][format]
        if not os.path.exists(subtitle_path):
            return jsonify({'error': 'Subtitle file not found'}), 404
            
        # Get original filename without extension and language code
        original_name = os.path.splitext(task['filename'])[0]
        
        # Extract format extension
        format_ext = format.split('_')[-1]
        
        # Create download filename with language code
        if format.startswith('en_'):
            download_name = f"{original_name}.en.{format_ext}"
        else:
            lang_code = format.split('_')[0]
            download_name = f"{original_name}.{lang_code}.{format_ext}"
        
        return send_file(
            subtitle_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/plain'
        )
        
    except Exception as e:
        logger.error(f"Error in download_subtitle: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def cleanup_files(task_id):
    """Clean up all files associated with a task after a delay."""
    try:
        task = tasks.get(task_id)
        if not task:
            return
            
        # Clean up video file
        if 'filepath' in task and os.path.exists(task['filepath']):
            try:
                os.remove(task['filepath'])
            except Exception as e:
                logger.error(f"Error removing video file: {e}")
                
        # Clean up subtitle files
        if 'output_files' in task:
            for format_type, file_path in task['output_files'].items():
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error removing {format_type} subtitle file: {e}")
                        
        # Remove task from tasks dictionary
        tasks.pop(task_id, None)
        
    except Exception as e:
        logger.error(f"Error in cleanup_files: {str(e)}")
        logger.error(traceback.format_exc())

def generate_srt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = str(timedelta(seconds=segment['start'])).replace('.', ',')[:12]
            end_time = str(timedelta(seconds=segment['end'])).replace('.', ',')[:12]
            f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n")

def generate_vtt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, segment in enumerate(segments, 1):
            start_time = str(timedelta(seconds=segment['start']))[:11].replace(',', '.')
            end_time = str(timedelta(seconds=segment['end']))[:11].replace(',', '.')
            f.write(f"{start_time} --> {end_time}\n{segment['text'].strip()}\n\n")

def generate_ass(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write ASS header
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for segment in segments:
            start_time = format_ass_time(segment['start'])
            end_time = format_ass_time(segment['end'])
            text = segment['text'].strip().replace('\n', '\\N')
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")

if __name__ == '__main__':
    app.run(debug=True)
