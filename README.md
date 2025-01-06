# Video Subtitle Generator

A web application that generates accurate subtitles for uploaded videos using OpenAI's Whisper model.

## Features

- High-accuracy speech-to-text conversion
- Perfect synchronization with video content
- Support for multiple video formats
- Standard SRT subtitle format output
- Mobile-optimized interface
- Easy-to-use drag-and-drop upload

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- Sufficient disk space for video processing

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd subtitle
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload a video file using the drag-and-drop interface or file selector

4. Wait for the subtitle generation process to complete

5. Download the generated SRT file

## Technical Details

- Backend: Python Flask
- Speech Recognition: OpenAI Whisper
- Frontend: HTML5, TailwindCSS, JavaScript
- Video Processing: FFmpeg

## Notes

- Maximum upload size is set to 500MB
- Supported video formats: MP4, AVI, MOV, etc.
- Generated subtitles are in SRT format
- Processing time depends on video length and system capabilities
