from flask import Flask, request, jsonify
import whisper
import numpy as np
import os
import json
import tempfile
from datetime import datetime
import soundfile as sf
from pathlib import Path
import re

app = Flask(__name__)
UPLOAD_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_repetitions(text: str, segments: list) -> list:
    """Detect word repetitions in speech."""
    words = text.lower().split()
    repetitions = []
    i = 0
    
    while i < len(words) - 1:
        current_word = words[i]
        count = 1
        start_time = None
        end_time = None
        
        # Look for repetitions
        j = i + 1
        while j < len(words) and words[j] == current_word:
            count += 1
            j += 1
            
        # If repetition found, get timing information
        if count > 1:
            # Find timing in segments
            for segment in segments:
                if current_word in segment['text'].lower():
                    if start_time is None:
                        start_time = segment['start']
                    end_time = segment['end']
            
            repetitions.append({
                'word': current_word,
                'count': count,
                'start': start_time,
                'end': end_time
            })
            i = j
        else:
            i += 1
    
    return repetitions

def analyze_grammar(text: str) -> list:
    """Basic grammar analysis."""
    errors = []
    sentences = re.split('[.!?]+', text)
    
    for sentence in sentences:
        # Check for common errors
        words = sentence.strip().lower().split()
        
        # Double negatives
        negatives = ['not', "n't", 'no', 'never', 'none', 'nothing']
        neg_count = sum(1 for word in words if any(neg in word for neg in negatives))
        if neg_count > 1:
            errors.append({
                'type': 'double_negative',
                'text': sentence.strip(),
                'description': 'Multiple negatives in sentence'
            })
        
        # Subject-verb agreement
        if len(words) >= 2:
            singular_subjects = ['i', 'he', 'she', 'it']
            plural_verbs = ['are', 'were', 'have']
            
            for i, word in enumerate(words[:-1]):
                if word in singular_subjects and words[i+1] in plural_verbs:
                    errors.append({
                        'type': 'subject_verb_agreement',
                        'text': f"{word} {words[i+1]}",
                        'description': 'Subject-verb agreement error'
                    })
        
        # Article usage
        articles = ['a', 'an']
        for i, word in enumerate(words[:-1]):
            if word in articles:
                next_word = words[i+1]
                if word == 'a' and next_word[0] in 'aeiou':
                    errors.append({
                        'type': 'article_usage',
                        'text': f"{word} {next_word}",
                        'description': 'Incorrect article usage'
                    })
                elif word == 'an' and next_word[0] not in 'aeiou':
                    errors.append({
                        'type': 'article_usage',
                        'text': f"{word} {next_word}",
                        'description': 'Incorrect article usage'
                    })
    
    return errors

def analyze_speech(audio_data, sample_rate):
    """Analyze speech and return all results."""
    # Initialize Whisper model
    model = whisper.load_model("medium")
    
    # Transcribe with detailed settings
    result = model.transcribe(
        audio_data,
        language="en",
        word_timestamps=True,
        initial_prompt="Include hesitations, fillers, and repetitions."
    )

    # Extract basic filler words
    filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
    fillers = []
    word_count = 0
    
    for segment in result['segments']:
        words = segment['text'].split()
        word_count += len(words)
        
        for word in words:
            word = word.lower().strip('.,!?')
            if word in filler_words:
                fillers.append({
                    'word': word,
                    'time': segment['start'],
                    'segment_text': segment['text']
                })

    # Detect repetitions
    repetitions = detect_repetitions(result['text'], result['segments'])
    
    # Analyze grammar
    grammar_errors = analyze_grammar(result['text'])

    # Calculate basic metrics
    duration = float(result['segments'][-1]['end']) if result['segments'] else 0
    speech_rate = (word_count / duration * 60) if duration > 0 else 0

    return {
        'transcription': result['text'],
        'duration': duration,
        'word_count': word_count,
        'speech_rate': speech_rate,
        'fillers': fillers,
        'repetitions': repetitions,
        'grammar_errors': grammar_errors,
        'segments': result['segments']
    }

def generate_reports(analysis_result):
    """Generate all reports in a single JSON."""
    return {
        'transcription_report': {
            'text': analysis_result['transcription'],
            'duration': analysis_result['duration'],
            'word_count': analysis_result['word_count']
        },
        'analysis_report': {
            'speech_rate': analysis_result['speech_rate'],
            'filler_count': len(analysis_result['fillers']),
            'repetition_count': len(analysis_result['repetitions']),
            'grammar_error_count': len(analysis_result['grammar_errors']),
            'duration_minutes': analysis_result['duration'] / 60
        },
        'detailed_report': {
            'segments': [
                {
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                }
                for seg in analysis_result['segments']
            ],
            'fillers': analysis_result['fillers'],
            'repetitions': analysis_result['repetitions'],
            'grammar_errors': analysis_result['grammar_errors']
        }
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        file = request.files["audio"]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            
            # Analyze
            analysis_result = analyze_speech(temp_file.name, 16000)
            
            # Generate reports
            reports = generate_reports(analysis_result)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(UPLOAD_FOLDER) / f"analysis_{timestamp}.json"
            
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(reports, f, indent=2)

            return jsonify({
                "status": "success",
                "timestamp": timestamp,
                "reports": reports,
                "file_path": str(output_path)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup
        if 'temp_file' in locals():
            os.unlink(temp_file.name)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)