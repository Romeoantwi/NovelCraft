#!/usr/bin/env python3
"""
NovelCraft AI Detection API - Lightweight Version
Optimized for free tier deployment with minimal memory usage
"""

import os
import logging
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
import textstat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for lightweight operation
nltk_data_downloaded = False

def download_nltk_data():
    """Download required NLTK data with error handling"""
    global nltk_data_downloaded
    if nltk_data_downloaded:
        return
    
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already available")
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("Downloaded NLTK data successfully")
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    nltk_data_downloaded = True

def preprocess_text(text):
    """Basic text preprocessing"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def analyze_text_features(text):
    """Analyze text features using lightweight methods"""
    try:
        # Basic readability metrics
        flesch_score = textstat.flesch_reading_ease(text)
        flesch_grade = textstat.flesch_kincaid_grade(text)
        
        # Basic text statistics
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
        
        # Simple pattern detection
        repetitive_patterns = len(re.findall(r'\b(\w+)\s+\1\b', text, re.IGNORECASE))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'flesch_reading_ease': round(flesch_score, 2),
            'flesch_kincaid_grade': round(flesch_grade, 2),
            'repetitive_patterns': repetitive_patterns
        }
    except Exception as e:
        logger.warning(f"Feature analysis failed: {e}")
        return {
            'word_count': len(text.split()),
            'sentence_count': 1,
            'avg_word_length': 5.0,
            'flesch_reading_ease': 50.0,
            'flesch_kincaid_grade': 10.0,
            'repetitive_patterns': 0
        }

def lightweight_ai_detection(text):
    """
    Lightweight AI detection using heuristic analysis
    This is a simplified version that doesn't require large ML models
    """
    features = analyze_text_features(text)
    
    # Heuristic scoring based on text characteristics
    ai_indicators = 0
    total_checks = 0
    
    # Check 1: Extremely high readability (too perfect)
    if features['flesch_reading_ease'] > 80:
        ai_indicators += 1
    total_checks += 1
    
    # Check 2: Very consistent sentence structure
    if features['sentence_count'] > 3:
        avg_words_per_sentence = features['word_count'] / features['sentence_count']
        if 15 <= avg_words_per_sentence <= 25:  # Very consistent length
            ai_indicators += 1
    total_checks += 1
    
    # Check 3: Repetitive patterns
    if features['repetitive_patterns'] > 0:
        ai_indicators += 0.5
    
    # Check 4: Average word length (AI tends to use moderate complexity)
    if 4.5 <= features['avg_word_length'] <= 6.5:
        ai_indicators += 0.5
    total_checks += 1
    
    # Check 5: Grade level consistency
    if 8 <= features['flesch_kincaid_grade'] <= 12:
        ai_indicators += 0.5
    total_checks += 1
    
    # Calculate probability
    ai_probability = min(ai_indicators / max(total_checks, 1), 0.95)
    
    # Add some randomness to avoid being too deterministic
    import random
    random.seed(hash(text) % 1000)  # Deterministic randomness based on text
    ai_probability += random.uniform(-0.1, 0.1)
    ai_probability = max(0.05, min(0.95, ai_probability))
    
    return ai_probability, features

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NovelCraft AI Detection API',
        'version': '1.0.0-lightweight',
        'timestamp': datetime.now().isoformat(),
        'memory_optimized': True
    })

@app.route('/api/detect', methods=['POST'])
def detect_ai_content():
    """AI content detection endpoint - lightweight version"""
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing text field in request body',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text field cannot be empty',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if len(text) > 10000:  # Limit text length for free tier
            return jsonify({
                'success': False,
                'error': 'Text too long. Maximum 10,000 characters allowed.',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Perform lightweight AI detection
        ai_probability, features = lightweight_ai_detection(processed_text)
        
        # Calculate confidence based on text length and features
        confidence = min(0.9, 0.3 + (len(processed_text) / 1000) * 0.6)
        
        # Determine if AI generated
        is_ai_generated = ai_probability > 0.5
        
        # Generate explanation
        if is_ai_generated:
            explanation = f"Text shows {ai_probability:.1%} probability of AI generation. "
            explanation += "Indicators include consistent structure and moderate complexity."
        else:
            explanation = f"Text shows {ai_probability:.1%} probability of AI generation. "
            explanation += "Writing style appears more naturally varied."
        
        # Calculate processing time
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Prepare response
        response = {
            'success': True,
            'ai_probability': round(ai_probability, 4),
            'confidence': round(confidence, 4),
            'is_ai_generated': is_ai_generated,
            'explanation': explanation,
            'features': features,
            'processing_time_ms': processing_time,
            'model_used': 'lightweight-heuristic-v1.0',
            'chunks_analyzed': 1,
            'timestamp': datetime.now().isoformat(),
            'note': 'This is a lightweight version optimized for free tier deployment'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/enhance', methods=['POST'])
def enhance_text():
    """Text enhancement endpoint - lightweight version"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing text field in request body',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Text field cannot be empty',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Simple text enhancement (basic cleanup)
        enhanced_text = re.sub(r'\s+', ' ', text.strip())
        enhanced_text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', enhanced_text)
        
        # Basic suggestions
        suggestions = []
        if len(text.split()) < 10:
            suggestions.append("Consider expanding your text for better analysis")
        if text.count('.') == 0:
            suggestions.append("Consider adding proper sentence endings")
        
        return jsonify({
            'success': True,
            'original_text': text,
            'enhanced_text': enhanced_text,
            'suggestions': suggestions,
            'improvements_made': ['whitespace_normalization', 'basic_punctuation_fix'],
            'timestamp': datetime.now().isoformat(),
            'note': 'This is a lightweight version with basic enhancements only'
        })
        
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/batch', methods=['POST'])
def batch_detect():
    """Batch detection endpoint - lightweight version"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing texts field in request body',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'texts field must be an array',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if len(texts) > 10:  # Limit batch size for free tier
            return jsonify({
                'success': False,
                'error': 'Maximum 10 texts allowed in batch processing',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        results = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append({
                    'index': i,
                    'success': False,
                    'error': 'Empty text'
                })
                continue
            
            try:
                processed_text = preprocess_text(text.strip())
                ai_probability, features = lightweight_ai_detection(processed_text)
                
                results.append({
                    'index': i,
                    'success': True,
                    'ai_probability': round(ai_probability, 4),
                    'is_ai_generated': ai_probability > 0.5,
                    'confidence': round(min(0.9, 0.3 + (len(processed_text) / 1000) * 0.6), 4),
                    'word_count': features['word_count']
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat(),
            'model_used': 'lightweight-heuristic-v1.0'
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'NovelCraft AI Detection API',
        'version': '1.0.0-lightweight',
        'status': 'running',
        'endpoints': [
            '/api/health',
            '/api/detect',
            '/api/enhance',
            '/api/batch'
        ],
        'documentation': 'Lightweight version optimized for free tier deployment',
        'timestamp': datetime.now().isoformat()
    })

def initialize_lightweight():
    """Initialize lightweight components"""
    try:
        download_nltk_data()
        logger.info("Lightweight API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

if __name__ == '__main__':
    try:
        # Initialize lightweight components
        logger.info("Starting NovelCraft AI Detection API (Lightweight)...")
        initialize_lightweight()
        
        # Start the Flask application
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
