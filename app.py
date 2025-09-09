import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import nltk
import textstat
from typing import Dict, Any, List
import logging
import re
from datetime import datetime
import threading
import gc

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models
ai_detector = None
tokenizer = None
model = None
model_lock = threading.Lock()

# Model configuration
AI_DETECTION_MODEL = "fakespot-ai/roberta-base-ai-text-detection-v1"
BACKUP_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
CONFIDENCE_THRESHOLD = 0.7

def download_nltk_data():
    """Download required NLTK data"""
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            try:
                nltk.download(data, quiet=True)
                logger.info(f"Downloaded NLTK data: {data}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK data {data}: {e}")

def load_models():
    """Load AI detection models"""
    global ai_detector, tokenizer, model
    
    with model_lock:
        if ai_detector is None:
            try:
                logger.info(f"Loading AI detection model: {AI_DETECTION_MODEL}")
                ai_detector = pipeline(
                    "text-classification",
                    model=AI_DETECTION_MODEL,
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("AI detection model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load primary model: {e}")
                try:
                    logger.info(f"Loading backup model: {BACKUP_MODEL}")
                    ai_detector = pipeline(
                        "text-classification",
                        model=BACKUP_MODEL,
                        return_all_scores=True,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    logger.info("Backup AI detection model loaded successfully")
                except Exception as backup_e:
                    logger.error(f"Failed to load backup model: {backup_e}")
                    raise Exception("Could not load any AI detection model")

def preprocess_text(text: str) -> str:
    """Clean and preprocess text for analysis"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    return text

def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """Split text into chunks for processing"""
    if len(text) <= max_length:
        return [text]
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_length]]

def calculate_confidence(ai_probability: float, text_length: int) -> float:
    """Calculate confidence score based on AI probability and text characteristics"""
    base_confidence = abs(ai_probability - 0.5) * 2
    
    # Adjust confidence based on text length
    if text_length < 50:
        length_factor = 0.7
    elif text_length < 200:
        length_factor = 0.85
    else:
        length_factor = 1.0
    
    confidence = base_confidence * length_factor
    return min(confidence, 1.0)

def generate_explanation(ai_probability: float, features: Dict) -> str:
    """Generate human-readable explanation of the detection result"""
    if ai_probability > 0.7:
        explanation = "This text shows strong indicators of AI generation, including consistent patterns and structure typical of machine-generated content."
    elif ai_probability > 0.3:
        explanation = "This text shows mixed characteristics. It may be human-written with some AI assistance, or AI-generated content that has been edited."
    else:
        explanation = "This text appears to be human-written, showing natural variation in style, structure, and vocabulary typical of human authors."
    
    return explanation

def analyze_text_features(text: str) -> Dict[str, Any]:
    """Analyze basic text features for AI detection"""
    if not text:
        return {}
    
    try:
        features = {
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_sentence_length': 0,
            'readability_score': textstat.flesch_reading_ease(text),
            'unique_words_ratio': 0,
            'character_count': len(text)
        }
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            features['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences)
        
        words = text.lower().split()
        if words:
            unique_words = set(words)
            features['unique_words_ratio'] = len(unique_words) / len(words)
        
        return features
        
    except Exception as e:
        logger.warning(f"Error analyzing text features: {e}")
        return {'word_count': len(text.split()), 'error': str(e)}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NovelCraft AI Detection API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': [
            '/api/detect - AI content detection',
            '/api/enhance - Text enhancement and analysis',
            '/api/health - Health check'
        ]
    })

@app.route('/api/detect', methods=['POST'])
def detect_ai_content():
    """Detect AI-generated content in text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Empty text provided'
            }), 400
        
        if len(text) < 10:
            return jsonify({
                'success': False,
                'error': 'Text too short for reliable analysis (minimum 10 characters)'
            }), 400
        
        start_time = time.time()
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Chunk text for processing
        chunks = chunk_text(processed_text)
        
        # Analyze each chunk
        ai_scores = []
        chunk_weights = []
        
        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue
                
            try:
                results = ai_detector(chunk)
                
                # Extract AI probability (handle different model outputs)
                ai_prob = 0.5  # default
                if isinstance(results[0], list):
                    for result in results[0]:
                        if result['label'].upper() in ['AI', 'FAKE', 'GENERATED', 'MACHINE']:
                            ai_prob = result['score']
                            break
                        elif result['label'].upper() in ['HUMAN', 'REAL', 'AUTHENTIC']:
                            ai_prob = 1 - result['score']
                            break
                else:
                    if results[0]['label'].upper() in ['AI', 'FAKE', 'GENERATED', 'MACHINE']:
                        ai_prob = results[0]['score']
                    elif results[0]['label'].upper() in ['HUMAN', 'REAL', 'AUTHENTIC']:
                        ai_prob = 1 - results[0]['score']
                
                ai_scores.append(ai_prob)
                chunk_weights.append(len(chunk))
                
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")
                ai_scores.append(0.5)
                chunk_weights.append(len(chunk))
        
        # Calculate weighted average
        if ai_scores and chunk_weights:
            total_weight = sum(chunk_weights)
            ai_probability = sum(score * weight for score, weight in zip(ai_scores, chunk_weights)) / total_weight
        else:
            ai_probability = 0.5
        
        # Analyze text features
        features = analyze_text_features(processed_text)
        
        # Calculate confidence
        confidence = calculate_confidence(ai_probability, len(processed_text))
        
        # Determine if AI-generated
        is_ai_generated = ai_probability > CONFIDENCE_THRESHOLD
        
        # Generate explanation
        explanation = generate_explanation(ai_probability, features)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Determine which model was used
        model_used = AI_DETECTION_MODEL if 'fakespot-ai' in str(ai_detector) else BACKUP_MODEL
        
        response = {
            'success': True,
            'ai_probability': round(ai_probability, 4),
            'confidence': round(confidence, 4),
            'is_ai_generated': is_ai_generated,
            'explanation': explanation,
            'features': features,
            'processing_time_ms': processing_time,
            'model_used': model_used,
            'chunks_analyzed': len(chunks),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in AI detection: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    finally:
        # Clean up memory
        gc.collect()

@app.route('/api/enhance', methods=['POST'])
def enhance_text():
    """Basic text enhancement without spacy"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Empty text provided'
            }), 400
        
        word_count = len(text.split())
        if word_count < 10:
            return jsonify({
                'success': False,
                'error': 'Text too short for meaningful enhancement (minimum 10 words)'
            }), 400
        
        start_time = time.time()
        
        # Basic readability analysis
        readability_score = textstat.flesch_reading_ease(text)
        
        # Determine readability level
        if readability_score >= 90:
            readability_level = "Very Easy"
        elif readability_score >= 80:
            readability_level = "Easy"
        elif readability_score >= 70:
            readability_level = "Fairly Easy"
        elif readability_score >= 60:
            readability_level = "Standard"
        elif readability_score >= 50:
            readability_level = "Fairly Difficult"
        elif readability_score >= 30:
            readability_level = "Difficult"
        else:
            readability_level = "Very Difficult"
        
        # Basic suggestions
        suggestions = []
        if readability_score < 30:
            suggestions.append({
                'type': 'readability',
                'priority': 'high',
                'message': 'Text is very difficult to read. Consider shorter sentences and simpler words.',
                'category': 'Readability'
            })
        elif readability_score < 50:
            suggestions.append({
                'type': 'readability',
                'priority': 'medium',
                'message': 'Text is somewhat difficult to read. Consider breaking up long sentences.',
                'category': 'Readability'
            })
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        response = {
            'success': True,
            'text_statistics': {
                'word_count': word_count,
                'character_count': len(text),
                'sentence_count': len([s for s in text.split('.') if s.strip()]),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            },
            'readability_metrics': {
                'flesch_reading_ease': readability_score
            },
            'readability_level': readability_level,
            'suggestions': suggestions,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in text enhancement: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

def initialize_models():
    """Initialize all models on startup"""
    try:
        download_nltk_data()
        load_models()
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

if __name__ == '__main__':
    try:
        # Initialize models on startup
        logger.info("Starting NovelCraft AI Detection API...")
        initialize_models()
        
        # Start the Flask application
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
