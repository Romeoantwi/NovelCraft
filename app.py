import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import nltk
import spacy
import textstat
import language_tool_python
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
nlp = None
grammar_tool = None
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
            except Exception as e:
                logger.warning(f"Failed to download {data}: {e}")

def load_ai_detection_model():
    """Load the AI detection model from Hugging Face"""
    global ai_detector, tokenizer, model
    
    try:
        logger.info(f"Loading AI detection model: {AI_DETECTION_MODEL}")
        
        # Try primary model first
        try:
            ai_detector = pipeline(
                "text-classification",
                model=AI_DETECTION_MODEL,
                tokenizer=AI_DETECTION_MODEL,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            logger.info(f"Successfully loaded primary model: {AI_DETECTION_MODEL}")
        except Exception as e:
            logger.warning(f"Primary model failed, trying backup: {e}")
            # Try backup model
            ai_detector = pipeline(
                "text-classification",
                model=BACKUP_MODEL,
                tokenizer=BACKUP_MODEL,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            logger.info(f"Successfully loaded backup model: {BACKUP_MODEL}")
            
    except Exception as e:
        logger.error(f"Failed to load AI detection models: {e}")
        # Create a fallback detector
        ai_detector = create_fallback_detector()

def create_fallback_detector():
    """Create a simple rule-based fallback detector"""
    class FallbackDetector:
        def __call__(self, text):
            # Simple heuristics for AI detection
            ai_indicators = [
                r'\bas an ai\b', r'\bi am an ai\b', r'\bi\'m an ai\b',
                r'\bas a language model\b', r'\bas an artificial intelligence\b',
                r'\bi don\'t have personal\b', r'\bi cannot\b', r'\bi can\'t\b',
                r'\bi\'m not able to\b', r'\bi don\'t have the ability\b'
            ]
            
            text_lower = text.lower()
            ai_score = 0
            
            for pattern in ai_indicators:
                if re.search(pattern, text_lower):
                    ai_score += 0.3
            
            # Length and repetition heuristics
            if len(text.split()) > 100:
                sentences = text.split('.')
                if len(set(sentences)) < len(sentences) * 0.8:  # High repetition
                    ai_score += 0.2
            
            ai_score = min(ai_score, 1.0)
            human_score = 1.0 - ai_score
            
            return [[
                {'label': 'HUMAN', 'score': human_score},
                {'label': 'AI', 'score': ai_score}
            ]]
    
    logger.warning("Using fallback AI detector")
    return FallbackDetector()

def load_models():
    """Load all required models and tools"""
    global nlp, grammar_tool
    
    logger.info("Loading NLP models...")
    
    # Load AI detection model
    load_ai_detection_model()
    
    # Load spaCy for NLP tasks
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model")
    except OSError:
        logger.info("Downloading spaCy model...")
        try:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load spaCy: {e}")
            nlp = None
    
    # Initialize grammar tool with error handling
    try:
        grammar_tool = language_tool_python.LanguageTool('en-US')
        logger.info("Loaded LanguageTool")
    except Exception as e:
        logger.error(f"Failed to load LanguageTool: {e}")
        grammar_tool = None
    
    logger.info("Model loading completed")

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'ai_detector': ai_detector is not None,
            'nlp': nlp is not None,
            'grammar_tool': grammar_tool is not None
        },
        'gpu_available': torch.cuda.is_available(),
        'version': '1.0.0'
    })

def preprocess_text(text: str) -> str:
    """Preprocess text for analysis"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Ensure minimum length for reliable detection
    if len(text.split()) < 10:
        raise ValueError("Text must contain at least 10 words for reliable analysis")
    
    return text

def calculate_text_features(text: str) -> Dict[str, Any]:
    """Calculate various text features for analysis"""
    features = {}
    
    try:
        # Basic statistics
        words = text.split()
        sentences = text.split('.')
        
        features.update({
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / max(1, len([s for s in sentences if s.strip()]))
        })
        
        # Readability metrics
        if len(text) > 100:  # Only calculate for longer texts
            features.update({
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'difficult_words': textstat.difficult_words(text)
            })
        
        # NLP features if spaCy is available
        if nlp:
            doc = nlp(text)
            features.update({
                'named_entities': len(doc.ents),
                'pos_diversity': len(set(token.pos_ for token in doc)),
                'dependency_diversity': len(set(token.dep_ for token in doc))
            })
    
    except Exception as e:
        logger.warning(f"Error calculating text features: {e}")
    
    return features

# AI Detection endpoint
@app.route('/api/detect', methods=['POST'])
def detect_ai_content():
    """Detect AI-generated content using state-of-the-art models"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        try:
            processed_text = preprocess_text(text)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Get AI detection results
        with model_lock:
            if ai_detector is None:
                return jsonify({'error': 'AI detection model not loaded'}), 500
            
            # Split long texts into chunks for better accuracy
            max_length = 512
            chunks = []
            words = processed_text.split()
            
            for i in range(0, len(words), max_length):
                chunk = ' '.join(words[i:i + max_length])
                if len(chunk.split()) >= 10:  # Only process meaningful chunks
                    chunks.append(chunk)
            
            if not chunks:
                chunks = [processed_text]  # Fallback to original text
            
            # Analyze each chunk
            ai_scores = []
            for chunk in chunks:
                try:
                    results = ai_detector(chunk)
                    # Handle different model output formats
                    if isinstance(results, list) and len(results) > 0:
                        if isinstance(results[0], list):
                            # Format: [[{'label': 'AI', 'score': 0.8}, {'label': 'HUMAN', 'score': 0.2}]]
                            scores = results[0]
                        else:
                            # Format: [{'label': 'AI', 'score': 0.8}, {'label': 'HUMAN', 'score': 0.2}]
                            scores = results
                        
                        # Find AI score
                        ai_score = 0.5  # Default neutral score
                        for score_dict in scores:
                            label = score_dict.get('label', '').upper()
                            if 'AI' in label or 'FAKE' in label or 'GENERATED' in label:
                                ai_score = score_dict.get('score', 0.5)
                                break
                            elif 'HUMAN' in label or 'REAL' in label:
                                ai_score = 1.0 - score_dict.get('score', 0.5)
                                break
                        
                        ai_scores.append(ai_score)
                    else:
                        ai_scores.append(0.5)  # Neutral score for failed chunks
                        
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    ai_scores.append(0.5)
            
            # Calculate weighted average (longer chunks have more weight)
            if ai_scores:
                chunk_weights = [len(chunk.split()) for chunk in chunks]
                total_weight = sum(chunk_weights)
                ai_probability = sum(score * weight for score, weight in zip(ai_scores, chunk_weights)) / total_weight
            else:
                ai_probability = 0.5
        
        # Calculate text features
        features = calculate_text_features(processed_text)
        
        # Determine verdict with confidence
        is_ai_generated = ai_probability > CONFIDENCE_THRESHOLD
        confidence = abs(ai_probability - 0.5) * 2  # Convert to 0-1 confidence scale
        
        # Generate explanation
        if ai_probability > 0.8:
            explanation = "High probability of AI generation detected. Text shows patterns typical of machine-generated content."
        elif ai_probability > 0.6:
            explanation = "Moderate probability of AI generation. Some characteristics suggest machine involvement."
        elif ai_probability < 0.3:
            explanation = "Strong indicators of human authorship. Text shows natural human writing patterns."
        elif ai_probability < 0.5:
            explanation = "Likely human-written with some uncertainty. Mixed signals in writing patterns."
        else:
            explanation = "Uncertain classification. Text shows mixed characteristics."
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        response = {
            'success': True,
            'ai_probability': round(ai_probability, 4),
            'confidence': round(confidence, 4),
            'is_ai_generated': is_ai_generated,
            'verdict': 'AI-Generated' if is_ai_generated else 'Human-Written',
            'explanation': explanation,
            'features': features,
            'processing_time_ms': processing_time,
            'model_used': AI_DETECTION_MODEL if 'fakespot-ai' in str(ai_detector) else BACKUP_MODEL,
            'chunks_analyzed': len(chunks),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in detect_ai_content: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500

# Text enhancement endpoint
@app.route('/api/enhance', methods=['POST'])
def enhance_text():
    """Enhance text with advanced grammar and style suggestions"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '').strip()
        enhancement_level = data.get('level', 'balanced')  # minimal, balanced, aggressive
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text.split()) < 3:
            return jsonify({'error': 'Text must contain at least 3 words'}), 400
        
        suggestions = []
        readability_metrics = {}
        style_suggestions = []
        
        # Grammar checking with LanguageTool
        if grammar_tool:
            try:
                matches = grammar_tool.check(text)
                
                # Process grammar suggestions
                for match in matches[:20]:  # Limit suggestions
                    suggestion = {
                        'type': 'grammar',
                        'message': match.message,
                        'original': text[match.offset:match.offset + match.errorLength],
                        'replacements': [r for r in match.replacements[:3] if r],
                        'offset': match.offset,
                        'length': match.errorLength,
                        'category': match.category.name if hasattr(match, 'category') else 'GRAMMAR',
                        'rule_id': match.ruleId,
                        'confidence': 0.8  # Default confidence for grammar rules
                    }
                    
                    # Add context
                    context_start = max(0, match.offset - 20)
                    context_end = min(len(text), match.offset + match.errorLength + 20)
                    suggestion['context'] = text[context_start:context_end]
                    
                    suggestions.append(suggestion)
                    
            except Exception as e:
                logger.warning(f"Grammar checking failed: {e}")
        
        # Style and readability analysis
        try:
            if len(text) > 50:  # Only for longer texts
                readability_metrics = {
                    'flesch_reading_ease': textstat.flesch_reading_ease(text),
                    'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                    'gunning_fog': textstat.gunning_fog(text),
                    'automated_readability_index': textstat.automated_readability_index(text),
                    'coleman_liau_index': textstat.coleman_liau_index(text),
                    'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                    'difficult_words': textstat.difficult_words(text),
                    'reading_time_minutes': textstat.reading_time(text, ms_per_char=14.69)
                }
                
                # Generate style suggestions based on readability
                if readability_metrics['flesch_reading_ease'] < 30:
                    style_suggestions.append({
                        'type': 'style',
                        'message': 'Text is very difficult to read. Consider using shorter sentences and simpler words.',
                        'category': 'READABILITY',
                        'confidence': 0.9
                    })
                elif readability_metrics['flesch_reading_ease'] < 50:
                    style_suggestions.append({
                        'type': 'style',
                        'message': 'Text is fairly difficult to read. Consider simplifying some sentences.',
                        'category': 'READABILITY',
                        'confidence': 0.7
                    })
                
                if readability_metrics['gunning_fog'] > 16:
                    style_suggestions.append({
                        'type': 'style',
                        'message': 'Text has a high fog index. Consider reducing complex sentences.',
                        'category': 'COMPLEXITY',
                        'confidence': 0.8
                    })
        
        except Exception as e:
            logger.warning(f"Readability analysis failed: {e}")
        
        # Advanced NLP analysis with spaCy
        if nlp:
            try:
                doc = nlp(text)
                
                # Analyze sentence structure
                long_sentences = [sent for sent in doc.sents if len(sent.text.split()) > 25]
                if long_sentences and enhancement_level in ['balanced', 'aggressive']:
                    for sent in long_sentences[:3]:
                        style_suggestions.append({
                            'type': 'style',
                            'message': f'Consider breaking this long sentence into shorter ones.',
                            'original': sent.text,
                            'category': 'SENTENCE_LENGTH',
                            'confidence': 0.6
                        })
                
                # Check for passive voice (simplified detection)
                passive_patterns = ['was', 'were', 'been', 'being']
                sentences = [sent.text for sent in doc.sents]
                for i, sentence in enumerate(sentences):
                    if any(pattern in sentence.lower() for pattern in passive_patterns):
                        if enhancement_level == 'aggressive':
                            style_suggestions.append({
                                'type': 'style',
                                'message': 'Consider using active voice instead of passive voice.',
                                'original': sentence,
                                'category': 'VOICE',
                                'confidence': 0.5
                            })
                
                # Word repetition analysis
                words = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                repeated_words = [(word, count) for word, count in word_freq.items() 
                                if count > 3 and len(word) > 4]
                
                if repeated_words and enhancement_level in ['balanced', 'aggressive']:
                    for word, count in repeated_words[:3]:
                        style_suggestions.append({
                            'type': 'style',
                            'message': f'The word "{word}" is repeated {count} times. Consider using synonyms.',
                            'category': 'REPETITION',
                            'confidence': 0.6
                        })
                
            except Exception as e:
                logger.warning(f"NLP analysis failed: {e}")
        
        # Combine all suggestions
        all_suggestions = suggestions + style_suggestions
        
        # Calculate text statistics
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        stats = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len([p.strip() for p in text.split('\n\n') if p.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(word.lower() for word in words)),
            'lexical_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0
        }
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        response = {
            'success': True,
            'suggestions': all_suggestions,
            'suggestion_count': len(all_suggestions),
            'readability': readability_metrics,
            'stats': stats,
            'enhancement_level': enhancement_level,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in enhance_text: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500

# Batch processing endpoint for large documents
@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple texts in batch for efficiency"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        texts = data.get('texts', [])
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'No texts array provided'}), 400
        
        if len(texts) > 10:
            return jsonify({'error': 'Maximum 10 texts per batch'}), 400
        
        results = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append({'error': 'Empty text', 'index': i})
                continue
            
            try:
                # Process each text (simplified for batch)
                processed_text = preprocess_text(text.strip())
                
                # AI detection
                with model_lock:
                    if ai_detector:
                        detection_results = ai_detector(processed_text[:512])  # Limit for batch
                        ai_score = 0.5
                        if detection_results and len(detection_results) > 0:
                            scores = detection_results[0] if isinstance(detection_results[0], list) else detection_results
                            for score_dict in scores:
                                label = score_dict.get('label', '').upper()
                                if 'AI' in label or 'FAKE' in label:
                                    ai_score = score_dict.get('score', 0.5)
                                    break
                    else:
                        ai_score = 0.5
                
                # Basic features
                features = calculate_text_features(processed_text)
                
                results.append({
                    'index': i,
                    'ai_probability': round(ai_score, 4),
                    'is_ai_generated': ai_score > CONFIDENCE_THRESHOLD,
                    'word_count': features.get('word_count', 0),
                    'readability_score': features.get('flesch_reading_ease', 0)
                })
                
            except Exception as e:
                results.append({'error': str(e), 'index': i})
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_count': len([r for r in results if 'error' not in r]),
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in batch_analyze: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'processing_time_ms': round((time.time() - start_time) * 1000, 2)
        }), 500

if __name__ == '__main__':
    # Download required NLTK data
    download_nltk_data()
    
    # Load models
    load_models()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
