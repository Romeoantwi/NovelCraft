import os
import sys
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models
nlp = None
grammar_tool = None
model_lock = threading.Lock()

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
    """Load NLP models for text enhancement"""
    global nlp, grammar_tool
    
    with model_lock:
        # Load spaCy model for NLP tasks
        if nlp is None:
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found, downloading...")
                os.system("python -m spacy download en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model downloaded and loaded")

        # Initialize grammar tool
        if grammar_tool is None:
            try:
                grammar_tool = language_tool_python.LanguageTool('en-US')
                logger.info("Grammar tool initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize grammar tool: {e}")

def analyze_readability(text: str) -> Dict[str, Any]:
    """Analyze text readability metrics"""
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid().grade_level(text),
            'gunning_fog': textstat.gunning_fog(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'reading_time_minutes': textstat.reading_time(text, ms_per_char=14.69)
        }
    except Exception as e:
        logger.warning(f"Error calculating readability: {e}")
        return {'error': str(e)}

def get_grammar_suggestions(text: str) -> List[Dict[str, Any]]:
    """Get grammar and style suggestions"""
    suggestions = []
    
    if not grammar_tool:
        return suggestions
    
    try:
        matches = grammar_tool.check(text)
        
        for match in matches[:10]:  # Limit to top 10 suggestions
            suggestion = {
                'offset': match.offset,
                'length': match.errorLength,
                'message': match.message,
                'replacements': [r for r in match.replacements[:3]],  # Top 3 replacements
                'rule_id': match.ruleId,
                'category': match.category,
                'context': match.context,
                'original_text': text[match.offset:match.offset + match.errorLength]
            }
            suggestions.append(suggestion)
            
    except Exception as e:
        logger.warning(f"Error getting grammar suggestions: {e}")
        suggestions.append({
            'message': f'Grammar check unavailable: {str(e)}',
            'type': 'error'
        })
    
    return suggestions

def analyze_style_and_tone(text: str) -> Dict[str, Any]:
    """Analyze writing style and tone"""
    if not nlp:
        return {'error': 'NLP model not available'}
    
    try:
        doc = nlp(text)
        
        # Count different types of words and structures
        word_counts = {
            'total_words': len([token for token in doc if token.is_alpha]),
            'unique_words': len(set([token.lemma_.lower() for token in doc if token.is_alpha])),
            'long_words': len([token for token in doc if len(token.text) > 6]),
            'short_sentences': 0,
            'long_sentences': 0,
            'avg_sentence_length': 0
        }
        
        sentences = list(doc.sents)
        if sentences:
            sentence_lengths = [len([token for token in sent if token.is_alpha]) for sent in sentences]
            word_counts['avg_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
            word_counts['short_sentences'] = len([l for l in sentence_lengths if l < 10])
            word_counts['long_sentences'] = len([l for l in sentence_lengths if l > 20])
        
        # POS tag analysis
        pos_counts = {}
        for token in doc:
            if token.is_alpha:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        # Calculate ratios
        total_words = word_counts['total_words']
        style_metrics = {
            'lexical_diversity': word_counts['unique_words'] / max(total_words, 1),
            'complex_word_ratio': word_counts['long_words'] / max(total_words, 1),
            'avg_sentence_length': word_counts['avg_sentence_length'],
            'sentence_variety': {
                'short_sentences': word_counts['short_sentences'],
                'long_sentences': word_counts['long_sentences'],
                'total_sentences': len(sentences)
            }
        }
        
        # Tone indicators (simplified)
        tone_indicators = {
            'formal_indicators': len([token for token in doc if len(token.text) > 8]),
            'personal_pronouns': len([token for token in doc if token.lemma_.lower() in ['i', 'you', 'we', 'they']]),
            'passive_voice_indicators': len([token for token in doc if token.dep_ == 'auxpass']),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!')
        }
        
        return {
            'style_metrics': style_metrics,
            'tone_indicators': tone_indicators,
            'pos_distribution': pos_counts
        }
        
    except Exception as e:
        logger.warning(f"Error analyzing style and tone: {e}")
        return {'error': str(e)}

def generate_enhancement_suggestions(text: str, readability: Dict, style: Dict, grammar: List) -> List[Dict[str, Any]]:
    """Generate comprehensive text enhancement suggestions"""
    suggestions = []
    
    # Readability suggestions
    if readability.get('flesch_reading_ease', 0) < 30:
        suggestions.append({
            'type': 'readability',
            'priority': 'high',
            'message': 'Text is very difficult to read. Consider shorter sentences and simpler words.',
            'category': 'Readability'
        })
    elif readability.get('flesch_reading_ease', 0) < 50:
        suggestions.append({
            'type': 'readability',
            'priority': 'medium',
            'message': 'Text is somewhat difficult to read. Consider breaking up long sentences.',
            'category': 'Readability'
        })
    
    # Style suggestions
    if style.get('style_metrics', {}).get('avg_sentence_length', 0) > 25:
        suggestions.append({
            'type': 'style',
            'priority': 'medium',
            'message': 'Average sentence length is quite long. Consider varying sentence structure.',
            'category': 'Style'
        })
    
    if style.get('style_metrics', {}).get('lexical_diversity', 0) < 0.4:
        suggestions.append({
            'type': 'style',
            'priority': 'medium',
            'message': 'Consider using more varied vocabulary to improve engagement.',
            'category': 'Style'
        })
    
    # Grammar suggestions (from LanguageTool)
    for gram_suggestion in grammar[:5]:  # Top 5 grammar suggestions
        suggestions.append({
            'type': 'grammar',
            'priority': 'high',
            'message': gram_suggestion.get('message', ''),
            'category': 'Grammar',
            'original_text': gram_suggestion.get('original_text', ''),
            'replacements': gram_suggestion.get('replacements', [])
        })
    
    return suggestions

def handler(request):
    """Vercel serverless function handler for text enhancement"""
    try:
        # Initialize models on first request
        load_models()
        download_nltk_data()
        
        if request.method != 'POST':
            return jsonify({'error': 'Method not allowed'}), 405
        
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
        
        # Analyze different aspects of the text
        readability_metrics = analyze_readability(text)
        grammar_suggestions = get_grammar_suggestions(text)
        style_analysis = analyze_style_and_tone(text)
        
        # Generate comprehensive suggestions
        enhancement_suggestions = generate_enhancement_suggestions(
            text, readability_metrics, style_analysis, grammar_suggestions
        )
        
        # Calculate overall scores
        readability_score = readability_metrics.get('flesch_reading_ease', 50)
        
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
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        response = {
            'success': True,
            'text_statistics': {
                'word_count': word_count,
                'character_count': len(text),
                'sentence_count': len([s for s in text.split('.') if s.strip()]),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            },
            'readability_metrics': readability_metrics,
            'readability_level': readability_level,
            'style_analysis': style_analysis,
            'suggestions': enhancement_suggestions,
            'grammar_suggestions': grammar_suggestions,
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
    
    finally:
        # Clean up memory
        gc.collect()
