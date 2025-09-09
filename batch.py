from flask import jsonify, request
from datetime import datetime
import asyncio
import concurrent.futures
import time
import logging

# Import detection logic from detect.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

def process_single_text(text_data):
    """Process a single text for batch operations"""
    try:
        # Import the detection handler
        from detect import handler as detect_handler
        
        # Create a mock request object
        class MockRequest:
            def __init__(self, data):
                self.data = data
            
            def get_json(self):
                return self.data
            
            @property
            def method(self):
                return 'POST'
        
        mock_request = MockRequest({'text': text_data['text']})
        result = detect_handler(mock_request)
        
        return {
            'id': text_data.get('id', ''),
            'success': True,
            'result': result.get_json() if hasattr(result, 'get_json') else result
        }
        
    except Exception as e:
        return {
            'id': text_data.get('id', ''),
            'success': False,
            'error': str(e)
        }

def handler(request):
    """Vercel serverless function handler for batch processing"""
    try:
        if request.method != 'POST':
            return jsonify({'error': 'Method not allowed'}), 405
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts array provided'
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'texts must be an array'
            }), 400
        
        if len(texts) > 50:  # Limit batch size
            return jsonify({
                'success': False,
                'error': 'Maximum 50 texts per batch'
            }), 400
        
        start_time = time.time()
        
        # Process texts in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_text = {executor.submit(process_single_text, text_data): text_data for text_data in texts}
            
            for future in concurrent.futures.as_completed(future_to_text):
                result = future.result()
                results.append(result)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        response = {
            'success': True,
            'total_processed': len(texts),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'results': results,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500
