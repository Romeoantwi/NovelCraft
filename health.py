from flask import jsonify
from datetime import datetime
import sys
import os

def handler(request):
    """Vercel serverless function handler for health check"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'NovelCraft AI Detection API',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'endpoints': [
                '/api/detect - AI content detection',
                '/api/enhance - Text enhancement and analysis',
                '/api/health - Health check',
                '/api/batch - Batch processing'
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
