"""
Simple Production Server - RAG Chatbot
Optimized for 3 concurrent users on Windows
"""

import os
from pathlib import Path
from waitress import serve

# Create required directories
#for folder in ['uploads', 'logs', 'data', 'temp', 'exports']:
#    Path(folder).mkdir(exist_ok=True)

# Import Flask app
from flask_server import app

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Chatbot Server Starting...")
    print("="*50)
    print(f"📡 URL: http://localhost:5000")
    print("🛑 Stop: Press Ctrl+C")
    print("="*50 + "\n")
    
    # Start production server
    serve(
        app,
        host='0.0.0.0',
        port=5000,
        threads=4,          # 4 worker threads
        channel_timeout=300, # 5 min timeout
        connection_limit=100 # Max 100 connections
    )
