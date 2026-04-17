import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import the FastAPI app
from main import app

# Vercel expects the app to be named 'app'
# But it's already named app in main.py