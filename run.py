"""Entry point — run with: python -m uvicorn run:app --reload"""
import sys
sys.path.insert(0, ".")
from backend.main import app
