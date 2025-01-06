import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from app.database import db
from app.models.prediction_history import PredictionHistory
from config import Config

def create_tables():
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize database
    db.init_app(app)
    
    # Create tables within app context
    with app.app_context():
        db.create_all()
        print("Created prediction_history table")

if __name__ == '__main__':
    create_tables()
