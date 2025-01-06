import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.email_service import email_service

def test_email():
    app = create_app()
    with app.app_context():
        email_service.send_test_email('trongdang030925@gmail.com')

if __name__ == '__main__':
    test_email()
