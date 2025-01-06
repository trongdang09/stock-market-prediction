import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.email_service import email_service

def test_prediction_email():
    # Dữ liệu dự đoán mẫu
    test_prediction = {
        'symbol': 'AAPL',
        'current_price': 197.57,
        'predicted_price': 199.85,
        'prediction_date': '2024-12-21',
        'prediction_time': '00:17:00 20/12/2023'
    }
    
    app = create_app()
    with app.app_context():
        # Gửi email thông báo dự đoán
        email_service.send_prediction_notification(
            'trongdang030925@gmail.com',
            test_prediction['symbol'],
            test_prediction
        )

if __name__ == '__main__':
    test_prediction_email()
