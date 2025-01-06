import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.ai_models.prediction_service import prediction_service
from app.services.email_service import email_service
from config import Config

def test_prediction():
    app = create_app()
    with app.app_context():
        symbol = 'AAPL'
        result = prediction_service.predict_next_day(symbol)
        print("Prediction result:", result)
        
        # Check if prediction was successful
        if 'error' not in result:
            # Send email notification
            email_service.send_prediction_notification(
                recipient_email=Config.RECIPIENT_EMAIL,
                symbol=symbol,
                prediction_data={
                    'current_price': result['current_price'],
                    'predicted_price': result['predicted_price'],
                    'prediction_date': result['prediction_date']
                }
            )
            print(f"Email notification sent to {Config.RECIPIENT_EMAIL}")
        else:
            print(f"Error in prediction: {result['error']}")
            # Send error notification email
            email_service.send_prediction_notification(
                recipient_email=Config.RECIPIENT_EMAIL,
                symbol=symbol,
                prediction_data={
                    'current_price': 0,
                    'predicted_price': 0,
                    'prediction_date': None,
                    'error_message': result['error']
                }
            )
            print(f"Error notification email sent to {Config.RECIPIENT_EMAIL}")

if __name__ == '__main__':
    test_prediction()
