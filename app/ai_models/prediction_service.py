import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from app.services.email_service import email_service
from config import Config
from app.models.prediction_history import PredictionHistory
from app.database import db
from app.views.ai_prediction import handle_prediction
from app.scripts.send_telegram_notification import telegram_bot_thread

class StockPredictionService:
    def __init__(self):
        try:
            # Load the pre-trained models
            with open('app/ai_models/arima_model.pkl', 'rb') as f:
                self.arima_model = pickle.load(f)
            self.lstm_model = tf.keras.models.load_model('app/ai_models/lstm_model.keras')
            
            # Configuration for LSTM input
            self.sequence_length = 60
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
            self.scaler = MinMaxScaler()
            
            # Email configuration
            self.notification_email = Config.GMAIL_USER
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.arima_model = None
            self.lstm_model = None
        
    def prepare_lstm_data(self, data):
        # Normalize the data
        normalized_data = data[self.features].apply(lambda x: (x - x.mean()) / x.std())
        
        # Create sequences
        sequences = []
        for i in range(len(normalized_data) - self.sequence_length):
            sequences.append(normalized_data.iloc[i:(i + self.sequence_length)].values)
        return np.array(sequences)
    
    def get_stock_data(self, symbol, period='60d'):
        """Fetch stock data using yfinance"""
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    
    def predict_next_day(self, symbol):
        """Make predictions using both ARIMA and LSTM models"""
        try:
            # Fetch recent data
            data = self.get_stock_data(symbol)
            if data.empty:
                return {"error": "No data available for this symbol"}

            # ARIMA prediction
            arima_pred = None
            if self.arima_model:
                try:
                    arima_pred = self.arima_model.forecast(steps=1)[0]
                except Exception as e:
                    print(f"ARIMA prediction failed: {str(e)}")

            # LSTM prediction
            lstm_pred = None
            if self.lstm_model and len(data) >= self.sequence_length:
                sequences = self.prepare_lstm_data(data)
                if len(sequences) > 0:
                    lstm_pred = self.lstm_model.predict(sequences[-1:])[-1][0]

            # Combine predictions
            final_prediction = None
            if arima_pred is not None and lstm_pred is not None:
                final_prediction = (arima_pred + lstm_pred) / 2
            elif arima_pred is not None:
                final_prediction = arima_pred
            elif lstm_pred is not None:
                final_prediction = lstm_pred

            if final_prediction is None:
                return {"error": "Prediction failed"}

            prediction_date = datetime.now().date() + timedelta(days=1)
            current_price = round(float(data['Close'].iloc[-1]), 2)
            predicted_price = round(float(final_prediction), 2)

            # Save prediction to database
            prediction_record = PredictionHistory(
                symbol=symbol,
                current_price=current_price,
                predicted_price=predicted_price,
                prediction_date=prediction_date
            )
            db.session.add(prediction_record)
            db.session.commit()

            prediction_result = {
                "symbol": symbol,
                "predicted_price": predicted_price,
                "prediction_date": prediction_date.strftime('%Y-%m-%d'),
                "current_price": current_price,
                "prediction_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "history_id": prediction_record.id
            }

            # Send email notification
            if self.notification_email:
                email_service.send_prediction_notification(
                    self.notification_email,
                    symbol,
                    prediction_result
                )

            return prediction_result
        except Exception as e:
            return {"error": str(e)}

    def perform_prediction(self, file_path):
        # ... logic dự đoán ...
        # Sau khi dự đoán hoàn tất, gọi hàm handle_prediction
        handle_prediction(file_path)
        
        # Gửi thông báo kết quả dự đoán thành công
        message = "KẾT QUẢ ĐÃ DỰ ĐOÁN THÀNH CÔNG CHO NGƯỜI DÙNG"
        telegram_bot_thread(message)

# Create singleton instance
prediction_service = StockPredictionService()
