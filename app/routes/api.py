from flask import Blueprint, jsonify, request
from app.ai_models.arima_model import predict_arima
from app.ai_models.lstm_model import predict_lstm
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

api = Blueprint('api', __name__)

@api.route('/api/analyze/<symbol>')
def analyze_stock(symbol):
    try:
        # Lấy dữ liệu lịch sử
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = yf.download(symbol, start=start_date, end=end_date)

        if df.empty:
            return jsonify({'error': 'Không tìm thấy dữ liệu cho mã cổ phiếu này'})

        # Tính các chỉ số kỹ thuật
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = calculate_macd(df['Close'])
        df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])

        # Chuẩn bị dữ liệu trả về
        response = {
            'prices': {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'prices': df['Close'].tolist()
            },
            'indicators': {
                'rsi': round(df['RSI'].iloc[-1], 2),
                'macd': round(df['MACD'].iloc[-1], 2),
                'bollinger': f"{round(df['BB_lower'].iloc[-1], 2)} - {round(df['BB_upper'].iloc[-1], 2)}"
            },
            'news': get_market_news(symbol)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

def calculate_rsi(prices, periods=14):
    # Tính RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    # Tính MACD
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    # Tính Bollinger Bands
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def get_market_news(symbol):
    # Giả lập tin tức thị trường
    return [
        {
            'title': f'Tin tức về {symbol}',
            'content': f'Nội dung tin tức về {symbol}',
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    ]

def send_telegram_notification(message):
    bot_token = '8104804623:AAFSCDiTimsVa8TqtGsSxeNchyl4JvjjluA'  # Replace with your actual bot token
    chat_id = '5097716190'  # Replace with your actual chat ID
    url = f'https://api.telegram.org/bot8104804623:AAFSCDiTimsVa8TqtGsSxeNchyl4JvjjluA/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.post(url, json=payload)
    return response.json()

def handle_prediction_success(result):
    send_telegram_notification(f'Prediction successful: {result}')

@api.route('/api/mock-prediction/<symbol>')
def mock_prediction(symbol):
    # API giả lập cho dự đoán
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Simulate a successful prediction
    result = df['Close'].iloc[-1] * (1 + np.random.normal(0, 0.02))
    handle_prediction_success(result)

    return jsonify({
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'predDates': [(end_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)],
    })