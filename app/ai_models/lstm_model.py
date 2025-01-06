import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from datetime import datetime, timedelta

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def predict_lstm(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        # Lấy dữ liệu
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            return {"prediction": None, "confidence": 0}
        
        # Chuẩn bị dữ liệu
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Tham số
        time_steps = 60
        X, y = prepare_data(data_scaled, time_steps)
        
        # Chia dữ liệu
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Tạo mô hình
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Huấn luyện mô hình
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Dự đoán
        last_sequence = data_scaled[-time_steps:]
        next_pred = model.predict(last_sequence.reshape(1, time_steps, 1))
        prediction = scaler.inverse_transform(next_pred)[0][0]
        
        # Tính độ tin cậy
        test_pred = model.predict(X_test)
        mse = np.mean((y_test - test_pred.reshape(-1))**2)
        confidence = min(100, 100 * (1 - mse))
        
        return {
            "prediction": round(prediction, 2),
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        print(f"Error in LSTM prediction: {e}")
        return {"prediction": None, "confidence": 0} 