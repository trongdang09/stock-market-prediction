import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def predict_arima(symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Lấy dữ liệu
    df = get_stock_data(symbol, start_date, end_date)
    if df is None:
        return {"prediction": None, "confidence": 0}
    
    # Chuẩn bị dữ liệu
    data = df['Close'].values
    
    try:
        # Tạo và huấn luyện mô hình ARIMA
        model = ARIMA(data, order=(5,1,0))
        model_fit = model.fit()
        
        # Dự đoán
        forecast = model_fit.forecast(steps=1)
        prediction = forecast[0]
        
        # Tính độ tin cậy dựa trên độ lệch chuẩn
        confidence = min(100, 100 * (1 - model_fit.resid.std() / np.mean(data)))
        
        return {
            "prediction": round(prediction, 2),
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        print(f"Error in ARIMA prediction: {e}")
        return {"prediction": None, "confidence": 0} 