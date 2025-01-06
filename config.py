import os

class Config:
    SECRET_KEY = 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///stock_prediction.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Gmail Settings
    GMAIL_USER = 'trongdang030925@gmail.com'
    GMAIL_APP_PASSWORD = 'vxezexxkmhmllchw'  # Mật khẩu ứng dụng Gmail
    RECIPIENT_EMAIL = 'trongdang030925@gmail.com'  # Email người nhận thông báo dự đoán