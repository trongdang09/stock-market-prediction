from app.scripts.send_telegram_notification import send_predictions_to_telegram
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

def handle_prediction(file_path):
    # ... xử lý tải lên và dự đoán ...
    # Giả sử kết quả dự đoán được lưu vào file CSV
    prediction_csv_path = 'path/to/your/predictions.csv'
    
    # Kiểm tra xem file CSV có tồn tại và không rỗng
    try:
        with open(prediction_csv_path, 'r') as f:
            if f.read().strip():
                # Gửi thông báo đến Telegram
                send_predictions_to_telegram(prediction_csv_path)
                logging.info("Notification sent to Telegram.")
            else:
                logging.warning("CSV file is empty. No notification sent.")
    except FileNotFoundError:
        logging.error("CSV file not found. No notification sent.")
    
    # ... các xử lý khác ...