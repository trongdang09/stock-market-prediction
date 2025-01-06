import csv
import telegram
import logging
import requests
import threading

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named app.log
        logging.StreamHandler()          # Also log to console
    ]
)

# Thay thế 'YOUR_TELEGRAM_BOT_TOKEN' và 'YOUR_CHAT_ID' bằng giá trị thực tế
bot_token = '8104804623:AAFSCDiTimsVa8TqtGsSxeNchyl4JvjjluA'
chat_id = '5097716190'

# Khởi tạo bot
bot = telegram.Bot(token=bot_token)

def send_telegram_message(message):
    """Gửi tin nhắn tới Telegram"""
    logging.info(f"Preparing to send message: {message}")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info("Message sent successfully")
        else:
            logging.error(f"Failed to send message: {response.text}")
    except Exception as e:
        logging.error(f"Error sending message: {e}")

def telegram_bot_thread(message):
    """Chạy hàm send_telegram_message trên một luồng riêng biệt"""
    thread = threading.Thread(target=send_telegram_message, args=(message,))
    thread.start()

# Đọc file CSV và gửi thông báo
def send_predictions_to_telegram(csv_file_path):
    logging.info(f"Reading CSV file: {csv_file_path}")
    try:
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                message = f"Prediction Result:\n{row}"
                logging.info(f"Sending message for row: {row}")
                telegram_bot_thread(message)
    except Exception as e:
        logging.error(f"Error processing CSV file: {e}")

# Gọi hàm với đường dẫn tới file CSV của bạn
# Update this path to the actual location of your CSV file
send_predictions_to_telegram('/path/to/your/actual/predictions.csv') 