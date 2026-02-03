# Stock Market Prediction

Ứng dụng web dự đoán giá chứng khoán sử dụng mô hình **ARIMA** và **LSTM**, xây dựng với Flask.

## Tính năng

- **Đăng ký / Đăng nhập** – Quản lý tài khoản người dùng
- **Phân tích thị trường** – Upload file CSV, xem biểu đồ và các chỉ số (RSI, MA20, MA50, khối lượng…)
- **Dự đoán AI** – Dự đoán giá cổ phiếu bằng mô hình ARIMA hoặc LSTM (từ dữ liệu CSV hoặc symbol)
- **So sánh mô hình** – So sánh kết quả dự đoán ARIMA vs LSTM
- **Lịch sử dự đoán** – Xem và lọc lịch sử dự đoán theo symbol
- **Thông báo** – Gửi email và hỗ trợ Telegram khi có kết quả dự đoán

## Công nghệ

- **Backend:** Flask, Flask-SQLAlchemy, Flask-Login, Flask-WTF
- **ML/Thống kê:** scikit-learn, statsmodels (ARIMA), TensorFlow (LSTM)
- **Dữ liệu:** pandas, numpy, yfinance
- **Trực quan:** Plotly
- **Cơ sở dữ liệu:** SQLite

## Cấu trúc thư mục

```
stock-market-prediction-master/
├── app/
│   ├── ai_models/       # ARIMA, LSTM, prediction_service
│   ├── models/          # User, PredictionHistory
│   ├── routes/          # auth, main, api, predictions
│   ├── services/        # email_service
│   ├── scripts/         # send_telegram_notification
│   ├── templates/       # HTML
│   └── static/          # CSS, JS, images
├── data/                # Dữ liệu mẫu (CSV)
├── instance/            # SQLite DB
├── config.py            # Cấu hình (SECRET_KEY, Gmail, DB)
├── run.py               # Điểm chạy ứng dụng
├── requirements.txt
├── ARIMA.ipynb          # Notebook huấn luyện ARIMA
└── LSTM.ipynb           # Notebook huấn luyện LSTM
```

## Cài đặt

### 1. Clone / mở thư mục dự án

```bash
cd stock-market-prediction-master
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Cài đặt phụ thuộc

```bash
pip install -r requirements.txt
```

### 4. Cấu hình

Chỉnh file `config.py`:

- `SECRET_KEY` – Key bí mật cho session (nên đổi trong production)
- `GMAIL_USER`, `GMAIL_APP_PASSWORD` – Gmail để gửi thông báo
- `RECIPIENT_EMAIL` – Email nhận thông báo dự đoán

Database SQLite sẽ được tạo tự động trong `instance/stock_prediction.db` khi chạy app lần đầu.

## Chạy ứng dụng

```bash
python run.py
```

Mở trình duyệt: **http://127.0.0.1:5000**

## Sử dụng nhanh

1. **Trang chủ** (`/`) – Giới thiệu và điều hướng
2. **Đăng ký / Đăng nhập** – Tạo tài khoản hoặc đăng nhập
3. **Phân tích thị trường** (`/market-analysis`) – Upload CSV (cột: Date, Open, High, Low, Close, Volume), xem biểu đồ và báo cáo
4. **Dự đoán AI** (`/ai-prediction`) – Chọn ARIMA hoặc LSTM, upload CSV (cột Date, Close) hoặc nhập symbol, xem dự đoán và biểu đồ
5. **So sánh** (`/cp` hoặc `/comparison`) – Sau khi chạy dự đoán với cả ARIMA và LSTM, vào trang này để so sánh

## API (ví dụ)

- `GET /api/predict/<symbol>` – Dự đoán theo symbol (ví dụ: AAPL, BTC-USD), gửi thông báo email
- `GET /api/predictions/history` – Lịch sử dự đoán (query: `symbol`, `limit`)

## Định dạng CSV

- **Phân tích thị trường:** Cần các cột: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **Dự đoán AI:** Cần ít nhất: `Date`, `Close`. Dự đoán LSTM yêu cầu tối thiểu 60 ngày dữ liệu

## Lưu ý

- Trong `config.py` có thông tin Gmail thật; nên dùng biến môi trường hoặc file cấu hình riêng, không commit mật khẩu lên Git
- Mô hình ARIMA/LSTM đã train sẵn nằm trong `app/ai_models/` (`.pkl`, `.keras`); có thể train lại qua notebook `ARIMA.ipynb`, `LSTM.ipynb`

## License

MIT (hoặc theo giấy phép của dự án gốc).
