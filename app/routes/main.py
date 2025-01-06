import os
from flask import Blueprint, render_template, request, flash, current_app, jsonify, session, redirect, url_for
from flask_login import login_required
from werkzeug.utils import secure_filename
import pandas as pd
from app.ai_models.arima_model import predict_arima
from app.ai_models.lstm_model import predict_lstm
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from statsmodels.tsa.arima.model import ARIMA

main = Blueprint('main', __name__)

# Thêm các hàm phân tích
def analyze_csv_data(df):
    """Phân tích dữ liệu từ DataFrame"""
    try:
        # Tính các chỉ số cơ bản
        df['Daily_Return'] = df['Close'].pct_change()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Tính RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Tạo báo cáo phân tích
        analysis = {
            'last_price': df['Close'].iloc[-1],
            'price_change': df['Daily_Return'].iloc[-1] * 100,
            'avg_volume': df['Volume'].mean(),
            'rsi': df['RSI'].iloc[-1],
            'ma20': df['MA20'].iloc[-1],
            'ma50': df['MA50'].iloc[-1]
        }
        
        # Tạo HTML table từ kết quả phân tích
        html_table = f"""
        <table class="table table-striped">
            <tr><th>Chỉ số</th><th>Giá trị</th></tr>
            <tr><td>Giá đóng cửa gần nhất</td><td>{analysis['last_price']:.2f}</td></tr>
            <tr><td>Thay đổi giá (%)</td><td>{analysis['price_change']:.2f}%</td></tr>
            <tr><td>Khối lượng trung bình</td><td>{analysis['avg_volume']:.0f}</td></tr>
            <tr><td>RSI</td><td>{analysis['rsi']:.2f}</td></tr>
            <tr><td>MA20</td><td>{analysis['ma20']:.2f}</td></tr>
            <tr><td>MA50</td><td>{analysis['ma50']:.2f}</td></tr>
        </table>
        """
        
        return html_table
    except Exception as e:
        return f"Lỗi khi phân tích dữ liệu: {str(e)}"

@main.route('/upload-csv', methods=['POST'])
@login_required
def upload_csv():
    if 'file' not in request.files:
        flash('Không tìm thấy file', 'error')
        return render_template('market_analysis.html')
    
    file = request.files['file']
    if file.filename == '':
        flash('Chưa chọn file', 'error')
        return render_template('market_analysis.html')
    
    if file and file.filename.endswith('.csv'):
        try:
            # Đọc file CSV
            df = pd.read_csv(file)
            
            # Kiểm tra các cột bắt buộc
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                flash('File CSV không đúng định dạng. Cần các cột: ' + ', '.join(required_columns), 'error')
                return render_template('market_analysis.html')
            
            # Chuyển đổi cột Date
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Lọc dữ liệu theo ngày nếu có
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['Date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['Date'] <= end_date]
            
            if len(df) == 0:
                flash('Không có dữ liệu trong khoảng thời gian đã chọn', 'error')
                return render_template('market_analysis.html')
            
            # Chuẩn bị dữ liệu cho biểu đồ
            chart_data = {
                'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': {
                    'open': [float(x) for x in df['Open'].tolist()],
                    'high': [float(x) for x in df['High'].tolist()],
                    'low': [float(x) for x in df['Low'].tolist()],
                    'close': [float(x) for x in df['Close'].tolist()]
                },
                'volume': [float(x) for x in df['Volume'].tolist()]
            }
            
            # Phân tích dữ liệu
            analysis_result = analyze_csv_data(df)
            
            return render_template('market_analysis.html', 
                                analysis_result=analysis_result,
                                chart_data=chart_data)
            
        except Exception as e:
            flash(f'Lỗi khi xử lý file: {str(e)}', 'error')
            return render_template('market_analysis.html')
    
    flash('Chỉ chấp nhận file CSV', 'error')
    return render_template('market_analysis.html')

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/market-analysis')
@login_required
def market_analysis():
    return render_template('market_analysis.html')

@main.route('/ai-prediction')
@login_required
def ai_prediction():
    return render_template('ai_prediction.html', chart_data=None, show_results=False)

@main.route('/predict-from-csv', methods=['POST'])
@login_required
def predict_from_csv():
    try:
        if 'file' not in request.files:
            flash('Không tìm thấy file', 'error')
            return render_template('ai_prediction.html', chart_data=None, show_results=False)
        
        file = request.files['file']
        if file.filename == '':
            flash('Chưa chọn file', 'error')
            return render_template('ai_prediction.html', chart_data=None, show_results=False)
        
        if not file or not file.filename.endswith('.csv'):
            flash('Chỉ chấp nhận file CSV', 'error')
            return render_template('ai_prediction.html', chart_data=None, show_results=False)
            
        # Đọc file CSV
        df = pd.read_csv(file)
        
        # Kiểm tra các cột bắt buộc
        required_columns = ['Date', 'Close']
        if not all(col in df.columns for col in required_columns):
            flash('File CSV cần có các cột: ' + ', '.join(required_columns), 'error')
            return render_template('ai_prediction.html', chart_data=None, show_results=False)
        
        # Chuyển đổi cột Date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Lọc dữ liệu theo ngày nếu có
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['Date'] <= end_date]
        
        if len(df) == 0:
            flash('Không có dữ liệu trong khoảng thời gian đã chọn', 'error')
            return render_template('ai_prediction.html', chart_data=None, show_results=False)
        
        # Thực hiện dự đoán
        model_type = request.form.get('model_type', 'arima')
        prediction_days = 30  # Số ngày dự đoán
        
        # Tạo danh sách ngày dự đoán trong tương lai
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=prediction_days,
            freq='D'
        )
        
        predictions = None
        
        if model_type == 'arima':
            predictions = predict_future_arima(df, prediction_days)
            session['arima_prediction'] = predictions
        else:
            predictions = predict_future_lstm(df, prediction_days)
            session['lstm_prediction'] = predictions
        
        # Chuẩn bị dữ liệu cho biểu đồ
        chart_data = {
            'dates': df['Date'].dt.strftime('%d/%m/%Y').tolist(),
            'historical': df['Close'].tolist(),
            'future_dates': [(last_date + timedelta(days=i+1)).strftime('%d/%m/%Y') for i in range(prediction_days)],
            'predictions': predictions['values']
        }
        
        # Tạo bảng kết quả dự đoán
        prediction_table = create_prediction_table(predictions, future_dates)
        
        return render_template('ai_prediction.html',
                           prediction_result=prediction_table,
                           chart_data=chart_data,
                           show_results=True)  # Thêm flag để hiển thị kết quả
                           
    except Exception as e:
        flash(f'Lỗi khi xử lý dự đoán: {str(e)}', 'error')
        return render_template('ai_prediction.html', chart_data=None, show_results=False)

def create_prediction_table(predictions, dates):
    """Tạo bảng HTML hiển thị kết quả dự đoán"""
    html = """
    <table class="table table-striped table-bordered">
        <thead>
        <tr>
            <th>Thời gian</th>
            <th>Giá dự đoán</th>
            <th>Độ tin cậy</th>
        </tr>
        </thead>
        <tbody>
    """
    
    # Sử dụng ngày cuối cùng từ dữ liệu thực tế làm điểm bắt đầu
    start_date = dates[-1] if isinstance(dates, pd.DatetimeIndex) else dates.iloc[-1]
    prediction_dates = pd.date_range(
        start=start_date + pd.Timedelta(days=1),
        periods=len(predictions['values']),
        freq='D'
    )
    
    for date, value, conf in zip(prediction_dates, predictions['values'], predictions['confidence']):
        html += f"""
        <tr>
            <td>{date.strftime('%d/%m/%Y')}</td>
            <td>{value:.2f}</td>
            <td>{conf:.1f}%</td>
        </tr>
        """
    
    html += "</tbody></table>"
    return html

def predict_future_arima(df, days):
    """Dự đoán giá trong tương lai sử dụng ARIMA"""
    try:
        model = ARIMA(df['Close'].values, order=(5,1,0))
        model_fit = model.fit()
        
        # Dự đoán và chuyển đổi sang Python float
        forecast = model_fit.forecast(steps=days)
        confidence = float(95 - (model_fit.resid.std() / df['Close'].mean() * 100))
        
        return {
            'values': [float(x) for x in forecast],
            'confidence': [float(confidence)] * days
        }
    except Exception as e:
        print(f"Error in ARIMA prediction: {e}")
        return {
            'values': [float(df['Close'].iloc[-1])] * days,
            'confidence': [50.0] * days
        }

def predict_future_lstm(df, days, force_train=False):
    """Dự đoán giá trong tương lai sử dụng LSTM"""
    try:
        # Chuẩn bị dữ liệu
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        # Kiểm tra đủ dữ liệu cho huấn luyện
        if len(scaled_data) < 60:
            raise ValueError("Cần ít nhất 60 ngày dữ liệu để dự đoán")
        
        # Tạo và huấn luyện mô hình LSTM
        model = create_lstm_model(scaled_data, force_train)
        if model is None:
            raise ValueError("Không thể tạo hoặc load model LSTM")
            
        # Dự đoán
        last_sequence = scaled_data[-60:].reshape(1, 60, 1)
        predictions = []
        confidence = []
        
        current_sequence = last_sequence.copy()
        for _ in range(days):
            # Dự đoán giá tiếp theo
            next_pred = model.predict(current_sequence, verbose=0)
            
            # Chuyển đổi giá trị dự đoán về thang đo gốc và thành Python float
            predicted_price = float(scaler.inverse_transform(next_pred)[0,0])
            predictions.append(predicted_price)
            confidence.append(90.0)  # Độ tin cậy cố định
            
            # Cập nhật sequence cho lần dự đoán tiếp theo
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0,0]
        
        return {
            'values': [float(x) for x in predictions],  # Chuyển đổi tất cả giá trị thành float
            'confidence': [float(x) for x in confidence]
        }
        
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán LSTM: {str(e)}")
        return {
            'values': [float(df['Close'].iloc[-1])] * days,
            'confidence': [50.0] * days
        }

def create_lstm_model(data, force_train=False):
    """Tạo và huấn luyện mô hình LSTM hoặc load từ file"""
    try:
        model_path = 'save_model.keras'
        
        # Thử load model đã lưu nếu không bắt buộc train lại
        if os.path.exists(model_path) and not force_train:
            try:
                return load_model(model_path)
            except:
                print("Không thể load model, sẽ train lại")
        
        # Chuẩn bị dữ liệu huấn luyện
        X, y = prepare_sequences(data)
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Không đủ dữ liệu để huấn luyện model")
            
        # Tạo model mới
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Biên dịch model
        model.compile(optimizer='adam', loss='mse')
        
        # Huấn luyện model
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Lưu model
        try:
            model.save(model_path)
            print("Đã lưu model thành công")
        except Exception as e:
            print(f"Lỗi khi lưu model: {str(e)}")
        
        return model
        
    except Exception as e:
        print(f"Lỗi khi tạo model LSTM: {str(e)}")
        return None

def prepare_sequences(data):
    """Chuẩn bị dữ liệu cho LSTM"""
    try:
        X, y = [], []
        if len(data) <= 60:
            return np.array(X), np.array(y)
            
        for i in range(len(data) - 60):
            X.append(data[i:(i + 60)])
            y.append(data[i + 60])
            
        return np.array(X), np.array(y)
        
    except Exception as e:
        print(f"Lỗi khi chuẩn bị dữ liệu: {str(e)}")
        return np.array([]), np.array([]) 

@main.route('/predict', methods=['POST'])
def predict():
    # Code xử lý dự đoán
    try:
        # Xử lý dự đoán và tính toán kết quả
        
        # Lưu kết quả vào session
        session['prediction1'] = prediction1
        session['prediction2'] = prediction2
        session['accuracy1'] = accuracy1
        session['accuracy2'] = accuracy2
        session['chart1_data'] = chart1_data
        session['chart2_data'] = chart2_data
        
        # Kiểm tra dự đoán trước đó
        if 'previous_prediction' in session:
            previous_model = session['previous_prediction'].get('model')
            previous_result = session['previous_prediction'].get('result')
            previous_accuracy = session['previous_prediction'].get('accuracy')
            
            comparison = {
                'current': {
                    'model': selected_model,
                    'result': result,
                    'accuracy': accuracy
                },
                'previous': {
                    'model': previous_model,
                    'result': previous_result,
                    'accuracy': previous_accuracy
                }
            }
        else:
            comparison = None
        
        # Lưu kết quả dự đoán hiện tại vào session
        session['previous_prediction'] = {
            'model': selected_model,
            'result': result,
            'accuracy': accuracy
        }
        
        # Dự đoán với mô hình ARIMA
        arima_predictions = arima_model.predict(data)
        
        # Dự đoán với mô hình LSTM
        lstm_predictions = lstm_model.predict(data)
        
        # Chuyển kết quả đến trang so sánh
        return render_template('comparison.html', arima=arima_predictions, lstm=lstm_predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@main.route('/comparison')
def comparison():
    arima_prediction = session.get('arima_prediction', None)
    lstm_prediction = session.get('lstm_prediction', None)
    
    if not isinstance(arima_prediction, dict) or not isinstance(lstm_prediction, dict):
        flash('Vui lòng thực hiện dự đoán cả hai mô hình trước khi so sánh', 'warning')
        return redirect(url_for('main.home'))
        
    if 'values' not in arima_prediction or 'values' not in lstm_prediction:
        flash('Dữ liệu dự đoán không hợp lệ', 'warning')
        return redirect(url_for('main.home'))
        
    return render_template('comparison.html', 
                         arima_prediction=arima_prediction['values'],
                         lstm_prediction=lstm_prediction['values'])

@main.route('/get-comparison-data')
@login_required
def get_comparison_data():
    arima_pred = session.get('arima_prediction')
    lstm_pred = session.get('lstm_prediction')
    
    predictions = []
    if arima_pred:
        predictions.append(arima_pred)
    if lstm_pred:
        predictions.append(lstm_pred)
    
    print("Debug - Session data:")
    print("ARIMA:", arima_pred)
    print("LSTM:", lstm_pred)
    
    return jsonify(predictions)

@main.route('/cp')
@login_required
def cp():
    # Kiểm tra xem đã có đủ dữ liệu chưa
    if 'arima_prediction' not in session or 'lstm_prediction' not in session:
        flash('Vui lòng thực hiện dự đoán với cả hai mô hình ARIMA và LSTM trước khi so sánh', 'warning')
        return redirect(url_for('main.ai_prediction'))
    
    arima_data = session.get('arima_prediction')
    lstm_data = session.get('lstm_prediction')
    
    return render_template('cp.html',
                         arima_prediction=arima_data,
                         lstm_prediction=lstm_data)

@main.route('/predict_arima', methods=['POST'])
def predict_arima():
    try:
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'error': 'Không có dữ liệu đầu vào'}), 400
        
        values = data['values']
        if not isinstance(values, (list, np.ndarray)) or len(values) < 60:
            return jsonify({'error': 'Cần ít nhất 60 ngày dữ liệu'}), 400
        
        # Chuyển đổi values thành numpy array
        values = np.array(values)
        
        # Thực hiện dự đoán ARIMA
        model = ARIMA(values, order=(5,1,0))
        model_fit = model.fit()
        
        # Dự đoán
        forecast = model_fit.forecast(steps=data.get('days', 30))
        confidence = float(95 - (model_fit.resid.std() / np.mean(values) * 100))
        
        predictions = {
            'values': [float(x) for x in forecast],
            'confidence': [confidence] * len(forecast)
        }
        
        # Lưu kết quả vào session
        session['arima_prediction'] = predictions
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@main.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'error': 'Không có dữ liệu đầu vào'}), 400
        
        values = data['values']
        if not isinstance(values, (list, np.ndarray)) or len(values) < 60:
            return jsonify({'error': 'Cần ít nhất 60 ngày dữ liệu'}), 400
        
        # Chuẩn bị dữ liệu cho LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(values).reshape(-1, 1))
        
        # Tạo và huấn luyện model LSTM
        model = create_lstm_model(scaled_data)
        if model is None:
            raise ValueError("Không thể tạo model LSTM")
            
        # Dự đoán
        last_sequence = scaled_data[-60:].reshape(1, 60, 1)
        predictions = []
        days = data.get('days', 30)
        
        current_sequence = last_sequence.copy()
        for _ in range(days):
            next_pred = model.predict(current_sequence, verbose=0)
            predicted_price = float(scaler.inverse_transform(next_pred)[0,0])
            predictions.append(predicted_price)
            
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0,0]
        
        result = {
            'values': predictions,
            'confidence': [90.0] * len(predictions)
        }
        
        # Lưu kết quả vào session
        session['lstm_prediction'] = result
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400