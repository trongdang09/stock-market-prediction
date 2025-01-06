import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import Config
from datetime import datetime

class EmailService:
    def __init__(self):
        self.sender_email = Config.GMAIL_USER
        self.sender_password = Config.GMAIL_APP_PASSWORD
        
    def send_prediction_notification(self, recipient_email, symbol, prediction_data):
        # Check if there's an error
        if 'error_message' in prediction_data:
            subject = f"❌ Lỗi Dự Đoán Giá Cổ Phiếu {symbol}"
            body = f"""
            ⚠️ THÔNG BÁO LỖI DỰ ĐOÁN
            ═══════════════════════════════

            📊 Mã Cổ Phiếu: {symbol}
            
            ❌ Lỗi: {prediction_data['error_message']}
            
            ⏰ Thời điểm: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
            
            📝 Ghi chú:
            • Vui lòng kiểm tra lại dữ liệu đầu vào
            • Liên hệ admin nếu lỗi vẫn tiếp tục xảy ra
            
            ═══════════════════════════════
            Stock Prediction System
            """
        else:
            subject = f"🔔 Dự Đoán Giá Cổ Phiếu {symbol}"
            
            # Tính toán phần trăm thay đổi
            current_price = prediction_data.get('current_price', 0)
            predicted_price = prediction_data.get('predicted_price', 0)
            change = predicted_price - current_price
            change_percent = (change / current_price * 100) if current_price else 0
            
            # Xác định xu hướng
            trend = "📈 TĂNG" if change > 0 else "📉 GIẢM" if change < 0 else "➡️ ĐỨNG GIÁ"
            
            # Tạo nội dung email
            body = f"""
            🏢 THÔNG BÁO DỰ ĐOÁN GIÁ CỔ PHIẾU
            ═══════════════════════════════

            📊 Mã Cổ Phiếu: {symbol}
            
            💰 Giá Hiện Tại: ${current_price:,.2f}
            🎯 Giá Dự Đoán: ${predicted_price:,.2f}
            
            📊 Phân Tích:
            {trend}
            • Thay đổi: ${abs(change):,.2f}
            • Tỷ lệ: {abs(change_percent):.2f}%
            
            ⏰ Thông Tin Thời Gian:
            • Ngày dự đoán: {prediction_data.get('prediction_date')}
            • Thời điểm dự đoán: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
            
            📝 Lưu ý:
            • Dự đoán này được tạo bởi hệ thống AI
            • Kết quả chỉ mang tính tham khảo
            • Vui lòng cân nhắc kỹ trước khi đưa ra quyết định đầu tư
            
            ═══════════════════════════════
            Stock Prediction System
            """
        
        # Thiết lập email
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        
        # Thêm nội dung vào email
        message.attach(MIMEText(body, "plain"))
        
        try:
            # Tạo kết nối SMTP và gửi email
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                text = message.as_string()
                server.sendmail(self.sender_email, recipient_email, text)
                print(f"✅ Đã gửi email thành công tới {recipient_email}")
            return True
        except Exception as e:
            print(f"❌ Lỗi gửi email: {str(e)}")
            return False
            
    def send_test_email(self, recipient_email):
        """Send a test email to verify email configuration"""
        subject = "Test Email from Stock Prediction System"
        body = """
        This is a test email from your Stock Prediction System.
        If you receive this email, it means your email configuration is working correctly.
        
        Best regards,
        Stock Prediction System
        """
        
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                text = message.as_string()
                server.sendmail(self.sender_email, recipient_email, text)
                print(f"Test email sent successfully to {recipient_email}")
            return True
        except Exception as e:
            print(f"Failed to send test email: {str(e)}")
            return False
            
    def _get_price_change_message(self, prediction_data):
        try:
            current_price = float(prediction_data.get('current_price', 0))
            predicted_price = float(prediction_data.get('predicted_price', 0))
            
            if current_price and predicted_price:
                change = predicted_price - current_price
                change_percent = (change / current_price) * 100
                
                if change > 0:
                    return f"Expected to INCREASE by ${abs(change):.2f} ({abs(change_percent):.2f}%)"
                elif change < 0:
                    return f"Expected to DECREASE by ${abs(change):.2f} ({abs(change_percent):.2f}%)"
                else:
                    return "No significant change expected"
                    
        except (ValueError, TypeError):
            pass
        return "Unable to calculate price change"

email_service = EmailService()
