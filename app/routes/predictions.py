from flask import Blueprint, jsonify, request, render_template
from app.ai_models.prediction_service import prediction_service
from app.services.email_service import email_service
from app.models.prediction_history import PredictionHistory
from flask_login import login_required

predictions = Blueprint('predictions', __name__)

@predictions.route('/api/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    try:
        prediction = prediction_service.predict_next_day(symbol)
        
        # Always send email notification to the specified address
        email_service.send_prediction_notification(
            'trongdang030925@gmail.com',
            symbol,
            prediction
        )
            
        return jsonify({
            'success': True,
            'data': prediction,
            'message': 'Prediction results have been sent to trongdang030925@gmail.com'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@predictions.route('/api/predictions/history', methods=['GET'])
def get_prediction_history():
    try:
        # Get query parameters
        symbol = request.args.get('symbol')
        limit = request.args.get('limit', default=10, type=int)
        
        # Query predictions
        query = PredictionHistory.query
        
        # Filter by symbol if provided
        if symbol:
            query = query.filter_by(symbol=symbol)
            
        # Get latest predictions first
        predictions = query.order_by(PredictionHistory.created_at.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'data': [pred.to_dict() for pred in predictions]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@predictions.route('/api/predictions/history/<symbol>', methods=['GET'])
def get_symbol_history(symbol):
    try:
        # Get latest 30 predictions for the symbol
        predictions = PredictionHistory.query\
            .filter_by(symbol=symbol)\
            .order_by(PredictionHistory.created_at.desc())\
            .limit(30)\
            .all()
            
        return jsonify({
            'success': True,
            'data': [pred.to_dict() for pred in predictions]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@predictions.route('/history')
@login_required
def prediction_history():
    # Lấy lịch sử dự đoán từ database
    history = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
    return render_template('prediction_history.html', predictions=history)

@predictions.route('/api/predictions/history')
@login_required
def get_prediction_history_api():
    try:
        history = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).all()
        predictions = []
        for pred in history:
            predictions.append({
                'symbol': pred.symbol,
                'current_price': pred.current_price,
                'predicted_price': pred.predicted_price,
                'prediction_date': pred.prediction_date.strftime('%Y-%m-%d'),
                'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        return jsonify({'success': True, 'data': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
