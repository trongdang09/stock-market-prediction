from flask import Flask
from flask_login import LoginManager
from config import Config
from app.database import db
from app.ai_models.prediction_service import prediction_service

login_manager = LoginManager()

def create_app():
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    app.config.from_object(Config)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 phút

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from app.routes.auth import auth
    from app.routes.main import main
    from app.routes.api import api
    from app.routes.predictions import predictions

    app.register_blueprint(auth)
    app.register_blueprint(main)
    app.register_blueprint(api)
    app.register_blueprint(predictions)

    app.secret_key = 'your-secret-key-here'  # Thay đổi thành một key phức tạp hơn trong môi trường production

    return app