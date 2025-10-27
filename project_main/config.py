import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'random-forest-secret-key-2024'
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Model default parameters
    DEFAULT_N_ESTIMATORS = 100
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    
    # Visualization settings
    PLOT_STYLE = 'seaborn'
    PLOT_COLORS = ['#2c5aa0', '#28a745', '#ffc107', '#dc3545', '#17a2b8']
    
    @staticmethod
    def init_app(app):
        # Ensure upload and results directories exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}