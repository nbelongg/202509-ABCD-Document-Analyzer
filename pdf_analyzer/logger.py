import logging
import logging.config
import os
import pytz
from datetime import datetime

LOG_DIR = os.path.join('logs')
API_LOG_FILE = os.path.join(LOG_DIR, 'api.log')
STREAMLIT_LOG_FILE = os.path.join(LOG_DIR, 'streamlit.log')
ERROR_LOG_FILE = os.path.join(LOG_DIR, 'error.log')

os.makedirs(LOG_DIR, exist_ok=True)

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=pytz.timezone('Asia/Kolkata'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        return s

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            '()': ISTFormatter,
            'format': '%(asctime)s [%(levelname)s] %(name)s [%(funcName)s]: %(message)s'
        },
    },
    'handlers': {
        'api_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': API_LOG_FILE,
            'formatter': 'standard',
            'maxBytes': 1 * 1024 * 1024,
            'backupCount': 25,
        },
        'streamlit_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': STREAMLIT_LOG_FILE,
            'formatter': 'standard',
            'maxBytes': 1 * 1024 * 1024,
            'backupCount': 25,
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': ERROR_LOG_FILE,
            'formatter': 'standard',
            'maxBytes': 50 * 1024 * 1024,  # 50 MB
            'backupCount': 5,
        },
    },
    'loggers': {
        'api': {
            'handlers': ['api_file', 'console', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'streamlit': {
            'handlers': ['streamlit_file', 'console', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn': {
            'handlers': ['api_file', 'console', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn.error': {
            'handlers': ['api_file', 'console', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'uvicorn.access': {
            'handlers': ['api_file', 'console', 'error_file'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

# Set up logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create loggers at startup
api_logger = logging.getLogger('api')
streamlit_logger = logging.getLogger('streamlit')


def get_logger(service_name):
    if service_name == 'api':
        return api_logger
    elif service_name == 'streamlit':
        return streamlit_logger
    else:
        raise ValueError(f"No logger configured for service: {service_name}")
