import logging
import logging.config
import os
import pytz
from datetime import datetime


LOG_DIR = os.path.join('logs')
API_LOG_FILE = os.path.join(LOG_DIR, 'api.log')
STREAMLIT_LOG_FILE = os.path.join(LOG_DIR, 'streamlit.log')

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
            'format': '%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
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
        }
    },
    'loggers': {
        'api': {
            'handlers': ['api_file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'streamlit': {
            'handlers': ['streamlit_file', 'console'],
            'level': 'INFO',
            'propagate': False,
        }
    }
}

# Set up logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Create loggers at startup
api_logger = logging.getLogger('api')
streamlit_logger = logging.getLogger('streamlit')
