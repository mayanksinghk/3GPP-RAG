import logging.config
import os

def setup_logging(default_level=logging.INFO):
    """
    Initializes application-wide logging configuration.
    Call this ONCE at the absolute entry point (e.g., main.py or wsgi.py).
    """
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Keeps third-party logs alive
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(filename)s %(lineno)d %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {  # Root Logger
                "handlers": ["console", "file"],
                "level": os.getenv("LOG_LEVEL", "INFO"),
                "propagate": True
            },
            "third_party_lib": {  # Mute noisy internal frameworks
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            }
        }
    }
    
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig(log_config)
