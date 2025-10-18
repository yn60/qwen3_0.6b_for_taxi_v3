# File: /gym-taxi-qwen3/gym-taxi-qwen3/config/settings.py

import os

class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "replace_me")
    DEBUG = os.environ.get("DEBUG", "False") == "True"
    GYM_ENV_NAME = os.environ.get("GYM_ENV_NAME", "Taxi-v3")

    QWEN_MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-0.6B")
    QWEN_DEVICE = os.environ.get("QWEN_DEVICE")
    QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "64"))
    QWEN_TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.3"))

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}