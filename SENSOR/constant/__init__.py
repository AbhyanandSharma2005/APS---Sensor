"""
Constant Module
Contains all constant values used throughout the APS Sensor project
"""

from SENSOR.constant.application import *
from SENSOR.constant.database import *
from SENSOR.constant.env_variable import *
from SENSOR.constant.s3_bucket import *
from SENSOR.constant.training_pipeline import *

__all__ = [
    'APP_NAME',
    'APP_VERSION',
    'LOG_DIR',
    'DB_URL',
    'DB_NAME',
    'DB_PORT',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_REGION',
    'S3_BUCKET_NAME',
    'S3_MODEL_KEY_PREFIX',
    'TARGET_COLUMN',
    'PIPELINE_NAME',
    'ARTIFACT_DIR',
]
