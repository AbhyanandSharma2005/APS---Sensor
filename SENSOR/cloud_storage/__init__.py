"""
AWS S3 Storage Module
Handles all cloud storage operations with AWS S3
"""

import sys
import logging
from typing import Optional, Dict, Any

from SENSOR.cloud_storage.s3_storage import S3Storage

logger = logging.getLogger(__name__)

__all__ = ['S3Storage']
