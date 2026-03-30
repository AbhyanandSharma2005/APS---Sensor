"""
SENSOR Components Module

This module contains all the pipeline components for the APS Sensor Failure Prediction project.
Each component handles a specific stage of the machine learning pipeline.

Components:
-----------
1. DataIngestion: Handles data loading from various sources (CSV, MongoDB)
2. DataValidation: Validates data quality and schema correctness
3. DataTransformation: Performs feature engineering and preprocessing
4. ModelTrainer: Trains machine learning models
5. ModelEvaluation: Evaluates model performance on test data
6. ModelPusher: Pushes best models to production/storage
"""

import logging
from SENSOR.logger import logging

# Configure logger for components module
logger = logging.getLogger(__name__)

# Import component classes (to be implemented)
try:
    from SENSOR.components.data_ingestion import DataIngestion
except ImportError:
    logger.warning("DataIngestion not yet implemented")

try:
    from SENSOR.components.data_validation import DataValidation
except ImportError:
    logger.warning("DataValidation not yet implemented")

try:
    from SENSOR.components.data_transformation import DataTransformation
except ImportError:
    logger.warning("DataTransformation not yet implemented")

try:
    from SENSOR.components.model_trainer import ModelTrainer
except ImportError:
    logger.warning("ModelTrainer not yet implemented")

try:
    from SENSOR.components.model_evaluation import ModelEvaluation
except ImportError:
    logger.warning("ModelEvaluation not yet implemented")

try:
    from SENSOR.components.model_pusher import ModelPusher
except ImportError:
    logger.warning("ModelPusher not yet implemented")


# Export all component classes
__all__ = [
    'DataIngestion',
    'DataValidation',
    'DataTransformation',
    'ModelTrainer',
    'ModelEvaluation',
    'ModelPusher',
]


"""
Pipeline Flow:
==============

1. DataIngestion
   - Loads raw data from CSV files or MongoDB
   - Returns: Raw dataset with train/test split
   
2. DataValidation
   - Validates data schema and quality
   - Checks for missing values, data types, and anomalies
   - Returns: Validation status and report
   
3. DataTransformation
   - Performs feature engineering and preprocessing
   - Handles missing values, encoding, scaling
   - Returns: Transformed features ready for modeling
   
4. ModelTrainer
   - Trains multiple ML models on training data
   - Performs hyperparameter tuning
   - Returns: Trained model and metadata
   
5. ModelEvaluation
   - Evaluates model performance on test data
   - Computes various performance metrics
   - Returns: Evaluation report and comparison
   
6. ModelPusher
   - Pushes the best performing model to production
   - Handles model versioning and storage
   - Returns: Deployment confirmation
"""
