from SENSOR.pipeline.training_pipeline import TrainPipeline
from SENSOR.exception import SensorException
from SENSOR.logger import logging
import sys

if __name__ == "__main__":
    try:
        logging.info("Starting APS Sensor Training Pipeline...")
        train_pipeline = TrainPipeline()
        
        # Run complete pipeline
        logging.info("Running data ingestion...")
        data_ingestion_artifact = train_pipeline.start_data_ingestion()
        logging.info(f"Data Ingestion Complete: {data_ingestion_artifact}")
        
        logging.info("Running data validation...")
        data_validation_artifact = train_pipeline.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        logging.info(f"Data Validation Complete: {data_validation_artifact}")
        
        logging.info("Running data transformation...")
        data_transformation_artifact = train_pipeline.start_data_transformation(data_validation_artifact=data_validation_artifact)
        logging.info(f"Data Transformation Complete: {data_transformation_artifact}")
        
        logging.info("Running model training...")
        model_trainer_artifact = train_pipeline.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        logging.info(f"Model Training Complete: {model_trainer_artifact}")
        
        logging.info("Running model evaluation...")
        model_evaluation_artifact = train_pipeline.start_model_evaluation(
            data_validation_artifact=data_validation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        logging.info(f"Model Evaluation Complete: {model_evaluation_artifact}")
        
        logging.info("Pushing model...")
        model_pusher_artifact = train_pipeline.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
        logging.info(f"Model Push Complete: {model_pusher_artifact}")
        
        logging.info("✅ Training Pipeline Completed Successfully!")
        
    except SensorException as e:
        logging.error(f"❌ SensorException: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Exception: {e}")
        sys.exit(1)
