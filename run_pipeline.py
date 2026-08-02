import sys
import os

# Add SENSOR folder to path to handle import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create alias for case-insensitive imports
import SENSOR as sensor
sys.modules['sensor'] = sensor

# Now run the training pipeline
from SENSOR.pipeline.training_pipeline import TrainPipeline
from SENSOR.exception import SensorException
from SENSOR.logger import logging

if __name__ == "__main__":
    try:
        logging.info("=" * 80)
        logging.info("Starting APS Sensor Training Pipeline...")
        logging.info("=" * 80)
        
        train_pipeline = TrainPipeline()
        
        logging.info("\n📊 Step 1: Running data ingestion...")
        data_ingestion_artifact = train_pipeline.start_data_ingestion()
        logging.info(f"✅ Data Ingestion Complete!")
        
        logging.info("\n🔍 Step 2: Running data validation...")
        data_validation_artifact = train_pipeline.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        logging.info(f"✅ Data Validation Complete!")
        
        logging.info("\n🔄 Step 3: Running data transformation...")
        data_transformation_artifact = train_pipeline.start_data_transformation(data_validation_artifact=data_validation_artifact)
        logging.info(f"✅ Data Transformation Complete!")
        
        logging.info("\n🤖 Step 4: Running model training...")
        model_trainer_artifact = train_pipeline.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        logging.info(f"✅ Model Training Complete!")
        
        logging.info("\n📈 Step 5: Running model evaluation...")
        model_evaluation_artifact = train_pipeline.start_model_evaluation(
            data_validation_artifact=data_validation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        logging.info(f"✅ Model Evaluation Complete!")
        
        logging.info("\n☁️ Step 6: Pushing model...")
        model_pusher_artifact = train_pipeline.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
        logging.info(f"✅ Model Push Complete!")
        
        logging.info("\n" + "=" * 80)
        logging.info("🎉 Training Pipeline Completed Successfully!")
        logging.info("=" * 80)
        
    except SensorException as e:
        logging.error(f"❌ SensorException: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Exception: {e}")
        sys.exit(1)
