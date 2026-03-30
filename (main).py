from SENSOR.exception import SensorException
import os
import sys

from SENSOR.logger import logging
from SENSOR.utils import dump_csv_file_to_mongodb_collection

# def test_exception():
#     try:
#         logging.info("It will show and error after division by zero")
#         a = 1/0
#     except Exception as e:
#         raise SensorException(e,sys)

# ...existing code...
if __name__ == "__main__":
    file_path = r"C:\Users\abhyanand\OneDrive\Documents\GitHub\APS---Sensor\aps_failure_training_set1.csv"
    database_name = "aps"
    collection_name = "sensor"
    dump_csv_file_to_mongodb_collection(file_path, database_name, collection_name)
# ...existing code...








    # try:
    #     test_exception
    # except Exception as e:
    #     print(e)