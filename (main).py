import os
import sys
import logging

from SENSOR.utils2 import dump_csv_file_to_mongodb_collecton

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
    dump_csv_file_to_mongodb_collecton(file_path, database_name, collection_name)
    print("✅ Data successfully loaded to MongoDB!")
# ...existing code...








    # try:
    #     test_exception
    # except Exception as e:
    #     print(e)