from SENSOR.exception import SensorException
import os
import sys

from SENSOR.logger import logging

def test_exception():
    try:
        logging.info("It will show and error after division by zero")
        a = 1/0
    except Exception as e:
        raise SensorException(e,sys)






if __name__ == "__main__":         #this is the module execution control part
    try:
        test_exception
    except Exception as e:
        print(e)