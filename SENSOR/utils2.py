from dotenv import load_dotenv
import pymongo
import pandas as pd 
import logging
import json
import certifi
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

MONGODB_URL_KEY = "MONGO_DB_URL"
ca = certifi.where()

def get_mongo_client():
    """Get MongoDB client connection"""
    try:
        mongo_db_url = os.getenv(MONGODB_URL_KEY)
        if "localhost" in mongo_db_url:
            client = pymongo.MongoClient(mongo_db_url)
        else:
            client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
        return client
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise

def dump_csv_file_to_mongodb_collecton(file_path:str, database_name:str, collection_name:str)->None:
    try:
        client = get_mongo_client()
        df = pd.read_csv(file_path)
        logging.info(f"Rows and columns: {df.shape}")

        df.reset_index(drop=True, inplace=True)
        json_records = list(json.loads(df.T.to_json()).values())

        client[database_name][collection_name].insert_many(json_records)
        logging.info(f"✅ Successfully inserted {len(json_records)} records to {database_name}.{collection_name}")
        client.close()
    except Exception as e : 
        logging.error(f"❌ Error: {e}")
        raise



"""  
.env
aps file insert
sensr : __init__ file for laoding env
utils file  for dumping data into database
config for reading mongodb url 
"""


"""
main for running 
entity - config for creating folder realted to data ingestion
in training init _ all tle constant for trainning is kept there 
in config put the schemas over there 




we crete training pipeline and
artrifacts we put code
then run the pipeline 
"""