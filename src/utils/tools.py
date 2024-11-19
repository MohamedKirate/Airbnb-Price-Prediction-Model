import os
import pandas as pd
import  joblib
from sklearn.model_selection import train_test_split
from src.logging.loger import logging


import pandas as pd
import joblib
import json


def read_csv(path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path)
        return data
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        raise e

def save_csv(data: pd.DataFrame, path: str):
    try:
        data.to_csv(path, index=False)
        logging.info(f"CSV saved successfully to {path}")
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        raise e

def load_pkl(path: str):
    try:
        with open(path, 'rb') as file:
            pkl_file = joblib.load(file)
        return pkl_file
    except Exception as e:
        logging.error(f"Error loading pickle: {e}")
        raise e

def save_pkl(obj, path: str):
    try:
        with open(path, 'wb') as file:
            joblib.dump(obj, file)
        logging.info(f"Pickle saved successfully to {path}")
    except Exception as e:
        logging.error(f"Error saving pickle: {e}")
        raise e


def split_data(data:pd.DataFrame,split_size:float,random_state:int):
    try:

        train,test= train_test_split(data,test_size=split_size,random_state=random_state)
        logging.info(f"Data split successfully: {train.shape} training samples, {test.shape} testing samples.")
        return train,test
    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        raise e

def train_test_split_tool(data_x:pd.DataFrame,data_y:pd.DataFrame,split_size:float,random_state:int):
    try:
        x_train,x_test,y_train,y_test= train_test_split(data_x,data_y,test_size=split_size,random_state=random_state)
        logging.info(f"Data split successfully.")
        logging.info(f"training samples: x train {x_train.shape} and y train {y_train.shape}.")
        logging.info(f"testing samples: x train {x_test.shape} and y train {y_test.shape}.")
        return x_train,x_test,y_train,y_test
    except Exception as e:
        raise e
    
def save_json(object, path: str):
    try:
        with open(path, 'w') as file: 
            json.dump(object, file, indent=4)
        logging.info(f"JSON saved successfully to {path}.")
    except Exception as e:
        logging.error(f"Failed to save JSON to {path}: {e}")
        raise e


