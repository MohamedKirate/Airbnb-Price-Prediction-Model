import os
import pandas as pd
import numpy as np
from typing import List,Dict

from src.entity.entity import DataIngestionConfig
from src.artifacts.artifact import DataIngestionArtifact
from src.logging.loger import logging
from src.utils.tools import read_csv,save_csv,split_data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        self.data_ingestion_config= data_ingestion_config

    def data_cleaning(self,data:pd.DataFrame,dropna_column:str,drop_numerical_na:List[str]):
        try:
            data= data.dropna(subset=[dropna_column])
            logging.info(f"Cleaning data: Dropping rows with NaN in column {dropna_column}")

            data[drop_numerical_na]=data[drop_numerical_na].fillna(0)
            logging.info(f"Filling NaN in numerical columns {drop_numerical_na} with 0")

            return data
        except Exception as e:
            logging.error(f"Error during data cleaning: {e}")
            raise e
    
    def get_data(self,data_path,data_columns,dropna_column,drop_numerical_na):
        try:
            logging.info(f"Reading data from {data_path}")
            data= read_csv(data_path)

            logging.info(f"Selecting columns: {data_columns}")
            data= data[data_columns]

            logging.info("Performing data cleaning")
            data= self.data_cleaning(data,dropna_column,drop_numerical_na)

            return data
        except Exception as e:
            logging.error(f"Error during data retrieval: {e}")
            raise e

    def init_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")

            data= self.get_data(self.data_ingestion_config.input_path,
                                self.data_ingestion_config.data_columns,
                                self.data_ingestion_config.drop_na_column,
                                self.data_ingestion_config.numerical_na_columns)
            
            logging.info(f"Saving cleaned data to {self.data_ingestion_config.data_path}")
            save_csv(data,self.data_ingestion_config.data_path)
            
            logging.info("Splitting data into train and test sets")
            train,test=split_data(data,self.data_ingestion_config.split_size,self.data_ingestion_config.random_state)

            logging.info(f"Saving training data to {self.data_ingestion_config.train_path}")
            save_csv(train,self.data_ingestion_config.train_path)

            logging.info(f"Saving testing data to {self.data_ingestion_config.test_path}")
            save_csv(test,self.data_ingestion_config.test_path)

            return DataIngestionArtifact(
                self.data_ingestion_config.data_path,
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )
        
        except Exception as e:
            logging.error(f"Error during data ingestion process: {e}")
            raise e