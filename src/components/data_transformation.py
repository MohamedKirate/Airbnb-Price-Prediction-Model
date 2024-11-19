from src.artifacts.artifact import DataTransformationArtifact
from src.entity.entity import DataTransformationConfig
from src.logging.loger import logging
from src.utils.tools import read_csv,save_pkl,save_csv

import pandas as pd
import os
import numpy as np
from typing import List,Dict
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig):
        self.data_transformation_config= data_transformation_config

    def get_data(self,data_path:pd.DataFrame,data_columns:List[str]):
        try:
            logging.info(f"Reading data from {data_path}")
            data=read_csv(data_path)

            logging.info(f"Filtering columns: {data_columns}")
            data=data[data_columns]
        
            return data
        except Exception as e:
            logging.error(f"Error in getting data: {e}")
            raise e
    
    def feature_engenering(self,data:pd.DataFrame,data_path,categorical_trans_feature:List[str],feature_name:str,noise:str):
        try:
            logging.info(f"Creating feature '{feature_name}' by combining {categorical_trans_feature}")

            data[feature_name]= data[categorical_trans_feature].apply(lambda row: ' '.join(row.values),axis=1)
            data[feature_name]= data[feature_name].str.replace(noise,' ',regex=True)

            logging.info(f"Dropping original categorical features: {categorical_trans_feature}")
            data= data.drop(columns=categorical_trans_feature,axis=1)

            save_csv(data,data_path)
            return data
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise e


    def transformation_pipeline(self,data:pd.DataFrame,preprocess_path:str,target_scaler_path:str,categorical_columns:List[str]
                                ,numerical_columns:List[str],target:str):
        try:
            logging.info("Creating transformation pipelines")
            data= data
            categorical_pipeline=Pipeline([
            ('Tfidf',TfidfVectorizer()),
                ])

            numerical_pipeline=Pipeline([
                ('scaler',MinMaxScaler()),
                ])


            preprocess_obj=ColumnTransformer([
                ('categorical_pipeline',categorical_pipeline,categorical_columns),
                ('numerical_pipeline',numerical_pipeline,numerical_columns)
                ])
            
            preprocess_obj=preprocess_obj.fit(data)
            
            logging.info("Fitting target scaler")
            scaler= MinMaxScaler()
            scaler=scaler.fit(data[[target]])

            logging.info(f"Saving preprocessing object to {preprocess_path}")
            save_pkl(preprocess_obj,preprocess_path)

            logging.info(f"Saving target scaler to {target_scaler_path}")
            save_pkl(scaler,target_scaler_path)
            
        except Exception as e:
            logging.error(f"Error in transformation pipeline: {e}")
            raise e


    def init_data_transformation(self):
        try:
            logging.info("Starting data transformation process")
            data= self.get_data(self.data_transformation_config.data_path,
                                self.data_transformation_config.data_features)
            
            data= self.feature_engenering(data,self.data_transformation_config.data_path,
                                          self.data_transformation_config.categorical_transform_features,
                                        self.data_transformation_config.new_feature_name,
                                        self.data_transformation_config.noise)
            self.transformation_pipeline(data,self.data_transformation_config.preprocess_path,
                                         self.data_transformation_config.scaler_path,self.data_transformation_config.categorical_features,
                                         self.data_transformation_config.numerical_features,self.data_transformation_config.target)
            
            return (
                self.data_transformation_config.data_path,
                self.data_transformation_config.preprocess_path,
                self.data_transformation_config.scaler_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation initialization: {e}")
            raise e
        
