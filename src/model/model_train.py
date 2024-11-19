from src.entity.entity import ModelTrainingConfig
from src.artifacts.artifact import ModelTrainingArtifact,DataTransformationArtifact
from src.utils.tools import load_pkl,read_csv
from src.logging.loger import logging

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error,r2_score

import os
import pandas as pd
import numpy as np
from typing import List,Dict


class ModelTraining:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_training_config:ModelTrainingConfig):
        self.data_transformation_artifact= data_transformation_artifact
        self.model_training_config= model_training_config

    
    def get_dependencies(self,data_path:str,preprocess_path:str,target_scaler_path:str):
        try:
            logging.info("Loading dependencies for model training.")
            data= read_csv(data_path)
            preprocess_obj= load_pkl(preprocess_path)
            target_scaler= load_pkl(target_scaler_path)

            return data, preprocess_obj ,target_scaler

        except Exception as e:
            logging.error(f"Error loading dependencies: {e}")
            raise e
    
    def preprocessing(self,data:pd.DataFrame,preprocess_obj,target_scaler,target:str):
        try:

            logging.info("Preprocessing the dataset.")
            X_preprocess= preprocess_obj.transform(data)
            Y_scaler= target_scaler.transform(data[target].values.reshape(-1,1))

            return X_preprocess,Y_scaler
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e
    
    def evaluate(self,true,predict):
        try:
            mse= mean_squared_error(true,predict)
            r2=r2_score(true,predict)
            return mse,r2
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e
    
    def model_training(self,models:Dict[str,any],X,Y):
        try:

            logging.info("Training and evaluating models.")
            model_name_list=[]
            r2_list=[]

            for i in range(len(list(models))):

                model_name = list(models.keys())[i]
                model_name_list.append(model_name)

                logging.info(f"Training model: {model_name}")

                model= list(models.values())[i]

                model.fit(X,Y)

                y_predict= model.predict(X)

                mse_score,R2_score=self.evaluate(Y,y_predict)
                logging.info(f"R^2 Score value : {R2_score}")
                r2_list.append(R2_score)


            data= list(zip(model_name_list,r2_list))
            result= pd.DataFrame(data,columns=["Model Name","R2 Score",])

            logging.info("Selecting the best model based on R^2 score.")

            model_name = result.loc[result['R2 Score'].idxmax(),'Model Name']

            return model_name
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise e
        
    def init_model_training(self):
        try:
            logging.info("Initializing model training process.")
            data,preprocess_obj,target_scaler= self.get_dependencies(self.data_transformation_artifact.data_path,
                                                                    self.data_transformation_artifact.preprocess_path,
                                                                    self.data_transformation_artifact.target_scaler_path)
            X,Y= self.preprocessing(data,
                                    preprocess_obj,
                                    target_scaler,
                                    self.model_training_config.model_target)
            
            best_model_name= self.model_training(self.model_training_config.models,X,Y)

            logging.info(f"Model training completed. Best model: {best_model_name}")

            return best_model_name ,X,Y
        except Exception as e:
            logging.error(f"Error in model training initialization: {e}")
            raise e