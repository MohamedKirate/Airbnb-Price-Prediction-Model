from src.artifacts.artifact import ModelTrainingArtifact
from src.entity.entity import HyperTuningConfig
from src.utils.tools import save_pkl,save_json
from src.logging.loger import logging

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV


import os
import pandas as pd
import numpy as np
from typing import Dict,List
import mlflow
import mlflow.sklearn


class HyperTuning:
    def __init__(self,hyper_tuning_config:HyperTuningConfig,model_training_artifact:ModelTrainingArtifact):
        self.hyper_tuning_config= hyper_tuning_config
        self.model_training_artifact= model_training_artifact

    def evalute(self,true,predict):
        try:
            mse_score= mean_squared_error(true,predict)
            R2_Score= r2_score(true,predict)

            return mse_score,R2_Score
        
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise e
    
    def training(self,x_train,y_train,models,model_params,model_name:str,model_path:str,model_params_path:str):
        try:    
            logging.info(f"Starting hyperparameter tuning for model: {model_name}")
            model= models[model_name]
            model_parms= model_params[model_name]

            if isinstance(model_params_path, str) and os.path.isdir(os.path.dirname(model_params_path)):
                with mlflow.start_run(run_name=f"HyperTuning-{model_name}") as run:
                    
                    mlflow.set_experiment(self.hyper_tuning_config.experiment_name)

                    mlflow.log_params(model_parms)

                    gs = GridSearchCV(model, model_parms, cv=3, scoring="r2", verbose=1, n_jobs=-1)
                    gs.fit(x_train, y_train)

                    model = gs.best_estimator_
                    y_predict = model.predict(x_train)

                    mse_score, r2_score_value = self.evaluate(y_train, y_predict)

                    mlflow.log_metric("MSE", mse_score)
                    mlflow.log_metric("R2_Score", r2_score_value)

                    logging.info(f"Hyperparameter tuning completed for {model_name}. MSE: {mse_score}, R2 Score: {r2_score_value}")

                    save_json(gs.best_params_, model_params_path)
                    save_pkl(gs.best_estimator_, model_path)

                    mlflow.log_artifact(model_params_path)
                    mlflow.sklearn.log_model(model, f"{model_name}_model")
            else:
                raise ValueError(f"Invalid model_params_path: {model_params_path} is not a valid directory or file path.")
        except Exception as e:
            logging.error(f"Error during hyperparameter tuning for {model_name}: {e}")
            raise e
        
    
    def init_hyper_tuning(self,X,Y):
        try:
            logging.info("Initializing hyperparameter tuning process.")
            self.training(X,Y,
                        self.hyper_tuning_config.models,
                        self.hyper_tuning_config.model_params,
                        self.model_training_artifact.model_name,
                        self.hyper_tuning_config.model_path,
                        self.hyper_tuning_config.model_params_path)
        except Exception as e:
            logging.error(f"Error in init_hyper_tuning: {e}")
            raise e