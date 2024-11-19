import os
from typing import Dict,List

##       DATA INGESTION
DATA_INGESTION_INPUT_DIR:str='data'
DATA_INGESTION_OUTPUT_DIR:str='artifacts'
DATA_INGESTION_INPUT_PATH:str='listings.csv'
DATA_INGESTION_TARIN_PATH='train.csv'
DATA_INGESTION_TEST_PATH:str='test.csv'
DATA_INGESTION_DATA_PATH:str= 'data.csv'
DATA_INGESTION_SPLIT_SIZE:float=0.1
DATA_INGESTION_RANDOM_STATE:int=42
DATA_INGESTION_COLUMNS:List[str]=['neighbourhood_group_cleansed', 'room_type','accommodates', 'bathrooms', 'bedrooms', 
                                  'beds', 'price', 'availability_365','minimum_nights']

DATA_INGESTION_DROPNA_FAETURE :str = 'price'
DATA_INGESTION__NUMERICAL_NA_COLUMNS :List[str]= ['bathrooms','bedrooms','beds']


## DATA TRANSFORMATION
DATA_TRANSFORMATION_DIR:str='artifacts'
DATA_TRANSFORMATION_DATA_PATH:str= 'data.csv'

DATA_TRANSFORMATION_TARGET_FEATURE:str='price'
DATA_TRANSFORMATION_NEW_FEATURE_NAME:str='text'

DATA_TRANSFORMATION_FEATURES:List[str]=['neighbourhood_group_cleansed', 'room_type','accommodates', 'bathrooms',
                                         'bedrooms', 'beds', 'price', 'availability_365','minimum_nights']
DATA_TRANSFORMATION_CATEGORICAL_TRANSFORM_FEATURES:List[str]=['neighbourhood_group_cleansed', 'room_type']
DATA_TRANSFORMATION_PREPROCESS_DIR='preprocess'
DATA_TRANSFORMATION_PREPROCESS_PATH:str='preprocess.pkl'
DATA_TRANSFORMATION_SCALER_PATH:str='scaler.pkl'
DATA_TRANSFORMATION_NUMERICAL_FEATURES:List[str]=['accommodates', 'bathrooms', 'bedrooms', 'beds', 'availability_365','minimum_nights']
DATA_TRANSFORMATION_CATEGORICAL_FEATURE:str='text'
DATA_TRANSFORMATION_NOISE:str= '/'

##  Model Training 
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

MODEL_TRAINING_MODELS:Dict[str,any]={
    "XGBRegressor":XGBRegressor(),
    "CatBoostRegressor":CatBoostRegressor(),
    "RandomForestRegressor":RandomForestRegressor(),
    "GradientBoostingRegressor":GradientBoostingRegressor(),
    "AdaBoostRegressor":AdaBoostRegressor()
}
MODEL_TRAINING_TARGET:str='price'


## Hyper Tuning 

HYPER_TUNING_MODEL_PATH = os.path.join("model", "model.pkl")
HYPER_TUNING_MODEL_PARAMS_PATH = os.path.join("model", "model_params.json")


HYPER_TUNING_MODELS:Dict[str,any]={
    "XGBRegressor":XGBRegressor(),
    "CatBoostRegressor":CatBoostRegressor(),
    "RandomForestRegressor":RandomForestRegressor(),
    "GradientBoostingRegressor":GradientBoostingRegressor(),
    "AdaBoostRegressor":AdaBoostRegressor()
}

HYPER_TUNING_MODEL_PARAMS:Dict[str,any]={
    'XGBRegressor' :{
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.6, 0.8],
    },
    'CatBoostRegressor':{
        "iterations": [500, 700],
        "learning_rate": [0.01, 0.05],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
        "bagging_temperature": [0.2, 0.5],
    
    },
    'RandomForestRegressor':{
        'n_estimators': [100, 200,300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
},

    'GradientBoostingRegressor':{
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    
    },

    'AdaBoostRegressor':{
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 1.0],
        "loss": ["linear", "square", "exponential"],
        "random_state": [42],
    }
}

HYPER_TUNING_EXPERIMENT_NAME:str="House_Prediction_Model_Tuning"



