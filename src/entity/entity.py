import os
from typing import List,Dict
from src.constants.constant import (DATA_INGESTION_OUTPUT_DIR,DATA_INGESTION_DATA_PATH,DATA_INGESTION_INPUT_PATH,DATA_INGESTION_TARIN_PATH,
                                    DATA_INGESTION_TARIN_PATH,DATA_INGESTION_TEST_PATH,DATA_INGESTION_SPLIT_SIZE,DATA_INGESTION_RANDOM_STATE,
                                    DATA_INGESTION_INPUT_DIR,DATA_INGESTION_COLUMNS,DATA_INGESTION_DROPNA_FAETURE,DATA_INGESTION__NUMERICAL_NA_COLUMNS)


from src.constants.constant import (DATA_TRANSFORMATION_CATEGORICAL_FEATURE,DATA_TRANSFORMATION_CATEGORICAL_TRANSFORM_FEATURES,DATA_TRANSFORMATION_NUMERICAL_FEATURES,
                                    DATA_TRANSFORMATION_FEATURES,DATA_TRANSFORMATION_DIR,DATA_TRANSFORMATION_PREPROCESS_DIR,DATA_TRANSFORMATION_PREPROCESS_PATH,
                                    DATA_TRANSFORMATION_DATA_PATH,DATA_TRANSFORMATION_TARGET_FEATURE,DATA_TRANSFORMATION_SCALER_PATH,DATA_TRANSFORMATION_NOISE,
                                    DATA_TRANSFORMATION_NEW_FEATURE_NAME)

from src.constants.constant import MODEL_TRAINING_MODELS,MODEL_TRAINING_TARGET

from src.constants.constant import (HYPER_TUNING_MODEL_PARAMS,HYPER_TUNING_MODEL_PATH,HYPER_TUNING_MODELS,HYPER_TUNING_MODEL_PARAMS_PATH,
                                    HYPER_TUNING_EXPERIMENT_NAME)

import os



class DataIngestionConfig:
    def __init__(self):
        self.input_path :str =os.path.join(DATA_INGESTION_INPUT_DIR,DATA_INGESTION_INPUT_PATH)
        self.data_path :str =os.path.join(DATA_INGESTION_OUTPUT_DIR,DATA_INGESTION_DATA_PATH)
        self.train_path :str =os.path.join(DATA_INGESTION_OUTPUT_DIR,DATA_INGESTION_TARIN_PATH)
        self.test_path :str =os.path.join(DATA_INGESTION_OUTPUT_DIR,DATA_INGESTION_TEST_PATH)
        self.split_size :float = DATA_INGESTION_SPLIT_SIZE
        self.random_state :int = DATA_INGESTION_RANDOM_STATE
        self.data_columns:List[str] =DATA_INGESTION_COLUMNS
        self.drop_na_column :str = DATA_INGESTION_DROPNA_FAETURE
        self.numerical_na_columns:List[str]=DATA_INGESTION__NUMERICAL_NA_COLUMNS

class DataTransformationConfig:
    def __init__(self):
        self.input_dir= DATA_TRANSFORMATION_DIR
        self.preprocess_dir= DATA_TRANSFORMATION_PREPROCESS_DIR
        self.data_path= os.path.join(self.input_dir,DATA_TRANSFORMATION_DATA_PATH)
        self.preprocess_path= os.path.join(self.input_dir,self.preprocess_dir,DATA_TRANSFORMATION_PREPROCESS_PATH)
        self.scaler_path= os.path.join(self.input_dir,self.preprocess_dir,DATA_TRANSFORMATION_SCALER_PATH)

        self.data_features=DATA_TRANSFORMATION_FEATURES
        self.categorical_features=DATA_TRANSFORMATION_CATEGORICAL_FEATURE
        self.numerical_features= DATA_TRANSFORMATION_NUMERICAL_FEATURES
        self.categorical_transform_features= DATA_TRANSFORMATION_CATEGORICAL_TRANSFORM_FEATURES

        self.target= DATA_TRANSFORMATION_TARGET_FEATURE
        self.noise:str= DATA_TRANSFORMATION_NOISE
        self.new_feature_name:str= DATA_TRANSFORMATION_NEW_FEATURE_NAME

class ModelTrainingConfig:
    def __init__(self):
        self.models:Dict[str,any]= MODEL_TRAINING_MODELS
        self.model_target:str=MODEL_TRAINING_TARGET

class HyperTuningConfig:
    def __init__(self):
        os.makedirs('model', exist_ok=True)

        self.model_params:Dict[str,any]=HYPER_TUNING_MODEL_PARAMS
        self.models:Dict[str,any]=HYPER_TUNING_MODELS
        
        self.model_path:str= HYPER_TUNING_MODEL_PATH
        self.model_params_path:str=HYPER_TUNING_MODEL_PARAMS_PATH
        self.experiment_name :str= HYPER_TUNING_EXPERIMENT_NAME


        






