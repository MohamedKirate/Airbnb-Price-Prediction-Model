from dataclasses import dataclass
import os
from typing import List,Dict


@dataclass
class DataIngestionArtifact:
    data_path:str
    train_path:str
    test_path:str

@dataclass
class DataTransformationArtifact:
    data_path:str
    preprocess_path:str
    target_scaler_path:str

@dataclass
class ModelTrainingArtifact:
    model_name:str
