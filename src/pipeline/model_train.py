from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.artifacts.artifact import DataIngestionArtifact,DataTransformationArtifact,ModelTrainingArtifact
from src.entity.entity import DataIngestionConfig,DataTransformationConfig,ModelTrainingConfig,HyperTuningConfig
from src.model.model_train import ModelTraining
from src.model.hyper_tuning import HyperTuning

if __name__=="__main__":
    data_ingestion_config= DataIngestionConfig()
    data_ingestion= DataIngestion(data_ingestion_config)
    data_ingestion.init_data_ingestion()

    data_transformation_config= DataTransformationConfig()
    data_transformation= DataTransformation(data_transformation_config)
    data_path,preprocess_path,scaler_path=data_transformation.init_data_transformation()

    model_training_config=ModelTrainingConfig()

    data_transformation_artifact= DataTransformationArtifact(data_path,preprocess_path,scaler_path)

    model_training= ModelTraining(data_transformation_artifact,model_training_config)
    model_name,X,Y=model_training.init_model_training()

    model_training_artifact=ModelTrainingArtifact(model_name)

    hyper_tuning_config= HyperTuningConfig()

    hyper_tuning = HyperTuning(hyper_tuning_config,model_training_artifact)
    hyper_tuning.init_hyper_tuning(X,Y)