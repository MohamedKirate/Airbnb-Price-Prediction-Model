from src.utils.tools import load_pkl
from src.logging.loger import logging

def prediction(X):
    try:
    
        logging.info("Loading model and preprocessing objects.")
        model = load_pkl('model/model.pkl')
        preprocess_obj = load_pkl('artifacts/preprocess/preprocess.pkl')
        scaler = load_pkl('artifacts/preprocess/scaler.pkl')

        logging.info("Transforming input data.")
        X_scaled = preprocess_obj.transform(X)

        logging.info("Making predictions with the trained model.")
        predictions = model.predict(X_scaled)

        logging.info("Applying inverse transformation to predictions.")
        true_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

        return true_predictions
    except Exception as e:
        logging.error(f"Error in model function: {e}")
        raise e



    