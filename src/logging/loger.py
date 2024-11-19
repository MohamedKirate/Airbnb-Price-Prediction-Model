import logging.handlers
import os
import logging
import sys
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%Y%m%d%H%M%S")}.log'

log_path = os.path.join(os.getcwd(), 'logs')

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__=="__main__":
    logging.info("it's working realy good")