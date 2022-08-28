import logging
from config import MODELS_DIR_PATH, PORT, HOST, API_VERSION

TEST_LOG_LEVEL = logging.DEBUG

TEST_MEDIA_PATH = "/media/nvme1tb/datasets/aromam/media_old/"
TEST_REQUESTS_PATH =  "./requests/"
SOURCE_DATASET_BASE_PATH = "/media/datasets/"
TEST_DATSET_DIR = "./data/test_datasets"
TEST_PROFILE = "aromam"

# tensorflow/serving tests options
TEST_TFS_HOST = '192.168.88.3'
TEST_TFS_RESTAPI_PORT = '8501'
TEST_TFS_GRPC_PORT = '8500'

# REST API test options
TEST_API_VERSION = API_VERSION
TEST_HOST = '192.168.88.3'
TEST_PORT = 5005


# test models options
TEST_MODELS_DIR_PATH = MODELS_DIR_PATH


ENGINE_INPUT_SHAPE = (1333,1333,3)