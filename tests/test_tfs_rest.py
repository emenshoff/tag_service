import os
import sys
import json
import requests

import logging

import numpy as np
import unittest

from time import time
from engine.dtype import dtype_map
from utils.io import read_img, encode_jpeg

from engine.tfs_exception import TFSSessionException
from tests.config import TEST_TFS_GRPC_PORT, TEST_TFS_RESTAPI_PORT, TEST_TFS_HOST, TEST_MODELS_DIR_PATH, TEST_LOG_LEVEL

from tf_load import tf


log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)


MODELS_TO_TEST = [    
    {"name": "mobile_tag_detector", "version": 2},
    {"name": "mobile_tag_classifier",  "version": 1},
    {"name": "server_digit_detector",  "version": 1},
    {"name": "mobile_tag_semantic_detector",   "version": 2}
]


tf2np_dtype_map = {
    tf.uint8: np.uint8,
    tf.float32: np.float32
}


class TestTFS_GRPC(unittest.TestCase):
           
    def test_prediction(self):
        passed = True
        for model_item in  MODELS_TO_TEST:
            try:
                self.run_model_prediction(model_item)      
            except (TFSSessionException, Exception) as ex:
                passed = False

        self.assertEqual(passed, True)


    def run_model_prediction(self, model_item):

        name = model_item["name"]
        version = model_item["version"]
        PROFILE_PATH = os.path.join(TEST_MODELS_DIR_PATH, f"{name}/{version}/profile.json")
        with open(PROFILE_PATH) as file:
            profile = json.load(file)
      

        addr = f"http://{TEST_TFS_HOST}:{TEST_TFS_RESTAPI_PORT}/v1/models/{name}"
        input_options = profile["export_targets"]["saved_model"]["options"]["inference_options"]["inputs"][0]
        dtype = tf2np_dtype_map[dtype_map[input_options["dtype"]]]
        height =  input_options["shape"]["height"]
        width = input_options["shape"]["width"]
        channels = input_options["shape"]["channels"]
        batch_size = input_options["shape"]["batch_size"]      
        shape = (batch_size, height, width, channels)

        outputs =  [output["name"] for output in profile["export_targets"]["saved_model"]["options"]["inference_options"]["outputs"]]
        # print(outputs)

        test_tensor = np.zeros(shape=shape, dtype=dtype)    
        log_msg = f'Checking REST_API access to tensorflow-serving model {name} ({addr}): ' 
        predictions = dict()

        # test request to check remote model signature is the same as in model profile loaded

        data = json.dumps({"signature_name": "serving_default", "instances": test_tensor.tolist()}) 
        headers = {"content-type": "application/json"} 
        request_msg = f'{addr}:predict'
        try:
            start_time = time()
            json_response = requests.post(request_msg, data=data, headers=headers) # post request to the served model
            end_time = time()
            if json_response.status_code == 200:                        
        #         print(f"response: {json_response.text}")
                # print(f"response: {json_response.json()}")
                predictions = json_response.json()['predictions']
                log_msg += f"Ok\n\t inference time: {(end_time - start_time):.2f}\n\t prediction: {predictions}"
                log.debug(log_msg)                  
            else:
                log_msg += f"Failure. response code: {json_response.status_code}\n\t {json_response}. Please check model profile!"
                log.debug(log_msg)
        except Exception as ex:
            msg = f'failure to connect to tensorflow-serving server : {ex}'
            log.error(msg)
            raise TFSSessionException(msg)
        return predictions