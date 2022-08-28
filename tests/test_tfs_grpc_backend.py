import os
import sys
import json
import requests
import asyncio

import logging

import numpy as np
import unittest

from time import time
from engine.dtype import dtype_map
from utils.io import read_img, encode_jpeg

from engine.tfs_grpc_backend import TFServingGRPCBackend
from tests.config import TEST_TFS_GRPC_PORT, TEST_TFS_RESTAPI_PORT, TEST_TFS_HOST, TEST_MODELS_DIR_PATH, TEST_MEDIA_PATH, TEST_LOG_LEVEL

from tf_load import tf


log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)

# engine config models description
MODELS_TO_TEST = [
                {
                    "name": "mobile_tag_detector",
                    "version": 2,
                    "type": "TFServingGRPCBackend",
                    "path": "mobile_tag_detector",
                    "host": "localhost",
                    "port": TEST_TFS_GRPC_PORT                    
                },
                {
                    "name": "mobile_tag_classifier",
                    "version": 1,
                    "type": "TFServingGRPCBackend",
                    "path": "mobile_tag_classifier",
                    "host": "localhost",
                    "port": TEST_TFS_GRPC_PORT
                    
                },                                
                {
                    "name": "server_digit_detector",
                    "version": 1,
                    "type": "TFServingGRPCBackend",
                    "path": "server_digit_detector",
                    "host": "localhost",
                    "port": TEST_TFS_GRPC_PORT 
                },
                {
                    "name": "mobile_tag_semantic_detector",
                    "version": 1,
                    "type": "TFServingGRPCBackend",
                    "path": "mobile_tag_semantic_detector",
                    "host": "localhost",
                    "port": TEST_TFS_GRPC_PORT 
                }
    ]


tf2np_dtype_map = {
    tf.uint8: np.uint8,
    tf.float32: np.float32
}




class TestTFS_REST_Model(unittest.TestCase):    
          
    
    # def test_prediction(self):
    #     passed = False
    #     log_msg = f"Running prediction test: "
    #     log.info(log_msg)
    #     start_time = time()
    #     predictions = []

    #     for model_instance in models:                     
    #         test_tensor = np.zeros(shape=model_instance._input_shape[1:])
    #         prediction = model_instance.predict([test_tensor])
    #         predictions.append(prediction)
    #         # log.info(prediction)
                
    #     end_time = time()
    #     log.info(f"Prediction test: Ok. Syncrounous inference time: {(end_time-start_time):.2f}")
    #     log.debug(predictions)
    #     passed = True

    #     self.assertEqual(passed, True)
        
    async def run_predictions_async(self, async_tasks):            
        return await asyncio.gather(*async_tasks)


    def test_prediction_async(self):

        models = []
        sem = asyncio.Semaphore(100)
        for model_cfg in MODELS_TO_TEST:
            log_msg = f"Testing model initialization :{model_cfg} "               
            model_instance = TFServingGRPCBackend(model_cfg, sem)
            models.append(model_instance)
            log_msg += "Ok"
            log.info(log_msg)

        passed = False
        log_msg = f"Running async prediction test: "
        log.info(log_msg)
        async_tasks = []          
        start_time = time()

        for model_instance in models:                     
            test_tensor = np.zeros(shape=model_instance._input_shape[1:])
            async_tasks.append(model_instance.predict([test_tensor]))
        
        predictions = asyncio.run(self.run_predictions_async(async_tasks))        

        end_time = time()
        log.info(f"Prediction test: Ok. Async inference time: {(end_time-start_time):.2f}")      
        log.debug(predictions)
        passed = True

        self.assertEqual(passed, True)
        