'''
Inference backend: tensorflow/servinf REST API 
'''


import json
import asyncio
import aiohttp
import requests
import logging

import numpy as np

from time import time

from config import PREPROCESSING_DEVICE

from engine.inference_backend import InferenceBackend
from engine.tfs_exception import TFSSessionException
from engine.dtype import dtype_map, tf2np_dtype_map
from tf_load import tf

log = logging.getLogger('models')


class TFServingRESTAPIBackend(InferenceBackend):
    '''TFServing model (REST API)  '''
    def __init__(
                    self, 
                    model_cfg: dict,                # model profile
                    inference_lock,                 # lock or semaphore for async inference
                    supports_batch:bool=False       # optional batched inference
        ):
        super().__init__(model_cfg, inference_lock, supports_batch) 

        # checking acces to tensorflow serving server  
        self._url = f"http://{self._host}:{self._port}/v1/models/{self._path}"           
        # test request to check remote model signature is the same as in model profile loaded        
        test_tensor = np.zeros(shape=self._input_shape[1:], dtype=tf2np_dtype_map[self._input_dtype])

        self.tag = f"{self._handler_name} ({self._url})"        
        
        log_msg = f"{self.tag}: Checking REST_API access to tensorflow-serving model... "
        log.info(log_msg)

        async def prediction_task(test_tensor):
            task = asyncio.create_task(self.predict([test_tensor]))
            return await task

        try:            
            running_loop = asyncio.get_event_loop()     
            # check if we are in runing event loop
            if running_loop is not None and running_loop.is_running: 
                test_prediction = asyncio.run_coroutine_threadsafe(
                    self.predict([test_tensor]), 
                    running_loop)   
            # execute in new event loop         
            else:
                test_prediction = asyncio.run(prediction_task(test_tensor))   
            log_msg = f"{self.tag} Ok! Ready for inference."
            log.info(log_msg)
        except (TFSSessionException, Exception) as ex:            
            log_msg = f"{self.tag} Initialization failure! Please check if tensorflow/serving is runing or check model profile!\n\t error: {ex}"
            log.critical(log_msg)
            raise TFSSessionException(log_msg)


    class NumpyEncoder(json.JSONEncoder):
        '''
        numpy serialization
        '''
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    

    def preprocess(self, batch:list):
        # just resize for tensorflow/serving
        return super().preprocess(batch)


    def postprocess(self, batch:list):
        # processed = np.squeeze(batch)       
        return batch


    async def _predict(self, tensor):
        '''
        task for async single prediction
        using tensorflow/serving REST API
        '''

        # using numpy arrays       
        if isinstance(tensor, tf.Tensor):
            tensor = tensor.numpy()

        # request parameters
        data = json.dumps({"signature_name": "serving_default", "instances": tensor.tolist()})#, cls=self.NumpyEncoder)
        headers = {"content-type": "application/json"} 
        request_msg = f'{self._url}:predict'    

        log_msg = f"{self.tag}: running async prediction... {request_msg} "
        log.debug(log_msg)
        
        # trying to connect
        start_time = time()
        try:
            async with self._lock:
                async with aiohttp.ClientSession() as session:
                        async with session.post(request_msg, data=data, headers=headers) as response:
                            
                            if response.status == 200:
                                try:
                                    content = await response.read()
                                    if content is not None:                                        
                                        prediction = await response.json() 
                                        prediction = prediction['predictions'][0]
                                        end_time = time()
                                        inference_time = end_time - start_time  
                                        log_msg = f"Async prediction: Success!\n\t inference time: {inference_time:.2f}\n\t response:\n\t {content}"                                                                             
                                        log.debug(log_msg)
                                    else:
                                        err_msg = f"empty response"
                                        log_msg = f"Async prediction: Failure, {err_msg}"
                                        log.debug(log_msg)
                                        raise TFSSessionException(err_msg)
                                except Exception as ex:
                                    err_msg = f"content read error: {ex}"
                                    log_msg = f"Async prediction: Failure, {err_msg}"
                                    log.debug(log_msg)
                                    raise TFSSessionException(err_msg)
                            else:
                                err_msg = f"Connection error: HTTP Status code: {response.status}"
                                log_msg = f"Async prediction: Failure, {err_msg}"
                                log.debug(log_msg)
                                raise TFSSessionException(err_msg)
                                
        except (TFSSessionException, Exception) as ex:            
            err_msg = f"{self.tag}: TFS session error: : {ex}"  #, exc_info=True
            log.error(err_msg)
            raise TFSSessionException(err_msg)
        return prediction