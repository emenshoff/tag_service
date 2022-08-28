'''
Local model
Tensorflow backend
(saved_model, direct inference)
ONLY Syncronous inference!!!
'''

import os
import logging
import asyncio


from time import time
from engine.inference_backend import InferenceBackend
from engine.dtype import dtype_map
from tf_load import tf

from config import MODELS_DIR_PATH, INFERENCE_DEVICE

log = logging.getLogger('models')

DEFAULT_SEMAPHORE_VALUE = 1


# some hack for async
async def run_prediction(model, tensor):
    await asyncio.sleep(0)
    with tf.device(INFERENCE_DEVICE):
        result = model(tensor)
    return result


class TFSavedModelBackend(InferenceBackend):
    '''TensorFlow saved model adapter'''
    def __init__(
                    self, 
                    model_cfg: dict,             # model config entry in engine profile
                    inference_lock,              # lock or semaphore for async inference
                    supports_batch:bool=False    # optional batched inference
    ):
        """Load model from local filesystem"""
        super().__init__(model_cfg, inference_lock, supports_batch)        
        
        # loading TF saved_model graph with weights   
        saved_model_path = os.path.join(self.profile["path"], self.profile["export_targets"]["saved_model"]["path"])     
        saved_model_path = os.path.join(MODELS_DIR_PATH, saved_model_path)
        start_time = time()
        self._model = tf.saved_model.load(saved_model_path)
        end_time = time()
        load_time = end_time - start_time
        log.info(f'{self.tag} Model: {self._name};  (path: {saved_model_path}) has been loaded... (took {load_time:.2f} sec)')

   
            
    async def _predict(self, tensor):
        '''
        Prediction task for syncronous inference
        '''
        start_time = time()
        prediction_task = asyncio.create_task(run_prediction(self._model, tensor))
        async with self._lock:
            prediction = await prediction_task
               
        end_time = time()
        inference_time = end_time - start_time
        log.debug(f"{self.tag} Prediction:\n\t inference time: {inference_time:.2f}\n\t  {prediction}")

        return prediction

