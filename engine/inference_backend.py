'''
Inference model base class
'''

import os
import json
import logging
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

log = logging.getLogger('models')

from tf_load import tf
from engine.dtype import dtype_map
from engine.engine_exception import EngineExceptionError

from config import PREPROCESSING_DEVICE, MODELS_DIR_PATH
from engine.engine_exception import EngineExceptionCritical, EngineExceptionError



class InferenceBackend:
    '''
    Base model for inference pipeline
    Supports single and batched inference
    '''
    __metaclass__ = ABCMeta

    def __init__(
                    self, 
                    model_cfg: dict,             # model config entry in engine profile
                    inference_lock,              # lock or semaphore for async inference
                    supports_batch:bool=False    # optional batched inference
                ):

        self._supports_batch = supports_batch
        try:

            model_path_name = model_cfg["path"]  
            model_path_version = model_cfg["version"]
            self._host = model_cfg['host']
            self._port = model_cfg['port']
            self._path = model_path_name
            self._handler_name = model_cfg["name"]

            # automaticly takes latest version
            if model_path_version in [-1,0]:  
                correct_named_dirs = []
                v_dirs = os.listdir(model_path_name)

                for version_dir in v_dirs:
                    try:
                        correct_named_dirs.append( int(version_dir) )
                    except Exception:
                        pass

                if len(model_path_version) !=0:
                    model_path_version = max(correct_named_dirs)
                else:
                    err_msg = f"Can't find correct model structure for {model_path_name}! Model folder should include version subfolder (1,2,3 e.t.c)!" 
                    log.error(err_msg)
                    raise EngineExceptionCritical(err_msg)
            else:
                model_folder_path = os.path.join(MODELS_DIR_PATH, f"{model_path_name}/{model_path_version}/")
       
            # loading model profile for inference            
            with open(os.path.join(model_folder_path, "profile.json")) as profile_file:
                profile = json.load(profile_file)

            self.profile = profile         # helpful to keep it for debug
            self._name = profile["name"]
            self._version = profile["version"]                    
            self.tag = self._handler_name

            # model input parameters:
            model_input_options = profile["export_targets"]["saved_model"]["options"]["inference_options"]["inputs"][0]
            self._input_dtype = dtype_map[model_input_options["dtype"]]
            img_width = model_input_options["shape"]["width"]
            img_height = model_input_options["shape"]["height"]
            img_channels = model_input_options["shape"]["channels"]            
            batch_size = model_input_options["shape"]["batch_size"]

            # classification option: {class_id: class_name}
            self._classes = self.profile["metadata"]["classes"]                                                    

            self._model = None
            self._lock = inference_lock
            
            if not supports_batch:
                batch_size = 1
            self._input_img_size = (img_height, img_width)
            self._input_shape = (batch_size, img_height, img_width, img_channels)  
            log.info(f"{model_folder_path} is using '{self._name}' under the hood")         

        except (KeyError, Exception) as ex:   
            msg = f'Failure to initialize! Error : {ex}. Please check model profile!'
            log.critical(msg)
            raise EngineExceptionCritical(msg)                


    @property
    def version(self):
        return self._version

    @property
    def name(self):
        return self._name


    def preprocess(self, batch:list):
        '''        
        preprocessing before feeding
        OVVERRIDE IN CHILD CLASS !
        '''
        # always resize before model feeding!
        # @tf.function
        def preprocessing_fn(x):
            with tf.device(PREPROCESSING_DEVICE):
                x = tf.image.resize(x, self._input_img_size)
                x = tf.cast(x, self._input_dtype)
                x = x[np.newaxis, ...]
            return x.numpy()

        processed_batch = map(preprocessing_fn, batch)        
            
        return processed_batch


    def postprocess(self, batch:list):
        '''
        postprocessing for outputs after inference
        OVVERRIDE IN CHILD CLASS !
        '''
        
        def postprocession_fn(x):
          
            # convert tf.Tensor to numpy ndarray
            if isinstance(x, tf.Tensor):
                with tf.device(PREPROCESSING_DEVICE):
                    x = x.numpy()
            return x
        
        processed_batch = map(postprocession_fn, batch)    

        return processed_batch


    async def predict(self, batch:list):
        '''
        can be run in the event loop
        '''

        # checking batch size
        if len(batch) > self._input_shape[0]:
            err_msg = f"{self.tag}, provided batch size ({len(batch)}) is larger than model batch size ({self._input_shape[0]})"
            raise EngineExceptionError(err_msg)

        model_input = self.preprocess(batch)        
            #f"{self._name}_v{self._version}"   
        if self._supports_batch:         
            # not tested!
            batch_tensor = np.concatenate(model_input, axis=0)                
            prediction_raw = await self._predict(batch_tensor)
            prediction_raw = np.split(prediction_raw, len(batch), axis=0)
            prediction = self.postprocess(prediction_raw)

        else:                
            prediction_raw = [await self._predict(input_tensor) for input_tensor in model_input]
          
            
        
        prediction = self.postprocess(prediction_raw)
               
        return prediction


    @abstractmethod
    async def _predict(self, tensor):
        '''
        Prediction for asyncronous inference
        OVVERRIDE IN CHILD CLASS !
        '''
        raise NotImplemented
