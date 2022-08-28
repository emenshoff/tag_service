
import math
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

import logging

log = logging.getLogger('models')

from tf_load import tf
from engine.inference_backend import InferenceBackend


class InferenceModel:
    '''
    Base model for inference pipeline
    Supports single and batched inference
    '''
    __metaclass__ = ABCMeta

    def __init__(
                    self, 
                    model:InferenceBackend, # model backend
                ):
        
            # model packend
            self._model = model
            self._inference_batch_size = self._model._input_shape[0]
            
     
    @property
    def tag(self):
        return self.model.tag

    @property
    def img_size(self):
        return self._model._input_img_size

    
    def _preprocess(self, tensor):
        '''
         task-specific preprocessing
        '''
        return tensor

    
    def _postprocess(self, tensor):
        '''
         task-specific postpreprocessing
        '''       
        return tensor
    

   
    async def predict(self, img_np_list:list):  
        '''
        prediction for numpy array, with preprocessing
        supports any size of input list (cut batches to)
        '''
        predictions = []
        packet_len = len(img_np_list)

        iterations = math.ceil(packet_len / self._inference_batch_size)
        
        # batch should be fitted to model's input batch size
        for i in range(iterations):
            low_index = i * self._inference_batch_size
            up_index = low_index + self._inference_batch_size if (low_index + self._inference_batch_size) < packet_len else packet_len            
            batch = [self._preprocess(img_tensor) for img_tensor in img_np_list[low_index : up_index]]       

            # add dumb (empty) data to fit model batch size
            batch_real_len = len(batch)
            batch_need_len = self._model._input_shape[0]
            if batch_real_len < batch_need_len:
                dumb_data = [np.zeros(shape=self._model._input_shape[1:]) for i in range(batch_need_len-batch_real_len)]
                batch.extend(dumb_data)

            predicted_batch = await self._model.predict(batch) 

            # drop dumb data
            if batch_real_len < batch_need_len:
                predicted_batch = predicted_batch[:batch_real_len]

            postprocessed_batch = [self._postprocess(img_tensor) for img_tensor in  predicted_batch] 
            predictions.extend(postprocessed_batch)
        return predictions  


    # def process(self, 
    #             img_np_list,       # array of numpy images
    #             attach_items=False): # attach cropped images of detected items if True, attach coordinates (boxes [y:x, h:w]) if False
    #     '''
    #     processes the array of images (img_np_list), running predictions for each element + extra features (in child classes)
    #     Should be overrided in child classes!
    #     '''
    #     return self.predict(img_np_list)


    async def process(self, 
                img_np_list,       # array of numpy images
                attach_items=False): # attach cropped images of detected items if True, attach coordinates (boxes [y:x, h:w]) if False
        '''
        processes the array of images (img_np_list), running predictions for each element + extra features (in child classes)
        Should be overrided in child classes!
        '''
        return await self.predict(img_np_list)

