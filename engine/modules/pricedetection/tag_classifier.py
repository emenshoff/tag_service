from config import PREPROCESSING_DEVICE
from engine.model_proxy import InferenceModel

import numpy as np
# #import cv2

import logging

log = logging.getLogger('models')

from tf_load import tf
from utils.io import resize_image
from engine.tfs_grpc_backend import TFServingGRPCBackend


 

class TagClassifier(InferenceModel):
    '''
    Classify brand of price tag.    
    Customizable classes (from json model profile)
    Batched inference
    '''
    def __init__(self, model):
        super().__init__(model)    

        self._classes = {int(key): val for key, val in self._model._classes.items()}    


    def _preprocess(self, tensor):

               
        with tf.device(PREPROCESSING_DEVICE):
            x = tf.Variable(tensor) 
            dtype = x.dtype#self._model._input_dtype
            if (dtype == tf.float32):
                x = tf.math.divide(x, 127.5)
                x = tf.math.subtract(x, 1)
            elif (dtype == tf.uint8):
                x = tf.math.divide(x, 255) 

        return x.numpy()


    async def process(self, img_array_np, attach_items=False):
        '''
        img_array_np : array of np tensors in rgb
        returns: array of predictions, index is the same as source img_array_np,  [dict = {class_name: [polygn1, polygon2...]} ]   
        '''                        
        result = []
        predictions = await self.predict(img_array_np)
        # for prediction, img_tensor in tqdm(zip(predictions, img_array_np), unit=' pricetag', desc=f'Segmentation: model:{self._name}'):
        for prediction, img_tensor in zip(predictions, img_array_np):
            # костыль
            if isinstance(self._model, TFServingGRPCBackend):
                prediction = prediction["dense"]
            predicted_cls_index = np.argmax(prediction)
            prediction_cls_name = self._classes.get(predicted_cls_index)
            if prediction_cls_name == None:
                    prediction_cls_name = 'Unknown'  # default brand that is not in brand list (may be new)
            result.append(
                {
                    "name": prediction_cls_name,
                    "class_index": predicted_cls_index
                }
            )

        return result      