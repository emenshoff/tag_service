'''
Digit extraction from image using DL Object Detection approach
Detects only horisontal alighned values, angle up to 45 degrees is ok.
'''

from engine.model_proxy import InferenceModel

import sys, os
import logging
#import cv2
import numpy as np


from .detection import Detector

log = logging.getLogger('models')

from tf_load import tf


from config import PREPROCESSING_DEVICE


class DigitDetector(Detector):
    ''' 
    Detects digits in segmented images.
    Single tensor inference 
    '''
    def __init__(self, model):
        super().__init__(model)
        
        # hardcoded classes to avoid model wrong class map load from config. please check your model!
        self._classes = {
                            10: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            4: 4,
                            5: 5,
                            6: 6,
                            7: 7,
                            8: 8,
                            9: 9,
                            0: 10  # default, unlabbelled class                          
                        }       

   
    def _extract_digits(self, img_tensor, prediction):  
        '''
        Extracts class number using ObjectDetection model
        returns a list of digits in the order they detected
        drops "non-symbol" classes
        '''
        # cv2.imshow('img_tensor', img_tensor)
        # cv2.waitKey()
        detected = self._extract_items(img_tensor, prediction, attach_items=False)   
        result = []      
        for val in detected.values():
            result.extend(val)
        result.sort(key=lambda coord: coord[1][1], reverse=True) # x-axis sorting by real symbol order (right-to-left)
        return result


    # digit symbol list to value translation
    def _symbol_list_to_value(self, digit_list):
        value = 0.0
        for position, symbol in enumerate(digit_list):
            symbol = float(symbol[0])            
            value += symbol * pow(10, position)            
        return value


    async def process(self, img_array_np, attach_items=False):
        '''
        Gets array of image tensors (numpy)
        Returns array[recognized float value for each number's image]
        '''
        result = []
        predictions = await self.predict(img_array_np)        

        # for prediction, img_tensor in tqdm(zip(predictions, img_array_np), unit=' symbol', desc=f'Detecting symbol: model:{self._name}'):
        for prediction, img_tensor in zip(predictions, img_array_np):
            digits = self._extract_digits(img_tensor, prediction)                
            value = self._symbol_list_to_value(digits)
            # log.debug(f"recognized value: {value}")
            result.append(value)

        return result