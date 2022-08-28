'''
Digit extraction from image using DL Object Detection approach
Detects only horisontal alighned values, angle up to 45 degrees is ok.
'''


import sys, os
import logging
from functools import reduce

# #import cv2
import numpy as np


from .detection import Detector


log = logging.getLogger('models')

from tf_load import tf
from utils.io import resize_image
from config import PREPROCESSING_DEVICE


filter_by_space_classes = {
        "BasePrice", 
        "PromoPrice", 
        "CardPrice", 
        "OldPrice", 
        "HSPrice", 
        "BulkPrice", 
        "iBasePrice",  "fBasePrice",
        "iPromoPrice", "fPromoPrice",
        "iCardPrice", "fCardPrice",
        "iOldPrice", "fOldPrice",
        "iHSPrice", "fHSPrice",
        "iBulkPrice", "fBulkPrice"

        }


class TagSemanticDetector(Detector):
    ''' 
    Extracts items from tag. This is semantic segmentation alternative.
    Single tensor inference 
    '''
    def __init__(self, model):
        super().__init__(model)
        '''
        
        '''
        

    def __apply_semantic_filters(self, img_tensor, detected_items):
        ''' 
        Additional semantic filters (Костыль)        
        '''                

        # deleting detected objects wich space is less than average in the semantic group

        # calculate average bbox space for each group        
        temp_dict = dict()
        result = dict()
        

        # calculating bbox space values
        for group_name, class_items in detected_items.items():
            if not group_name in filter_by_space_classes:
                result[group_name] = class_items

            temp_dict[group_name] = {
                                        "avr_group_space": 0,
                                        "items": []
                                    }
            avr_group_space = 0

            for item in class_items:
                class_name, bbox, score, img = item
                item_bbox_space = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                avr_group_space += item_bbox_space
                temp_dict[group_name]["items"].append((item_bbox_space, *item))

            group_size = len(class_items)
            if group_size > 0:
                avr_group_space /= group_size
                temp_dict[group_name]["avr_group_space"] = avr_group_space
            
        # drop items that has space less than average in the group
        for group_name in temp_dict:
            result[group_name] = []
            avr_group_space = temp_dict[group_name]["avr_group_space"]
            for item in temp_dict[group_name]["items"]:
                item_bbox_space, class_name, bbox, score, img = item
                if item_bbox_space >= avr_group_space:
                    result[group_name].append((
                                                    class_name, 
                                                    bbox, 
                                                    score,
                                                    img
                                            ))

        return result
  

    async def process(self, img_array_np, attach_items=False):
        '''
        Gets: array of image tensors (numpy or tf.tensor)
        Returns: array[extracted semantic items for each image]
        '''
        result = []
        predictions = await self.predict(img_array_np)        

        for prediction, img_tensor in zip(predictions, img_array_np):
            detected_items = self._extract_items(img_tensor, prediction, attach_items=True)
            filtered = self.__apply_semantic_filters(img_tensor, detected_items)
            result.append(filtered)

        return result