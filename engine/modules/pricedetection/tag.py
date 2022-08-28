'''
Price tag detection / tag items recognition
'''
import os
import sys
import base64
import logging
import numpy as np

from tqdm import tqdm

from api.protocol import API_ResultItem, API_PriceTagPrediction

from .tag_detection import TagDetector
from .tag_classifier import TagClassifier
from .tag_semantic import TagSemanticDetector


from utils.io import encode_jpeg
from utils.detection_utils import denorm_box_coordinates

from tf_load import tf

from .item_extractor import TagItemExtractor
from .tag_items import tag_items


log = logging.getLogger('engine')


class TagReader:
    ''' Extracts price tag content: prices, bar codes, promo features e.t.c.  from prica tag image '''    
    def __init__(
                    self, 
                    config,     # module config
                    model_pool  # inference models manager
                ):
    
        try:
            self._config = config
            self._promo_tag_classes = config["promo_tag_classes"]
            self._promo_features = config["promo_features"]
            detector_name = config["detector"]   
            tag_classifier_name = config["tag_classifier"]
            extractor_config = config["extractor_config"] 
            
            
            self.tag_detector = TagDetector(model_pool[detector_name])                     # detects tags on the picture
            self.tag_classifier = TagClassifier(model_pool[tag_classifier_name])    # classifies tag brand to chose appropriate semantic model                        
            self.item_extractor = TagItemExtractor(extractor_config, model_pool)    # tag item \ semantic processor, converts tag item into a specific value of a specific data type
            self.default_semantic_model = None


            self.semantic_model_pool = {}
            for seg_model_name, brand_list in config["semantic_models"].items():
                
                sem_model = TagSemanticDetector(model_pool[seg_model_name])           # semantic detector. extracts tag items from tag image
                
                for brand_name in brand_list:
                    self.semantic_model_pool[brand_name] = sem_model
                    if brand_name == "Unknown":
                        self.default_semantic_model = sem_model

            # hardsetting default model for tag semantic
            if self.default_semantic_model == None:
                if model_pool.get("default_semantic_model") is None:
                    err_msg = f"default_semantic_model should be defined in config!"
                    log.critical(err_msg)
                    raise Exception(err_msg)
            else:                
                self.default_semantic_model = TagSemanticDetector(model_pool["default_semantic_model"])

            if len(self.semantic_model_pool) == 0:
                err_msg = f"Empty segmentation config"
                raise Exception(err_msg)

        except KeyError as ex:
            err_msg = f"Wrong PriceDetector config, exception: {ex}"
            log.error(err_msg)
            raise Exception(err_msg)

        except Exception as ex:
            err_msg = f"PriceDetector config load error: {ex}"
            log.error(err_msg)
            raise Exception(err_msg)
   

    def _check_tag_is_promo(self, promo_box, pricetag_box):       # REFACTOR: only denormalize coordinates! 
        ''' check if price tag fits promo tag (in bounds), coords are denormalized NEEDS TO BE TESTED! ADD IOU TEST'''
        
        overlap_space = 0
        
        promo_x, promo_y, promo_w, promo_h = promo_box
        tag_x, tag_y, tag_w, tag_h = pricetag_box

        left = max(promo_x, tag_x)
        top = max(promo_y, tag_y)
        right = min(promo_w, tag_w)
        bottom = min(promo_h, tag_h)

        width = right - left
        height = top - bottom

        if width < 0 or height < 0:
            overlap_space = 0
            return False
        else:
            overlap_space = width * height
        
        promo_space = (promo_w - promo_x) * (promo_h - promo_y)
        # log.debug(f'overlap_space {overlap_space}')
        # log.debug(f'promo_space {promo_space}')

        return (promo_space - overlap_space) >= promo_space * (self.tag_detector._promo_overlap_threshold)
        
   
    async def detect_async(self, img_list_np, attach_item_image=False):
        '''  takes array[numpy_array, None, numpy_array] Detects price tags on the photo and extracts items from one  '''

        result = []       

        # tag detecion stage        
        for img in img_list_np:
            
            wrong_img_format = False
            result_status = 'success'
            error_msg = ''
            item_result = None
            is_promo = False
            pricetag_result_img = None           
            
            # checking image format
            if img is None:                
                result.append( None )
                continue                
            else:
                try:
                    h = img.shape[0] 
                    w = img.shape[1]
                    c = img.shape[2]
                except Exception as ex:
                    error_msg = f"wrong image format! exception={ex}"
                    wrong_img_format = True                    

                if h <= 0 or w <= 0 or c != 3:
                    error_msg =  f"wrong image_shape! shape = {img.shape}"
                    wrong_img_format = True
                    
            # we do not process wrong image format
            if wrong_img_format:
                result.append(
                        API_ResultItem(                                                
                            result_status="failure", # Status: Success or failure
                            error_message=error_msg, # Error message if failure
                            prediction= []
                        )
                    )
                continue

            # *** STAGE 1: detecting tags on the image
            detected = await self.tag_detector.process([img], attach_items=True)
            detected = detected[0]

            detected_promo_count = 0
            pricetag_items = []

            # arranging pricetags
            for pricetag_class_name in detected:
                for item_value in detected[pricetag_class_name]:
                    cls_str_id, pricetag_img, score, img = item_value
                    if cls_str_id == "promotag":
                        detected_promo_count += 1
                    else:
                        # pricetag_items.append((pricetag_class_name, *item_value))
                        pricetag_items.append(item_value)
            
            detected_pricetags_count = len(pricetag_items)        
            
            # no pricetag found case
            if detected_pricetags_count == 0:
                result_status = 'failure'
                error_msg = 'No price tag found on the image'                
                               
            # more than one pricetag found
            elif detected_pricetags_count > 1:                    
                result_status = 'failure'
                error_msg = f"More than 1 price tag found on the image: {detected_pricetags_count}. Skipping recognition..."
                
            # only one price tag found, that is good!
            else:                         
                if detected_promo_count != 0:
                    is_promo = True              

                # take only one price tag
                pricetag_class_name, box, score, pricetag_img  = pricetag_items[0]
              
                # brand classifier
                pricetag_brand = await self.tag_classifier.process([pricetag_img])
                pricetag_brand = pricetag_brand[0]["name"]
                
                # *** STAGE 2 semantic extraction
                semantic_extractor = self.semantic_model_pool.get(pricetag_brand)
                if semantic_extractor is None:                
                    semantic_extractor = self.default_semantic_model
                
                segments = await semantic_extractor.process([pricetag_img], attach_items=True)
                segments = segments[0]
             
                tag_items = await self.item_extractor.process(pricetag_img, segments, attach_item_image)         

                # promo detection logic
                # promo feature by price tag class (color feature)
                if pricetag_class_name in self._promo_tag_classes:
                    is_promo = True

                # promo feature by price tag items 
                if is_promo == False:
                    for pricetag_class_name in dict(tag_items).keys():
                        if pricetag_class_name in self._promo_features:
                            if dict(tag_items)[pricetag_class_name] is not None:                                
                                    is_promo = True
                                    break

                if attach_item_image:
                    encoded_jpeg = encode_jpeg(pricetag_img)
                    encoded_jpeg = encoded_jpeg.numpy()
                    pricetag_result_img = base64.b64encode(encoded_jpeg).decode('utf-8') #

                item_result = API_PriceTagPrediction(
                                tagclass=pricetag_class_name,   # pricetag class
                                brand=pricetag_brand,           # price tag brand name
                                promo=is_promo,                 # Promo or not
                                pricetag_image=pricetag_result_img, # pricetag image in np array format                                                                                                   
                                items=tag_items                                  
                )
            result.append(
                API_ResultItem(              
                        result_status=result_status,    # Status: Success or failure
                        error_message=error_msg,        # Error message if failure
                        prediction=item_result          # Detection results
                )
            )
        return result
