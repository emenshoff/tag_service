
import base64
import logging
import asyncio

log = logging.getLogger('engine')

from engine.profile import EngineProfile, ProfileLoader
from engine.config import SUPPORTED_IMAGE_FORMATS
from config import MAX_CONTENT_LENGTH
from time import time
from .tag import TagReader

from utils.io import encode_jpeg

from pipeline.binary_data import BinaryDataGenerator
from pipeline.http_data import HttpDataGenerator
from pipeline.filesystem_data import FileSystemDataGenerator
from api.protocol import API_ResultItem


class PriceDetectionProfileLoader(ProfileLoader):
    '''
    "Price detection task profile/config loader
    '''
    def load_profile(self, profile, model_pool):
        return PriceDetectionProfile(profile, model_pool)


class PriceDetectionProfile(EngineProfile):
    '''
    Prediction unit. Combines several models and different data sources.
    '''
    def __init__(self, config, model_pool):
        super().__init__(config, model_pool)
        '''
        Takes dict config and loads parameters
        '''
        try:
            self._id = config["id"]           
            self._version = config["version"]           
            self._core = TagReader(config["core"], model_pool)
        except Exception as ex:
            err_msg = f"Failure to load Price predictor config, exception: {ex}"
            log.error(err_msg)
            raise Exception(err_msg)
        log.info(f"Profile loaded: {self._id}.")


    def _get_data_generator(self,
                            content_type, 
                            content_prefix,
                            content
                            ):
        if content_type == "url":
            content_generator = HttpDataGenerator(
                                                    content=content, 
                                                    prefix=content_prefix, 
                                                    batch_size=MAX_CONTENT_LENGTH, 
                                                    target_shape=self._core.tag_detector.img_size   
                                                )

        elif content_type == "file":
            content_generator = FileSystemDataGenerator(
                                                            content=content, 
                                                            prefix=content_prefix, 
                                                            batch_size=MAX_CONTENT_LENGTH, 
                                                            target_shape=self._core.tag_detector.img_size
                                                       )
            
        elif content_type == "bin":
            if content_prefix not in SUPPORTED_IMAGE_FORMATS:               
                error_message = f"Unsupported image format: {content_prefix}."
                raise Exception(error_message)
            else:
                content_generator = BinaryDataGenerator(
                                                            content=content, 
                                                            prefix=content_prefix, 
                                                            batch_size=MAX_CONTENT_LENGTH, 
                                                            target_shape=self._core.tag_detector.img_size
                                                        )

        else:
            error_message = f"Unsupported content type: {content_type}"
            log.error(error_message)
            raise Exception(error_message)
        
        return content_generator
        

    async def get_predictions_async(self, 
                        content_type, 
                        content_prefix,
                        content, 
                        attach_original_image, 
                        attach_item_image):
        '''
        Run predictions in specific profile context
        '''

        
        start_time = time()
        content_generator = self._get_data_generator(content_type, 
                                                    content_prefix,
                                                    content)      

        result = []                             # final result         
        tasks = []                              # prediction tasks
                     
        log.info(f"<Profile '{self._id}'>: Running async predictions: for {len(content)} images...")

        async for data_batch in content_generator.get_data():
            tasks.append(
                self._prediction_task(
                    content_generator,
                    data_batch,
                    attach_item_image,
                    attach_original_image
                    )
            )
        r = await asyncio.gather(*tasks)
        result.extend(*r)        
       
        packet_len = len(content_generator.data)
        del(content_generator) 
        end_time = time()
        processing_time = (end_time-start_time)
        log.info(f"<Profile '{self._id}'>: Predictions complete... Took {processing_time:.2f} seconds. ({(packet_len/processing_time)} img/sec)")
        
        return result


    async def _prediction_task(self, content_generator, data_batch:list, attach_item_image:bool, attach_original_image:bool):

        result = []
       
        predictions_batch_result = await self._core.detect_async(img_list_np=data_batch, 
                                                    attach_item_image=attach_item_image)

        for i, data_item in enumerate(data_batch):  # filling the headers
        
            original_image = None
            err_message = ""
            result_status = "success"
            prediction = None
            
            data_err = content_generator.error_tape.get(i, None)                    # check if data load was failed
            
            if data_err is None:                                                    # read was successful
                if attach_original_image:
                    encoded_jpeg = encode_jpeg(data_item)                           # attach original image
                    encoded_jpeg = encoded_jpeg.numpy()
                    original_image = base64.b64encode(encoded_jpeg).decode('utf-8') # adding original image to response
                
                # taking fields from prediction's result
                if predictions_batch_result[i] is not None:
                    err_message = predictions_batch_result[i].error_message
                    prediction = predictions_batch_result[i].prediction
                    result_status = predictions_batch_result[i].result_status

                else:
                    err_message = "empty image data, failure to predict"                        
                    result_status = "failure"
                                    
            else:
                err_message = data_err
                result_status = "failure"
                # log.error(data_err)

            result.append(
                            API_ResultItem(
                                original_image=original_image, 
                                result_status=result_status,
                                error_message=err_message,
                                prediction=prediction
                            )
            )
    
        
        return result