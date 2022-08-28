'''
Base Object detection functionality. Provides basic threshold-based filtering and duplication supress
'''

from engine.model_proxy import InferenceModel

import logging
from engine.tfs_rest_backend import TFServingRESTAPIBackend

log = logging.getLogger('models')

from tf_load import tf
from utils.io import resize_image

from config import PREPROCESSING_DEVICE

from utils.detection_utils import denorm_box_coordinates, box_iou, restore_box_aspect_ratio


class Detector(InferenceModel):
    ''' 
    Base object detection functionality    
    Single tensor inference (4D, not batched)
    '''
    def __init__(
                    self, 
                    model, # model backend
        ):
        super().__init__(model)
        '''
        See detection config example.
        '''
        # all keys must be int type only.
        self._classes = {int(key): val for key, val in self._model._classes.items()}        

        try:          
            # getting thresholds from config
            self._threshold = self._model.profile["options"]["detection_threshold"]  #  detection threshold
            self._iou_threshold = self._model.profile["options"]["iou_threshold"]    #  threshold for duplicates drop
        except KeyError as ex:
            err_msg = f"{self.tag} Wrong detector config, exception: {ex}"
            log.error(err_msg)
            raise Exception(err_msg)


    def get_class_by_id(self, cls_id:int):
        ''' can be overrided in case of specific detection classes mapping '''
        return self._classes[cls_id]


    def _postprocess(self, prediction):
        '''
         task-specific postprocessing
        '''       
        result = dict()
        for key, value in prediction.items():
            # converting to numpy
            if isinstance(value, tf.Tensor):
                if len(value.shape) > 1:
                    value =  value.numpy()
                    # removing batch dimention
                    value = value.squeeze(axis=0)
                else:
                    value =  float(value.numpy())

           
            result[key] = value

        return result

    def ____filter_predictions(self, img_tensor, predictions):
        ''' 
        Base filtering \ preprocessing function for Object Detection:     
            0) Convert tf tensors to Numpy        
            1) Denormalize coordinates
            2) Removes duplicated characters by space overlap threshold / prediction score parameters 
        '''
        
        result = []
        delete_indexes = []     

        # if isinstance(self._model, TFServingRESTAPIBackend):
        #     predictions = predictions['predictions'] [0]

        # detection threshold filter
        for i, score in enumerate(predictions['detection_scores']):                  
            if score >= self._threshold: # take objects with score > threshold
                cls_id = int(predictions['detection_classes'][i])
                if cls_id in self._classes.keys():
                    box = predictions["detection_boxes"][i]
                    box = denorm_box_coordinates(self.img_size, box)
                    result.append((
                        cls_id,  # class id (int)
                        box,     # class score
                        score    # prediction score
                        ))
                        
                   
        # check for ObjectDetection duplicates. drop duplicate items with less prediction score
        for i, (_, box, score) in enumerate(result):
            j = (i + 1)
            if j < len(result):
                for (_, r_box, r_score) in result[j:]:                            
                    overlap =  box_iou(box, r_box) > self._iou_threshold
                    if overlap:
                        if score > r_score:                    
                            delete_indexes.append(j)
                        elif score < r_score:
                            delete_indexes.append(i)
                        else:
                            space = (box[3] - box[1]) * (box[2] - box[0])
                            r_space = (r_box[3] - r_box[1]) * (r_box[2] - r_box[0])
                            if space >= r_space:                    
                                delete_indexes.append(j)
                            else:
                                delete_indexes.append(i)
        # remove duplicates:
        cleared = list()
        for i, item in enumerate(result):
            if not i in delete_indexes:
                cleared.append(item)
        
        
        result = cleared # [(cls_id, box, score) for cls_id, box, score in cleared]
        return result

   
    def _extract_items(self, img_tensor, predictions, attach_items):      
        '''
        Extracts class number using ObjectDetection model,
        returns a dict {ClassName: image or bbox},
        drops unknown classes.
            img_tensor: original image tensor
            prediction: detection model prediction in TF Object Detection prediction signature
            attach_items: attach image to result if True or attach bounding box if false. The coordinates of bounding box are restored to original
        '''
        # cv2.imshow('img_tensor', img_tensor)
        # cv2.waitKey()
        detected_items = {}

        # remove duplicates and low-confidence predictions
        predictions = self.____filter_predictions(img_tensor, predictions) 

        for cls_id, box, score in predictions:            
            if cls_id in self._classes.keys():     
                cls_name = self.get_class_by_id(cls_id)
                
                if attach_items:
                    # get item subimage from model-compatible shape of orig img. REFACTOR
                    img = resize_image(img_tensor, self.img_size).numpy() 
                    y, x, h, w = box
                    item_image = img[  y : h, x : w  ]
                    item_result = (
                        cls_name, 
                        box,
                        score,
                        item_image) # image stored in the end

                    # cv2.imshow(f'opriginal image:', img)
                    # cv2.waitKey()
                    # cv2.imshow(f'semantic reading result {cls_name}:', item_image)
                    # cv2.waitKey()
                else:                    
                    orig_img_size = img_tensor.shape[:2]                    
                    restored_box = restore_box_aspect_ratio(box, self.img_size, orig_img_size)  # DEBUG
                    item_result = (
                        cls_name, 
                        restored_box, 
                        score)  
                
                if detected_items.get(cls_name, None) is None:
                    detected_items[cls_name] = []
                detected_items[cls_name].append(item_result)       
        
        return detected_items


    async def process(self, img_array_np, attach_items=False):
        '''
        Gets: [] of image tensors (numpy or tf.tensor)
        Returns: [extracted semantic items for each image]
        '''
        result = []
        predictions = await self.predict(img_array_np)        

        # for prediction, img_tensor in tqdm(zip(predictions, img_array_np), unit=' digit', desc=f'Detecting digit: model:{self._name}'):
        for prediction, img_tensor in zip(predictions, img_array_np):
            detected_items = self._extract_items(img_tensor, prediction, attach_items=attach_items)
            result.append(detected_items)

        return result