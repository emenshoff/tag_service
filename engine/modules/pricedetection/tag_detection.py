import os
import sys
import numpy as np


from .detection import Detector

import logging

log = logging.getLogger('models')

from tf_load import tf


class TagDetector(Detector):
    '''
    detects price tags and promo tags on photo
    Single tensor inference
    '''
    def __init__(self, model):
        super().__init__(model)        

        try:
            self._promo_overlap_threshold = self._model.profile["options"]["promo_overlap_threshold"]  # overlap space % between promo tag and a price tag
        except KeyError as ex:
            err_msg = f"Wrong detector {self._name} config, exception: {ex}"
            log.error(err_msg)
            raise Exception(err_msg)