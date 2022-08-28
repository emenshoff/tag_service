import logging
from abc import ABCMeta, abstractmethod


log = logging.getLogger('engine')

class OCRBackend:
    __metaclass__ = ABCMeta

    def __init__(self, options) -> None:
        self._options = options

    @abstractmethod
    def _preproicess_image(image):
        ''' ovverride in child class '''
        return None


    @abstractmethod
    def image_to_text(self, image) -> str:
        ''' ovverride in child class '''
        return ""