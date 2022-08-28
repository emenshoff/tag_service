import json
import logging

from .config import SUPPORTED_IMAGE_FORMATS, ENGINE_VERSION, SUPPORTED_PROFILES

from .modules.pricedetection.profile import PriceDetectionProfileLoader

from .model_manager import ModelManager

from engine.engine_exception import EngineExceptionCritical, EngineExceptionError

log = logging.getLogger('engine')


class Engine:
    '''
    Engine, can be extended for future tasks.
    Uses config and profiles for different clients and their specific models / tasks
    '''
    def __init__(self, config_path):        
        try:
            with open(config_path) as f:
                self._config = json.load(f)
            log.debug(f"Loading config: {self._config}")
            self._profles = dict()
            
            version = self._config["version"]

            if version != ENGINE_VERSION:
                err_msg  = f"Engine config load failure, wrong engine version in config {config_path} : {version}, supported version: {ENGINE_VERSION}"
                raise EngineExceptionCritical(err_msg)

            # self._model_pool = {model_profile["name"]: model_profile for model_profile in self._config["models"]}  # now configs only. might be refactored to use TF Serving
            self._model_pool = ModelManager(self._config["models"])  # model pool / manager for inference
            self._name = self._config["name"]
            self._version =  version

            for profile in self._config["profiles"]:
                self._profles[profile["id"]] = self._load_profile(profile, self._model_pool)

        except Exception as ex: 
            err_msg = f"Engine config load failure, exception: {ex}"            
            raise EngineExceptionCritical(err_msg)


    def _load_profile(self, profile, model_pool): # might be extended for different tasks in the future
        if profile["type"] in SUPPORTED_PROFILES:  
            return PriceDetectionProfileLoader().load_profile(profile, model_pool)
        else:
            err_msg = f"Unsupported engine profile type: {profile['type']}!"
            raise EngineExceptionCritical(err_msg)

    @property
    def version(self):
        return self._version


    async def get_predictions(self, 
                        profile_id,             # profle name
                        content_type,           # type of content
                        content_prefix,         # prefox for content items
                        content,                # content: ndarray
                        attach_original_image,     # need to pack source image into the reply
                        pack_items_data):       # need to pack detected  items into the reply
        
        try:
            profile = self._profles[profile_id]
            result = await profile.get_predictions_async(
            # result = profile.get_predictions(
                                        content_type, 
                                        content_prefix,
                                        content, 
                                        attach_original_image, 
                                        pack_items_data
                                    )
            return result

        except KeyError as ex:
            err_msg = f"Profile not found: {profile_id} {ex}"
            raise EngineExceptionError(err_msg)

        except Exception as ex:
            err_msg = f"{ex}"
            raise EngineExceptionError(err_msg)
