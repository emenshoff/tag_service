class ProfileLoader:
    '''
    Kind of "adapter" to load  config for specific task
    '''
    def load_profile(self, config, model_pool):
        raise NotImplementedError


class EngineProfile:
    def __init__(self, 
                config,     # dict, engine config with different profiles
                model_pool  # dict, models config.
                ):
        self._version = "Undefined"


    @property
    def version(self):
        return self._version


    def get_predictions(self, 
                        content_type,               # content type: bin, file, url
                        content_prefix,             # content prefix. see API docs
                        content,                    # content from request
                        attach_original_image,      # attach original image to response 
                        attach_item_image):         # attach item images for each recognized object (see API docs)
        '''
        Run predictions in specific profile context
        '''
        raise NotImplementedError

    async def get_predictions_async(self, 
                        content_type, 
                        content_prefix,
                        content, 
                        attach_original_image, 
                        attach_item_image):
        '''
        Run predictions in specific profile context
        '''
        raise NotImplementedError


