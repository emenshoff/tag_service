'''
Price detection API implementation
FastAPI version
'''
import os
import json
import fastapi
import logging

from engine.core import Engine
from config import ENGINE_CONFIG_PATH, API_VERSION, MAX_CONTENT_LENGTH
from api.protocol import API_Request, API_Response, ProtocolError


log = logging.getLogger('network')

# main WEB app
app = fastapi.FastAPI()
app.title = "Price tag recognition service"

log.info(f"****** Starting-up prediction worker. pid={os.getpid()} *******")

# main and the only engine 
engine = Engine(ENGINE_CONFIG_PATH)

supported_content = ["url", "file", "bin"]


@app.post("/api", response_model=API_Response)#, response_class=fastapi.responses.JSONResponse, tags=["API"]) #response_model=API_Response #tags=["API"]
async def index(request: API_Request):
        
    log.debug(f"Request= {request}")    
        
    try:
        # unpacking the request
        content = request.content
        profile = request.profile
        api_version = request.version
        content_type = request.content_type        
        get_item_image = request.get_item_image        
        content_prefix = request.content_prefix
        get_original_image = request.get_original_image

        # checking API version check
        if api_version != API_VERSION:
            error_message = f"Wrong API version! Supported: {API_VERSION}, API version in request: {api_version} "
            raise ProtocolError(error_message)
        
        # empty content check
        elif len(content) == 0:
            error_message = 'Empty "content" parameter!'
            raise ProtocolError(error_message)
        
        # max content length check        
        elif len(content) > MAX_CONTENT_LENGTH:
            error_message = f"Exceed maximum content size! Got {len(content)} items, protocol limit is {MAX_CONTENT_LENGTH} items."
            raise ProtocolError(error_message)

        elif profile == "":
            error_message =  f'"profile" parameter is not defined!'
            raise ProtocolError(error_message)

        else:
            try:
                # predicting
                predictions = await engine.get_predictions( 
                                                            profile,
                                                            content_type, 
                                                            content_prefix, 
                                                            content, 
                                                            get_original_image, 
                                                            get_item_image
                                                        )

                response = API_Response(status="success", content=predictions)       
                log.debug(f"Response: {response.json}")
                return response         
                
            except Exception as ex:
                log.error(f"Engine error: {ex}")
                status = "failure"
                error_message = f"Engine internal error! Please see application log for details..."
                response = API_Response(status=status, error_message=error_message, content=[])
                return response
            
    except Exception as ex:
        log.error(f'Protocol error: {ex}')#, exc_info=True)
        status = "failure"
        error_message = f"Protocol error! Exception: {ex}"
        response = API_Response(status=status, error_message=error_message, content=[])
        return response
    
    