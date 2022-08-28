'''
Model Manager provides loading different types of backends \ models.
Configurable for different model types and backends.
Uses config loaded from profile.json in model's folder
'''

import logging

from asyncio import Semaphore
from .config import SUPPORTED_BACKENDS, MAX_GRPC_SESSIONS, MAX_LOCAL_SESSIONS, MAX_REST_SESSIONS

from engine.tf_local_backend import TFSavedModelBackend
from engine.tfs_grpc_backend import TFServingGRPCBackend
from engine.tfs_rest_backend import TFServingRESTAPIBackend


log = logging.getLogger('engine')


class ModelManager:       
    ''' Model loader / manager / pool'''
    def __init__(self, models_config):

        self._bk_grpc_sem = Semaphore(MAX_GRPC_SESSIONS)
        self._bk_rest_sem = Semaphore(MAX_REST_SESSIONS)
        self._bk_local_sem = Semaphore(MAX_LOCAL_SESSIONS)

        # self._model_version_check_interval = 0
        self._models = dict()
        for model_cfg in models_config:
            model_type = model_cfg["type"]
            if model_type in SUPPORTED_BACKENDS:
                if model_type == "TFSavedModelBackend":
                    model_name = model_cfg["name"]
                    self._models[model_name] = TFSavedModelBackend(model_cfg, self._bk_local_sem)                
                elif model_type == "TFServingRESTAPIBackend":
                    model_name = model_cfg["name"]
                    self._models[model_name] = TFServingRESTAPIBackend(model_cfg, self._bk_rest_sem) 
                elif model_type == "TFServingGRPCBackend":
                    model_name = model_cfg["name"]
                    self._models[model_name] = TFServingGRPCBackend(model_cfg, self._bk_grpc_sem)                    
            else:
                err_msg = f"Model type {model_cfg['type']} is not supported!"
                log.critical(err_msg)
                raise Exception(err_msg)
            

    def __getitem__(self, model_name):
        return self._models[model_name]


    def reload_model(self, model_name):
        pass



