'''
Inference backend: tensorflow/servinf gRPC
'''


import sys
import logging
import asyncio
import numpy as np

from grpc import aio
from time import time, sleep
from config import PREPROCESSING_DEVICE, TFS_WARMUP_STEPS, TFS_WARMUP, TFS_WARMUP_DELAY
from engine.inference_backend import InferenceBackend
from engine.tfs_exception import TFSSessionException, TFSgRPCException

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


log = logging.getLogger('models')

from tf_load import tf


from engine.dtype import dtype_map, tf2np_dtype_map

# request timeout in seconds
GRPC_REQUEST_TIMEOUT = 2.0

# msg size limits   
MAX_SEND_MESSAGE_LENGTH = 0
MAX_RECV_MESSAGE_LENGTH = 20000000  # calculate max output tensor size in bytes

class TFServingGRPCBackend(InferenceBackend):
    '''TFServing gRPC model.'''
    def __init__(
                    self,  
                    model_cfg: dict,                # model profile
                    inference_lock,                 # lock or semaphore for async inference
                    supports_batch:bool=False       # optional batched inference
    ):
        super().__init__(model_cfg, inference_lock, supports_batch) 


        # gRPC signature parameters         
        self._signature_name = 'serving_default'
        self._addr = f"{self._host}:{self._port}"
        self._input_name = self.profile["export_targets"]["saved_model"]["options"]["inference_options"]["inputs"][0]["name"] #'input_tensor' 

        self.tag = f"{self._handler_name} ({self._addr})"

        # gathering outputs info
        self._outputs =  [output["name"] for output in self.profile["export_targets"]["saved_model"]["options"]["inference_options"]["outputs"]]

        # checking acces to tensorflow serving server       
        test_tensor = np.zeros(shape=self._input_shape[1:], dtype=tf2np_dtype_map[self._input_dtype])
        log.info(f'{self.tag}: Checking access to tensorflow-serving gRPC model: ')

        async def prediction_task(test_tensor):
            task = asyncio.create_task(self.predict([test_tensor]))
            return await task
        
        run_async = True
        try:
            running_loop = asyncio.get_running_loop()
            run_async = True
        except RuntimeError:
            run_async = False
        
        # check if we are in runing event loop
        try:
            if run_async: 
                test_prediction = asyncio.run_coroutine_threadsafe(
                    prediction_task(test_tensor), 
                    running_loop)   
            # execute in new event loop         
            else:
                test_prediction = asyncio.run(prediction_task(test_tensor))  
            # test_prediction = asyncio.run(prediction_task(test_tensor))  
            log_msg = f"{self.tag} Success. Ready for inference."
            log.info(log_msg)   

        except TFSgRPCException as connection_exception:
            log_msg = f"{self.tag} Initialization failure! Please check if tensorflow/serving is runing or check model profile!\n\t error: {connection_exception}"
            log.critical(log_msg)
            raise Exception(log_msg)

    def preprocess(self, batch:list):
        # just resize for tensorflow/serving
        return super().preprocess(batch)


    def postprocess(self, batch:list):
        # processed = np.squeeze(batch)       
        return batch#processed
    

    async def _predict(self, tensor):
        '''
        task for async single prediction
        '''

              
        result = dict()

        # convert tf.Tensor to numpy ndarray
        if isinstance(tensor, tf.Tensor):
            tensor = tensor.numpy()


        MAX_SEND_MESSAGE_LENGTH = sys.getsizeof(tensor)
                
        log_msg = f"{self.tag}: running tensorflow/serving gRPC async prediction: "      
        log.debug(log_msg)  
        
        try:
            start_time = time()
            async with self._lock:
                async with aio.insecure_channel(
                    self._addr,
                    options=[
                        ("grpc.max_send_message_length", MAX_SEND_MESSAGE_LENGTH),
                        ("grpc.max_receive_message_length", MAX_RECV_MESSAGE_LENGTH),
                    ]
                    ) as channel:
                
                    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                    request = predict_pb2.PredictRequest()
                    request.model_spec.name = self._path
                    request.model_spec.signature_name = self._signature_name
                    request.inputs[self._input_name].CopyFrom(
                        tf.make_tensor_proto(tensor, shape=tensor.shape)
                        )        
                                        
                    response = await stub.Predict(request)               
            
            end_time = time()
            inference_time = end_time - start_time
            
            if response is not None:
                log_msg = f"{self.tag} gRPC prediction:  Success!\n\t inference time: {inference_time:.2f}"
                log.debug(log_msg)                             
            else:
                log_msg = f"{self.tag} gRPC Prediction failure."
                log.debug(log_msg)
                raise TFSgRPCException(log_msg)
            
            for output_name in self._outputs:
                output_tensor_proto = response.outputs[output_name]
                with tf.device(PREPROCESSING_DEVICE):
                    output_shape = tf.TensorShape(output_tensor_proto.tensor_shape)
                    output_result = tf.reshape(output_tensor_proto.float_val, output_shape)
                result[output_name] = output_result.numpy().squeeze(axis=0)#[0] # squeezing
            
            log.debug(f"{self.tag}\n\t response:\n\t {result}")
                
            
        except (TFSSessionException, TFSgRPCException, Exception) as ex:
            msg = f'Please check gRPC model profile or tensorflow/serving connection!  : {ex}'
            log.error(msg)
            raise TFSgRPCException(msg)

        return result
    