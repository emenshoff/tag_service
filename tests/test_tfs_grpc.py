import os
import sys
import json
import grpc

import logging

import numpy as np
import unittest

from time import time
from engine.dtype import dtype_map
from utils.io import read_img, encode_jpeg

from engine.tfs_exception import TFSSessionException, TFSgRPCException
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tests.config import TEST_TFS_GRPC_PORT, TEST_TFS_RESTAPI_PORT, TEST_TFS_HOST, TEST_MODELS_DIR_PATH, TEST_LOG_LEVEL

from tf_load import tf


log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)


MODELS_TO_TEST = [    
    {"name": "mobile_tag_detector", "version": 2},
    {"name": "mobile_tag_classifier",  "version": 1},
    {"name": "server_digit_detector",  "version": 1},
    {"name": "mobile_tag_semantic_detector",   "version": 2}
]


tf2np_dtype_map = {
    tf.uint8: np.uint8,
    tf.float32: np.float32
}


class TestTFS_GRPC(unittest.TestCase):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

        
    def test_prediction(self):
        passed = True       
        for model_item in  MODELS_TO_TEST:
            try:
                self.run_model_prediction(model_item)      
            except (TFSgRPCException, Exception) as ex:
                
                passed = False

        self.assertEqual(passed, True)
        

    def run_model_prediction(self, model_item):
        name = model_item["name"]
        version = model_item["version"]
        PROFILE_PATH = os.path.join(TEST_MODELS_DIR_PATH, f"{name}/{version}/profile.json")
        with open(PROFILE_PATH) as file:
            profile = json.load(file)

        addr = f"{TEST_TFS_HOST}:{TEST_TFS_GRPC_PORT}"
        input_options = profile["export_targets"]["saved_model"]["options"]["inference_options"]["inputs"][0]
        dtype = tf2np_dtype_map[dtype_map[input_options["dtype"]]]
        height =  input_options["shape"]["height"]
        width = input_options["shape"]["width"]
        channels = input_options["shape"]["channels"]
        batch_size = input_options["shape"]["batch_size"]      
        shape = (batch_size, height, width, channels)

        outputs =  [output["name"] for output in profile["export_targets"]["saved_model"]["options"]["inference_options"]["outputs"]]
        # print(outputs)

        test_tensor = np.zeros(shape=shape, dtype=dtype)    
        log_msg = f'Checking gRPC access to tensorflow-serving model {name} ({addr}): ' 

        # testing connection and prediction

        MAX_SEND_MESSAGE_LENGTH = sys.getsizeof(test_tensor)
        MAX_RECV_MESSAGE_LENGTH = 20000000

        # request timeout in seconds
        GRPC_REQUEST_TIMEOUT = 2.0

        signature_name = 'serving_default'
        input_name = input_options["name"]
        
        result = dict()       
   
        
        '''
        run predictions using TF Serving gRPC
        '''
        try:
            start_time = time()
            channel = grpc.insecure_channel(
                addr,
                options=[
                    ("grpc.max_send_message_length", MAX_SEND_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_RECV_MESSAGE_LENGTH),
                ]
            )
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = name
            request.model_spec.signature_name = signature_name
            request.inputs[input_name].CopyFrom(
            tf.make_tensor_proto(test_tensor, shape=test_tensor.shape))        
            response = stub.Predict(request, GRPC_REQUEST_TIMEOUT)
            
            end_time = time()
            
            if response is None:
                log_msg += f"Failure. Please check gRPC model profile \ remote model signature or its version..."
                log.error(log_msg)
                raise TFSgRPCException(log_msg)
            else:
                for output_name in outputs:
                    output_tensor_proto = response.outputs[output_name]
                    output_shape = tf.TensorShape(output_tensor_proto.tensor_shape)
                    # print(output_name, output_shape, output_tensor_proto.dtype)
                    output_result = tf.reshape(output_tensor_proto.float_val, output_shape)
                    result[output_name] = output_result.numpy()[0]
                log_msg += f"Ok\n\t inference time: {(end_time - start_time):.2f}\n\t prediction: {result}"
                log.debug(log_msg)
            
        except (TFSSessionException, Exception) as ex:
            msg = f'Please check gRPC model profile \ remote model signature or its version : {ex}'
            log.error(msg)
            raise TFSgRPCException(msg)
        return result

