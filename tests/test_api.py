'''

'''

import os
import sys
import json
import base64
import time
import unittest
import random
import copy
import logging

import numpy as np
import pandas as pd
import json

# import urllib3
import requests


# from engine.core import Predictor
from utils.io import read_img, encode_jpeg
from tests.config import TEST_MEDIA_PATH, TEST_API_VERSION, TEST_PORT, TEST_HOST, TEST_PROFILE, ENGINE_INPUT_SHAPE, TEST_LOG_LEVEL


log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)

import config

host = TEST_HOST
port = TEST_PORT
url = f"http://{host}:{port}/api"

PAYLOAD = {
                        "version": TEST_API_VERSION,
                        "get_original_image": False,
                        "get_item_image": False,
                        "profile": TEST_PROFILE,
                        "content_type": None,
                        "content_prefix": "",
                        "content": None
                    }


timeout = 400.0


class TestAPI(unittest.TestCase):
   
    
    def test_api_version(self):
        # correct version test
        log.info("Running: test_api_version")        
        correct_version = config.API_VERSION
        incorrect_version = 65536.0
        payload = copy.deepcopy(PAYLOAD)
        payload["version"] = correct_version
        payload["content_type"] = "url"
        payload["content"] = [
                            "http://pricetag.test/0001f4bd78f89145cea4fc7dc32972f7.jpg",
                            "http://pricetag.test/00048fa056fec3570b536f3801c47c0c.jpg"
                        ]     
        resp = requests.request('POST', url, data=json.dumps(payload), timeout=timeout)
        log.debug(f'response raw: {resp}')
        resp = json.loads(resp.content)
        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        log.debug(f'response: {resp}')
        self.assertEqual(resp['error_message'], '')
        

        # incorrect version test
        payload["version"] = incorrect_version       
        log.debug(f'sending POST request, payload={json.dumps(payload)}') 
        resp = requests.request('POST', url, data=json.dumps(payload), timeout=timeout)
        log.debug(f'response raw: {resp}')
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp['error_message'], '')

        log.info("test_api_version: Passed")


    def test_http_content(self):
        # test data (better viryfy before test run)
        log.info("Running: test_http_content")   

        url_not_exists = "http://pricetag.test/434e56b9ec020cb7c1908f5fca12e72d.jpg"  # fake, not existing file
        url_exists = "http://pricetag.test/0001f4bd78f89145cea4fc7dc32972f7.jpg"      # existing file

        payload = copy.deepcopy(PAYLOAD)
        payload["content_type"] = "url"
        payload["content"] = [url_not_exists, url_exists]                           

        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload), timeout=timeout)
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp['content'][0]['error_message'], '')
        self.assertEqual(resp['content'][1]['error_message'], '')
        
        log.info("test_http_content: Passed")


    def test_file_content(self):
        
        log.info("Running: test_file_content")

        samples_count = 10
        media_path = TEST_MEDIA_PATH
        fnames = os.listdir(media_path)

        content  = random.sample(fnames, samples_count)

        # log.debug(content)

        payload = copy.deepcopy(PAYLOAD)
        payload["content_type"] = "file"
        payload["content"] = content
        payload["content_prefix"] = media_path        
        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertEqual(resp['error_message'], '')

        payload["get_original_image"] = True
        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp["content"][0]["original_image"], None)
        
        log.info("test_file_content: Passed")


    def test_binary_content(self):   

        log.info("Running: test_binary_content")
        samples_count = 10
        media_path = TEST_MEDIA_PATH
        fnames = os.listdir(media_path)
        fpaths = [os.path.join(media_path, fname) for fname in fnames]
        sample  = random.sample(fpaths, samples_count)
        content = [base64.encodebytes(encode_jpeg(read_img(fpath)).numpy()).decode('utf-8') for fpath in sample]      
        payload = copy.deepcopy(PAYLOAD)
        payload["content"] = content
        payload["content_type"] = "bin"
        payload["content_prefix"] = "jpg"
        payload["get_item_image"] = False
        
        log.debug(f'sending POST request, payload={json.dumps(payload)}')      
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertEqual(resp['error_message'], '')

        log.info("test_binary_content: Passed")
   

    def test_protocol(self):             

        log.info("Running: test_protocol")
        # empty content error handling
        content = []
        payload = copy.deepcopy(PAYLOAD)
        payload["content"] = content
        payload["content_type"] = "bin"
        payload["content_prefix"] = "jpg"       
        
        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp['error_message'], "")


        # test max content size
        content = ["file.jpg" for _ in range(config.MAX_CONTENT_LENGTH + 1)]
        payload = copy.deepcopy(PAYLOAD)
        payload["content"] = content
        payload["content_type"] = "bin"
        payload["content_prefix"] = "jpg"

        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp['error_message'],"")

        
        # test wrong content type
        content = ["duck.jpg"]
        content_type = "porn"
        payload = copy.deepcopy(PAYLOAD)
        payload["content"] = content
        payload["content_type"] = content_type
        payload["content_prefix"] = "jpg"

        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp['error_message'],"")
        
        # test empty payload        
        payload = None        
        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        error = resp.get('error_message')
        if error is not None:
            self.assertNotEqual(error,"")
        else:
            self.assertEqual(error, None)
        


        # image format for bin pack check
        content = ["file.jpg" ]
        content_type = "bin"
        content_prefix = "tiff"
        payload = copy.deepcopy(PAYLOAD)
        payload["content"] = content
        payload["content_type"] = content_type
        payload["content_prefix"] = content_prefix       

        log.debug(f'sending POST request, payload={json.dumps(payload)}')
        resp = requests.request('POST', url, data=json.dumps(payload))
        resp = json.loads(resp.content)
        log.debug(f'response: {resp}')

        self.assertNotEqual(resp['error_message'], "")


        log.info("test_protocol: Passed")
             



if __name__ == "__main__":
    unittest.main()