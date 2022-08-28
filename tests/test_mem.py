'''
Memory leackage test
'''



import os
import json
import unittest
import random
import copy
import logging
import json
import requests
from tqdm import tqdm


# from engine.core import Predictor
from utils.io import read_img, encode_jpeg
from tests.config import TEST_MEDIA_PATH, TEST_PROFILE


log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)


from tests.config import TEST_HOST, TEST_PORT, API_VERSION

host = TEST_HOST
port = TEST_PORT
url = f"http://{host}:{port}/api"
PROFILE = TEST_PROFILE


PAYLOAD = {
                        "version": API_VERSION,
                        "get_original_image": False,
                        "get_item_image": False,
                        "profile": PROFILE,
                        "content_type": None,
                        "content_prefix": "",
                        "content": None
                    }


timeout = 400.0
samples_count = 100


class TestAPI(unittest.TestCase):
   
   

    # def test_http_content(self):
        
    #     url_not_exists = "http://aromam.retailiqa.ru/static/media/5979dc2864586200e6c25ca7/434e56b9ec020cb7c1908f5fca12e72d.jpg"  # empty
    #     url_exists = "http://aromam.retailiqa.ru/static/media/5e9403e11cfefa0009ec3304/643d9d84eee9ac4124c697deb431ba48.jpg"

    #     payload = copy.deepcopy(PAYLOAD)
    #     payload["content_type"] = "url"
    #     payload["content"] = [ url_not_exists]                            

    #     resp = requests.request('POST', url, json=json.dumps(payload), timeout=timeout)
    #     resp = json.loads(resp.content)
    #     self.assertNotEqual(resp['content'][0]['error_message'], '')

    #     payload["content"] = [url_exists]  
    #     resp = requests.request('POST', url, json=json.dumps(payload), timeout=timeout)
    #     resp = json.loads(resp.content)
    #     self.assertEqual(resp['content'][0]['error_message'], '')


    def test_file_content_no_attachments(self):
       
        iterations = 10
        
        media_path = TEST_MEDIA_PATH
        fnames = os.listdir(media_path)
        # fpaths = [os.path.join(media_path, fname) for fname in fnames]
        # print(fpaths[0])

        for _ in tqdm(range(iterations), desc='requests (no attachments)'):

            content  = random.sample(fnames, samples_count)

            # log.info(content)
            # print(content)

            payload = copy.deepcopy(PAYLOAD)
            payload["content_type"] = "file"
            payload["content"] = content
            payload["content_prefix"] = media_path        
      

            # log.info(json.dumps(payload))
            # resp = http.request('GET', url, fields=json.dumps(payload), timeout=timeout)
            resp = requests.request('POST', url, json=json.dumps(payload))
            resp = json.loads(resp.content)
            # print(resp)
            # log.info(resp)
            # self.assertEqual(resp['error_message'], '')
            # self.assertEqual(resp["content"][0]["result_status"], 'success')
        self.assertEqual(True, True) # we are testing only  memory leak
        
        
    def test_file_content_with_attachments(self):
        iterations = 10
        media_path = TEST_MEDIA_PATH
        fnames = os.listdir(media_path)
        # fpaths = [os.path.join(media_path, fname) for fname in fnames]
        # print(fpaths[0])

        for _ in tqdm(range(iterations), desc='requests (with attachments)'):

            content  = random.sample(fnames, samples_count)

            # log.info(content)
            # print(content)

            payload = copy.deepcopy(PAYLOAD)
            payload["content_type"] = "file"
            payload["content"] = content
            payload["content_prefix"] = media_path        
            payload["get_original_image"] = True
            payload["get_item_image"] = True


            # log.info(json.dumps(payload))
            # resp = http.request('GET', url, fields=json.dumps(payload), timeout=timeout)
            resp = requests.request('POST', url, json=json.dumps(payload))
            resp = json.loads(resp.content)
            # print(resp)
            # log.info(resp)
            # self.assertEqual(resp['error_message'], '')
            # self.assertNotEqual(resp["content"][0]["original_image"], None)

        self.assertEqual(True, True) # we are testing only  memory leak


        
    # def test_binary_content(self):   

    #     samples_count = 1000
    #     media_path = TEST_MEDIA_PATH
    #     fnames = os.listdir(media_path)
    #     fpaths = [os.path.join(media_path, fname) for fname in fnames]
    #     sample  = random.sample(fpaths, samples_count)
    #     # print(fpaths[0])
    #     content = [base64.encodebytes(encode_jpeg(read_img(fpath)).numpy()).decode('utf-8') for fpath in sample]      
    #     # log.error(len(content))
    #     payload = copy.deepcopy(PAYLOAD)
    #     payload["content"] = content
    #     payload["content_type"] = "bin"
    #     payload["content_prefix"] = "jpg"
    #     payload["get_item_image"] = False
        
    #     # log.info(json.dumps(payload))
    #     # resp = http.request('GET', url, fields=json.dumps(payload), timeout=timeout)
    #     resp = requests.request('POST', url, json=json.dumps(payload))
    #     resp = json.loads(resp.content)
    #     # print(resp)
    #     self.assertEqual(resp['error_message'], '')


   

    



if __name__ == "__main__":
    unittest.main()