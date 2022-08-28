'''
Testing input data generators for inference
'''

import os
import unittest
import random
import base64
import logging
import asyncio

#import cv2

import numpy as np

from pipeline.data_generator import DataGenerator
from pipeline.http_data import HttpDataGenerator
from pipeline.binary_data import BinaryDataGenerator
from pipeline.filesystem_data import FileSystemDataGenerator

from utils.io import encode_jpeg, decode_jpg, read_img
from tests.config import TEST_MEDIA_PATH, TEST_LOG_LEVEL
from tests.config import ENGINE_INPUT_SHAPE

log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)


target_shape = ENGINE_INPUT_SHAPE


async def fetch_data(async_gen : DataGenerator):
    data = []
    async for batch in async_gen.get_data():
        data.extend(batch)

    return batch


class TestHttpGenerator(unittest.TestCase):
    def test_create(self):
        
        success = True
        batch_size = 100
        prefix = 'http://pricetag.test/'
        url1 = '000644ac7f58693e95c3ce080b095964.jpg'
        url2 = '0005864d56cf6dcf4c9956b01e22fa49.jpg'
        
        fname1 = '000644ac7f58693e95c3ce080b095964.jpg'
        fname2 = '0005864d56cf6dcf4c9956b01e22fa49.jpg' 

        try:
            http_gen = HttpDataGenerator([fname1, fname2],
                                         prefix=prefix, 
                                         batch_size=batch_size,
                                         target_shape=target_shape)
        except Exception as ex:
            print(ex)
            success = False

       
        log.debug(http_gen.content_meta)
        self.assertEqual(success, True)
        self.assertEqual(http_gen.prefix, prefix)
        self.assertEqual(http_gen.content_meta[0], prefix + url1)
        self.assertNotEqual(http_gen.content_meta[1],  url2) # diferent prefix check
        


    def test_prepare_data(self):
        batch_size = 100
        url1 = 'http://pricetag.test/nofile.jpg'  # empty
        url2 = 'http://pricetag.test/0005864d56cf6dcf4c9956b01e22fa49.jpg'
        url3 = 'https://www.tensorflow.org/images/tf_logo_social.png'
        http_gen = HttpDataGenerator([url1, url2, url3],
                              prefix="", 
                              batch_size=batch_size,
                              target_shape=target_shape)
        

        _ = asyncio.run(fetch_data(http_gen))

        len_meta = len(http_gen.content_meta)
        len_data = len(http_gen.data)

        if len(http_gen.data) == 3:
            log.debug(http_gen.data[0])
            if http_gen.data[0] is not None:
                self.assertEqual(http_gen.data[0].shape, target_shape)                                    
                self.assertEqual((http_gen.data[0] == np.zeros(target_shape)).all(), True)
            log.debug(type(http_gen.data[0]))
            log.debug(http_gen.data)
            log.debug(http_gen.error_tape)
        # self.assertEqual(http_gen.data[0].shape, target_shape)
        self.assertEqual(len_meta, len_data)
        self.assertNotEqual((http_gen.data[1] == np.zeros(target_shape)).all(), True)

        
        for data in http_gen.data:
            if data is not None:
                log.debug(f"shape = {data.shape}")
                self.assertEqual(ENGINE_INPUT_SHAPE, data.shape)
        

    
    def test_fetch(self):
        batch_size = 1000
        url1 = 'http://pricetag.test/nofile.jpg'  # empty
        url2 = 'http://pricetag.test/0005864d56cf6dcf4c9956b01e22fa49.jpg'
        http_gen = HttpDataGenerator([url1, url2],
                              prefix="", 
                              batch_size=batch_size,
                              target_shape=target_shape)
        asyncio.run(fetch_data(http_gen))

        self.assertEqual(len(http_gen.data), 2)

        for _, data in enumerate(http_gen.data):
            if data is not None:
                self.assertEqual(ENGINE_INPUT_SHAPE, data.shape)

    
    
class TestFilesystemDataGenerator(unittest.TestCase):
    def test_create(self):
        success = True
        batch_size = 1000
        prefix = TEST_MEDIA_PATH

        fnames = os.listdir(prefix)
        fnames = fnames[:batch_size]
        fpaths = [prefix + fname for fname in fnames]
        try:
            gen = FileSystemDataGenerator(fnames, prefix=prefix, batch_size=batch_size, target_shape=target_shape)
        except Exception as ex:
            success = False
            print(ex)
        
        asyncio.run(fetch_data(gen))

        self.assertEqual(success, True)
        self.assertEqual(len(fnames), len(gen.data))
        self.assertEqual(fpaths, gen.content_meta)
      
        
        for i, data in enumerate(gen.data):
            if data is not None:
                self.assertEqual(target_shape, data.shape)
                break


class TestBinaryGenerator(unittest.TestCase):
    def test_create(self):
        success = True
        batch_size = 100
        samples_count = 1000
        media_path = TEST_MEDIA_PATH
        fnames = os.listdir(media_path)
        fpaths = [os.path.join(media_path, fname) for fname in fnames]
        sample  = random.sample(fpaths, samples_count)
        prefix = 'jpg'       
        fcontent = [base64.b64encode(encode_jpeg(read_img(fpath)).numpy()).decode('utf-8') for fpath in sample ]
        try:
            gen = BinaryDataGenerator(fcontent, prefix=prefix, batch_size=batch_size, target_shape=target_shape)
        except Exception as ex:
            success = False
            print(ex)

        try:
            asyncio.run(fetch_data(gen))
        
            img = gen.data[0]
            self.assertEqual(ENGINE_INPUT_SHAPE, gen.data[0].shape)
            # cv2.imshow(f'{gen.content_meta[0]}', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
        except Exception as ex:
            print(ex)
            success = False

        
        self.assertEqual(success, True)
        # self.assertEqual(len(fnames), len(gen.data))
        # self.assertEqual(fpaths, gen.content_meta)
        log.debug(gen.data[0])
        log.debug(type(gen.data[0]))



if __name__ == "__main__":
    unittest.main()