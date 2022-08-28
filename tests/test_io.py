import os
import sys
import json
import time
import unittest
import random
import base64
import logging

import numpy as np
import pandas as pd

from tests.config import TEST_LOG_LEVEL
#import cv2


from tf_load import tf
from utils.io import *

log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)

media_path = 'data/test_data/'

jpeg_name = 'love.jpg'
png_name = 'pig.png'
gif_name = 'cat.gif'
bmp_name = 'bmp.bmp'
tiff_name = 'mark.tiff'

class TestIO(unittest.TestCase):


    def test_jpg_decode(self):
        success = True
        img_name = jpeg_name
        img_path = os.path.join(media_path, img_name)
        try:
            img = read_jpg(img_path).numpy()
            # cv2.imshow('test_jpg_decode', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # print(img)
        except Exception as ex:
            log.error(ex)
            success = False
        
        self.assertEqual(success, True)


    def test_png_decode(self):
        success = True
        img_name = png_name
        img_path = os.path.join(media_path, img_name)
        try:
            img = read_png(img_path).numpy()
            # cv2.imshow('test_png_decode', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # print(img)
        except Exception as ex:
            log.error(ex)
            success = False
        
        self.assertEqual(success, True)


    def test_bmp_decode(self):
        success = True
        img_name = bmp_name
        img_path = os.path.join(media_path, img_name)
        try:
            img = read_img(img_path).numpy()
            # cv2.imshow('test_bmp_decode', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # print(img)
        except Exception as ex:
            log.error(ex)
            success = False        
        
        self.assertEqual(success, True)


    def test_gif_decode(self):
        success = True
        img_name = gif_name
        img_path = os.path.join(media_path, img_name)
        try:
            img = read_img(img_path).numpy()
            # cv2.imshow('test_gif_decode', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # print(img)
        except Exception as ex:
            log.error(ex)
            success = False        
        
        self.assertEqual(success, True)


    def test_tiff_decode(self):
        success = True
        img_name = tiff_name
        img_path = os.path.join(media_path, img_name)
        try:
            img = read_img(img_path).numpy()
            # cv2.imshow('test', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # print(img)
        except Exception as ex:
            # log.error(ex)
            success = False
                
        self.assertNotEqual(success, True)

    

    def test_base64_decode(self):
        success = True
        img_name = jpeg_name
        img_path = os.path.join(media_path, img_name)
        try:
            # diff = []
            img = read_img(img_path).numpy()
            tf_img_base64 = encode_base64(img.tobytes()).numpy()
            py_img_base64 = base64.b64encode(img)
            # py_img_base64 = base64.standard_b64encode(img)
            

            tf_img_decoded =  decode_base64(tf_img_base64).numpy()
            py_img_decoded = base64.b64decode(py_img_base64)     

            # cv2.imshow('tf_img_decoded', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # cv2.imshow('py_img_decoded', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            

            # tf_by_py_img_decoded = base64.b64decode(tf_img_base64) 
            # py_by_tf_img_decoded = decode_base64(py_img_base64).numpy()

            # for a,b in zip(img.tobytes(), py_img_decoded):
            #     if a != b:
            #         diff.append((str(a), str(b)))
            # print(diff)

        except Exception as ex:
            log.error(ex)
            success = False
                
        # self.assertEqual(len(diff), 0)
        self.assertEqual(success, True)
        # self.assertEqual(tf_img_base64, py_img_base64)  # ! never equal!!!
        self.assertEqual(tf_img_decoded, py_img_decoded)
        # self.assertEqual(tf_by_py_img_decoded, py_by_tf_img_decoded)

        # self.assertEqual(img, tf_img_decoded)
        # self.assertEqual(img, py_img_decoded)

   
    # def test_base64_encode(self):
    #     success = True
    #     img_name = 'love.jpg'
    #     img_path = os.path.join(media_path, img_name)
    #     try:
    #         diff = []
    #         img = read_img(img_path).numpy()
    #         tf_img_base64 = encode_base64(img.tobytes()).numpy()
    #         py_img_base64 = base64.b64encode(img.tobytes())
    #         for a,b in zip(tf_img_base64, py_img_base64):
    #             if a != b:
    #                 diff.append((str(a), str(b)))
    #         print(diff)
    #         # print(tf_img_base64)
    #         # print(py_img_base64)
    #         # cv2.imshow('test_base64_encode', cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
    #         # cv2.waitKey()
    #         # print(img)
    #     except Exception as ex:
    #         log.error(ex)
    #         success = False
                
    #     self.assertEqual(success, True)
    #     self.assertEqual(tf_img_base64.decode(), py_img_base64.decode())


 # def test_png_encode(self):
    #     pass


    # def test_bmp_encode(self):
    #     pass


    def test_load_image(self):
        
        success = True
        img_size = (800,1333)

        # bmp, jpg, png, gif read test:
        for img_name in [jpeg_name,  png_name,  gif_name,  bmp_name]:
            img_path = os.path.join(media_path, img_name)

            try:
                img = load_image(img_path, img_size).numpy()
                self.assertEqual(img.shape[:2], img_size)
                # cv2.imshow(img_name, cv2.cvtColor(img,  cv2.COLOR_RGB2BGR))
                # cv2.waitKey()
            except Exception as ex:
                log.error(f'Reading failed, exception: {ex}')
                success = False

        #  tiff read test:
        img_path = os.path.join(media_path, tiff_name)
        try:
            img = load_image(img_path, img_size)
        except UnsupportedImageFormat as ex:
            log.error(f'Error handled, exception: {ex}')
            success = True


        self.assertEqual(success, True)


if __name__ == "__main__":
    unittest.main()