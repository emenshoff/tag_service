
from genericpath import exists
import os
import sys
import unittest
import random
import base64
import logging
import shutil

#import cv2

import numpy as np
from datasetutils.dataset import DatasetException


from datasetutils.tag_classifier_dataset import TagClassifierDataset
from datasetutils.dataset import Dataset

from utils.io import encode_jpeg, decode_jpg, read_img
from tests.config import SOURCE_DATASET_BASE_PATH, TEST_DATSET_DIR, TEST_LOG_LEVEL

log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)


# src_path = os.path.join(SOURCE_DATASET_BASE_PATH, "/tag_brands/classification/v1/source/")
src_path = "/media/datasets/tag_brands/classification/v1/source/"
path = TEST_DATSET_DIR
name = "classification_test_dataset"     
version = 1              
tgt_dtype = "uint8"      
tgt_shape = (640, 640, 3)                     
compression = 0   
clear_existing=True 
use_aug=False,        
task="classification"
label_format='csv'
dataset = None


class TestClassificationDataset(unittest.TestCase):
   

    def setUp(self):
        if not os.path.exists(TEST_DATSET_DIR):
            os.mkdir(TEST_DATSET_DIR)

        print(src_path)

    # def test_create_from_profile_not_exists(self):
    #     success = False
    #     # profile_path = os.path.join(TEST_DATSET_DIR, f"{name}/v1/profile.json")
    #     profile_path = os.path.join(TEST_DATSET_DIR, f"profile.json")

    #     try:
    #         global dataset
    #         dataset = Dataset.from_profile(profile_path)
    #         success = True
    #     except DatasetException as ex:
    #         success = False
    #         print(ex)
    #     self.assertEqual(success, True)


    # def test_create_from_profile_exists(self):
    #     success = False
    #     profile_path = os.path.join(TEST_DATSET_DIR, f"{name}/v1/profile.json")
    #     # profile_path = os.path.join(TEST_DATSET_DIR, f"profile.json")

    #     try:
    #         global dataset
    #         dataset = Dataset.from_profile(profile_path)
    #         success = True
    #     except DatasetException as ex:
    #         success = False
    #         print(ex)

    #     self.tearDown()
    #     self.assertEqual(success, True)


    # def test_constructor(self):        
    #     success = False       
    #     try:
    #         global dataset
    #         dataset = TagClassifierDataset(
    #             src_path,      
    #             path,          
    #             name,           
    #             version,              
    #             tgt_dtype,      
    #             tgt_shape,                     
    #             compression,   
    #             clear_existing, 
    #             use_aug,        
    #             task,
    #             label_format)

    #         success = True
    #     except DatasetException as ex:
    #         success = False
    #         print(ex)
  
    #     self.assertEqual(success, True)


    def test_build(self):        
        success = False
        try:
            global dataset
            dataset = TagClassifierDataset(
                src_path,      
                path,          
                name,           
                version,              
                tgt_dtype,      
                tgt_shape,                     
                compression,   
                False, 
                use_aug,        
                task,
                label_format)

            dataset.build(train_split=0.7,val_split=0.1,test_split=0.2)
            success = True
        except DatasetException as ex:
            success = False
            print(ex)
        
        self.assertEqual(success, True)

    
    def test_get_train(self):
        success = False
        try:
            test_data = dataset.get("train", 256, True, True, False)
            success = True
        except DatasetException as ex:
            success = False
            print(ex)
        self.assertEqual(success, True)


    # def test_get_val(self):
    #     self.assertEqual(True, True)


    # def test_get_test(self):
    #     self.assertEqual(True, True)


    def tearDown(self):
        if os.path.exists(TEST_DATSET_DIR):
            shutil.rmtree(TEST_DATSET_DIR, ignore_errors=True)