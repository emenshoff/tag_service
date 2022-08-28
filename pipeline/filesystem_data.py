'''
Generates data by file path, reads from filesystem
'''

import logging
import time
import os

from pipeline.data_generator import DataGenerator
from utils.io import UnsupportedImageFormat, load_image, zeros


log = logging.getLogger('data')


class FileSystemDataGenerator(DataGenerator):
    '''
    Generates inference data from filesystem
    '''

    def __init__(self, content, prefix, batch_size, target_shape):
        super().__init__(content, prefix, batch_size, target_shape)
        
        if prefix :
            self.content_meta = list(map(lambda x: os.path.join(prefix, x), self.content_meta))  # file paths


    def _prepare_data(self):
        t1 = time.time()
        for i, fpath in enumerate(self.content_meta):
            try:
                
                if self.shape is not None:
                    img_size = self.shape[:2]                    
                else:
                    img_size = None
                    
                decoded = load_image(fpath, img_size).numpy()
                
                # removing alpha channel for "bmp, png" or frame channel for gif
                if decoded.shape[2] > 3:
                    decoded = decoded[:,:,:3]

                self.data.append(decoded)

            except UnsupportedImageFormat as ex:
                # 
                err_msg = f"Image decoding error: {fpath} : {ex}. Skipping..."
                self.error_tape[i] = err_msg   
                log.error(err_msg)         
                dummy_data = None # zeros(self.shape).numpy() # 
                self.data.append(dummy_data) # loading dummy data (empty image)
         
            except Exception as ex:
                err_msg = f"Error loading image: {ex}. Skipping..."  
                self.error_tape[i] = err_msg   
                log.error(err_msg)         
                dummy_data = None #zeros(self.shape).numpy() # 
                self.data.append(dummy_data) # loading dummy data (empty image)              

        t2 = time.time()
        processing_time = t2 - t1
        data_len = len(self.content_meta)
        log.debug(f"Reading for {data_len} files completed. Processing time: {processing_time:.4f} sec. ({processing_time/data_len:.4f} sec per item).")

        self.ready = True
        
