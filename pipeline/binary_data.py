import base64
import time

import logging
from io import BytesIO
from PIL import Image
from pipeline.data_generator import DataGenerator
from utils.io import resize_image, decode_jpg, decode_png, decode_img, decode_base64, zeros


log = logging.getLogger('data')


class BinaryDataGenerator(DataGenerator):
    '''
    Generates inference data directly from .jpeg / .png /.bmp / .gif - encoded binaries
    '''
    def __init__(self, content, prefix, batch_size, target_shape):
        super().__init__(content, prefix, batch_size, target_shape)

        self.prefix = self.prefix.lower() 
        

    def _prepare_data(self):
        
        t1 = time.time()

        decode_fn = decode_img # as default decoder        

        if self.prefix == "jpg" or self.prefix == "jpeg" or self.prefix == "" or self.prefix is None:            
            decode_fn = decode_jpg
            
        elif self.prefix == "png":
            decode_fn = decode_png
            
        elif self.prefix in ["gif", "bmp"]:
            decode_fn = decode_img
        
        for i, data in enumerate(self.content_meta):
            try:
                decoded_raw = base64.b64decode(data)   # used standard python decoder...
                # checking img format
                binary_format = Image.open(BytesIO(decoded_raw)).format.lower()
                if self.prefix != binary_format.lower():
                    if self.prefix == "jpg" and binary_format == "jpeg":
                        pass
                    else:            
                        raise Exception(f"Image format ({binary_format}) does not comply to the prefix provided ({self.prefix})")
                decoded_img = decode_fn(decoded_raw)
                # removing alpha channel for "bmp, png" or frame channel for gif
                if decoded_img.shape[2] > 3:
                    decoded_img = decoded_img[:,:,:3]

                if self.shape is not None:
                    decoded_img = resize_image(decoded_img, self.shape[:2]).numpy()
                self.data.append(decoded_img)

            except Exception as ex:
                err_msg = f"Error decoding file on index {i}, Exception: {ex}. Skipping"
                self.error_tape[i] = err_msg   
                log.error(err_msg)         
                dummy_data = None #zeros(self.shape).numpy()  # 
                self.data.append(dummy_data) # loading dummy data (empty image)

        t2 = time.time()
        processing_time = t2 - t1
        data_len = len(self.content_meta)
        log.debug(f"Decoding for {data_len} files completed. Processing time: {processing_time:.4f} sec. ({processing_time/data_len:.4f} sec per item).")
        self.ready = True
    