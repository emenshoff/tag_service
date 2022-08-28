'''
Data generation and preparation for inference.
Gets data from some source in some format and store it in pre-defined shape tensors.
'''

import math
import asyncio

class DecodingException(Exception):
    pass


class DataFormatException(Exception):
    pass


class DataNotFoundException(Exception):
    pass


class DataGenerator():
    '''
    Generates data for inference
    
    '''
    def __init__(self, content, prefix, batch_size, target_shape=None):
        ''' 
         target_shape is optional, used for generating same shape data
        '''
        self.shape = target_shape    # data tensor shape, if not None, all images will be resized
        self.content_meta = content  # content metadata, got in client's request: file names 
        self.data = []               # data in numpy tensor format.
        self.prefix = prefix         # prefix: common files path/url or file format in case of binary data provided  
        self.batch_size  = batch_size if len(content) > batch_size else len(content) # generator output batch size
        self.ready = False           # ready flag   
        self.error_tape = dict()     # error "tape". all reading \ decoding errors for each element should be stored here 

    
    def _prepare_data(self):
        '''
        Initialization function. Must be called before get_data.
        Can be used for data pre-cashing.
        '''
        raise NotImplementedError
    

    async def get_data(self):

        self._prepare_data()

        iterations = math.ceil(len(self.data) / self.batch_size)         
        if self.ready:
            for i in range(iterations):
                data_len  = len(self.data)
                low_index = i * self.batch_size
                up_index = low_index + self.batch_size if (low_index + self.batch_size) < data_len else data_len
                yield self.data[ low_index : up_index ]
                await asyncio.sleep(0)
       
 