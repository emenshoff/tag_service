'''
Generates data by url, from http sources
'''
import asyncio
import aiohttp
import math
import logging


from tf_load import tf
from pipeline.data_generator import DataGenerator
from utils.io import load_img_bin, load_jpg_bin


# max client sessions
MAX_SESSIONS = 30


log = logging.getLogger('data')


async def _fetch_file(url, sem, target_shape):
    '''
    File download task
    url: file url
    sem: lock for limiting parrallel http sessions
    buff: dict to write the results and errors
    target_shape: 
    '''
    log.debug(f'fetching {url}')
    if target_shape is not None:
        img_size = target_shape[:2]                    
    else:
        img_size = None

    err_msg = ''
    data = None

    try:
        async with sem:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        try:
                            content = await response.read()
                            log.debug(f'content {url} : {content}')
                            if content is not None:
                               
                                try:
                                    if tf.io.is_jpeg(content):                                       
                                        decoded = load_jpg_bin(content, img_size).numpy()                                             
                                    else:                                            
                                        decoded = load_img_bin(content, img_size).numpy()
                                    # removing alpha channel for "bmp, png" or frame channel for gif                                   
                                    if decoded.shape[2] > 3:
                                        decoded = decoded[:,:,:3]                                   

                                    data = decoded
                                    log.debug(f"{url}: downloaded.")  
                                    return data, err_msg                                  

                                except  Exception as ex:
                                    err_msg = f"Can't decode file data! Exception = {ex}"
                                    raise Exception(err_msg)                                
                            else:
                                err_msg = f"Empty content in response"
                                raise Exception(err_msg)
                        except Exception as ex:
                            err_msg = f"Read error: {ex}"
                            raise Exception(err_msg)
                    else:
                        err_msg = f"HTTP error, http response code: {response.status}"
                        raise Exception(err_msg)                            
    
    except Exception as ex:
        err_msg = f"Error to download file: {url}, error: {ex}"  #, exc_info=True
        log.error(err_msg)
        data =  None
        return data, err_msg
    

class HttpDataGenerator(DataGenerator):
    '''
    Generates  inference data from urls:
    connects to url, fetches the data, stores it on the disk and generates into binary
    '''
    def __init__(self, content, prefix, batch_size, target_shape):
        super().__init__(content, prefix, batch_size, target_shape)               

        if prefix :
            self.content_meta = list(map(lambda x: prefix + x, self.content_meta))  # files url from dataset     
        

    def _prepare_data(self):
        self.ready = True

       
    async def get_data(self):

        self._prepare_data()       

        # fetching data
        sem = asyncio.Semaphore(MAX_SESSIONS)
        tasks = [_fetch_file(url, sem, self.shape) for url in self.content_meta]
        fetched_data = await asyncio.gather(*tasks)
        
        for i, result in enumerate(fetched_data):            
            data, err_msg = result
            self.data.append(data)
            if err_msg != '':
                self.error_tape[i] = err_msg

        iterations = math.ceil(len(self.data) / self.batch_size)         
        
        for i in range(iterations):
            data_len  = len(self.data)
            low_index = i * self.batch_size
            up_index = low_index + self.batch_size if (low_index + self.batch_size) < data_len else data_len
            yield self.data[ low_index : up_index ]
       
        