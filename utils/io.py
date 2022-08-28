import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging

from tf_load import tf

tf.get_logger().setLevel('ERROR')   

from config import IO_DEVICE, PREPROCESSING_DEVICE


class UnsupportedImageFormat(Exception):
    pass

import logging
log = logging.getLogger(__name__)

'''
Image io, encoding and decoding utilittes,  TensorFlow - based
'''

@tf.function
def resize_image(img_tensor, target_size):
    with tf.device(PREPROCESSING_DEVICE):
        img_tensor = tf.image.resize(img_tensor, target_size)
        img_tensor = tf.cast(img_tensor, tf.uint8)
        return img_tensor


@tf.function
def zeros(target_shape):
    with tf.device(PREPROCESSING_DEVICE):
        return tf.zeros(shape=target_shape, dtype=tf.uint8)


# png encoder
@tf.function
def encode_png(img_tensor):
    with tf.device(PREPROCESSING_DEVICE):
        return tf.io.encode_png(img_tensor, compression=-1)


# png decoder
@tf.function
def decode_png(binary_data):
    with tf.device(PREPROCESSING_DEVICE):
        return tf.io.decode_png(binary_data)


@tf.function
def encode_jpeg(img_tensor):
    with tf.device(PREPROCESSING_DEVICE):
       return tf.io.encode_jpeg(img_tensor)
    #     , format='', quality=95, progressive=False, optimize_size=False,
    #     chroma_downsampling=True, density_unit='in', x_density=300,
    #     y_density=300, xmp_metadata='', name=None
    # )


# jpeg decoder
@tf.function
def decode_jpg(binary_data):
    with tf.device(PREPROCESSING_DEVICE):        
        return tf.io.decode_jpeg(binary_data)

'''
    эксперименты показали:
    1) функции кодирования из tf.io и стандартного модуля base64 дают разные результаты и не взаимозаменяемы. 
    рекомендую использовать в клиентском коде только функции из стандартного модуля base64
    tf.io ругается на символы, закодированные при помощи b64encode / standard_b64encode
    кроме того, tf.io.encode_base64 от tf медленней  в 10 раз и жрет больше памяти. декодер - наоборот.
    2) tf.io.read_file жрет память ужасным образом (TF 2.5, built from source)
'''
# base64 decoder
@tf.function
def decode_base64(img_tf_byte_string):
    with tf.device(PREPROCESSING_DEVICE):
        return tf.io.decode_base64(img_tf_byte_string)


# base64 encoder
@tf.function
def encode_base64(img_tf_byte_string):
    with tf.device(PREPROCESSING_DEVICE):
       return tf.io.encode_base64(img_tf_byte_string)
      

# decode bmp, gif, MP, GIF, also works fine with JPEG, or PNG, but slower
@tf.function
def decode_img(binary_data):
    with tf.device(PREPROCESSING_DEVICE):
        return tf.io.decode_image(binary_data, channels=3, dtype=tf.dtypes.uint8,  expand_animations=False) # takes first frame for gif

        # img_tensor = tf.io.decode_image(binary_data)
        # img_tensor = tf.reduce_max(img_tensor,1) # scipping "frame" channel
        # decoded_img = tf.cast(img_tensor, tf.uint8)
        # return decoded_img


# read png from file
# @tf.function
def read_png(fpath, target_size=None):
    # img_string = open(fpath, 'rb').read()
    
    with tf.device(IO_DEVICE):
        # img_string = tf.io.read_file(fpath)    
        img_string = tf.io.gfile.GFile(fpath, 'rb').read()   
        img_tensor = tf.io.decode_png(img_string, channels=3, dtype=tf.dtypes.uint8)
        if target_size is not None:
            img_tensor = tf.image.resize(img_tensor, target_size)
            img_tensor = tf.cast(img_tensor, tf.uint8)
        return img_tensor


# read jpg from file
# @tf.function
def read_jpg(fpath, target_size=None):
    
    with tf.device(IO_DEVICE):
        # img_string = tf.io.read_file(fpath)   
        img_string = tf.io.gfile.GFile(fpath, 'rb').read()    
        img_tensor = tf.io.decode_jpeg(img_string)
        if target_size is not None:
            img_tensor = tf.image.resize(img_tensor, target_size)
            img_tensor = tf.cast(img_tensor, tf.uint8)
        return img_tensor


# read bpm, png, jpg gif from file
# @tf.function
def read_img(fpath, target_size=None):
    
    with tf.device(IO_DEVICE):
        # img_string = tf.io.read_file(fpath)       # commented for memory eating
        img_string = tf.io.gfile.GFile(fpath, 'rb').read()
        img_tensor = tf.io.decode_image(img_string, channels=3, dtype=tf.dtypes.uint8,  expand_animations=False) # takes first frame for gif
        if target_size is not None:
            img_tensor = tf.image.resize(img_tensor, target_size)
            img_tensor = tf.cast(img_tensor, tf.uint8)
        return img_tensor


# write jpeg
@tf.function
def write_img(file_path, binary_data):
    with tf.device(IO_DEVICE):
        contents = tf.io.encode_jpeg(binary_data)
        tf.io.write_file(file_path, contents)


@tf.function
def load_img_bin(binary_data, target_size=None):
    with tf.device(IO_DEVICE):
        
        img_tensor = tf.io.decode_image(binary_data, channels=3, dtype=tf.dtypes.uint8,  expand_animations=False) # takes first frame for gif
        
        # img_tensor = tf.reduce_max(img_tensor,1) # scipping "frame" channel
        if target_size is not None:
            img_tensor = tf.image.resize(img_tensor, target_size)
            img_tensor = tf.cast(img_tensor, tf.uint8)
        return img_tensor


@tf.function
def load_jpg_bin(binary_data, target_size=None):
    with tf.device(IO_DEVICE):
        img_tensor = tf.io.decode_jpeg(binary_data)
        if target_size is not None:
            img_tensor = tf.image.resize(img_tensor, target_size)
            img_tensor = tf.cast(img_tensor, tf.uint8)
        return img_tensor

# @tf.function
def load_image(fpath, target_size=None):
    '''
    Loads image from filesystem and resizes it.    
    Supports jpg, png, bmp and gif
    '''
    
    img_tensor = None    

    img_format = fpath.split('.')[-1]
    
    # ".original" extension patch
    if img_format == 'original':
        img_format = fpath.split('.')[-2] # back to real file extension

    if img_format == 'jpg' or img_format == 'jpeg':
        img_tensor =  read_jpg(fpath, target_size)
    elif img_format == 'png':
        img_tensor = read_png(fpath, target_size)
    elif img_format == 'bmp' or img_format == 'gif':
        img_tensor = read_img(fpath, target_size)
    else:
        raise  UnsupportedImageFormat(f'Unsupported image format: {img_format}')

    return img_tensor

