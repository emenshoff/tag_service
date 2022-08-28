import os
import sys
import shutil
import base64

import time
import logging
import random

from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
jpeg = TurboJPEG()


from utils.io import *
import config
from tests.config import TEST_MEDIA_PATH


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

batch_size = 2000
print(f'Batch size: {batch_size}')

prefix = TEST_MEDIA_PATH

fnames = os.listdir(prefix)
fnames = fnames[:batch_size]
fpaths = [os.path.join(prefix, fname) for fname in fnames]

data = []
std_encoded_data = []
tf_encoded_data = []


def img_load(path):
    
    with open(path, 'rb') as in_file:
        bgr_array = jpeg.decode(in_file.read(), flags=TJFLAG_FASTUPSAMPLE|TJFLAG_FASTDCT)
    return bgr_array[:, :, ::-1]


def img_save(rgb_array, path):
    bgr_array = rgb_array[:, :, ::-1]
    with open(path, 'wb') as out_file:
        out_file.write(jpeg.encode(bgr_array))


def time_it(fn):
    # log.debug(f"Benchmark for {fn.__name__}. IO_DEVICE: {config.IO_DEVICE}")
    # log.debug(f"Files to process: {batch_size}")
    t1 = time.time()
    fn()
    t2 = time.time()
    processing_time = t2 - t1
    # log.debug(f"Benchmark for {fn.__name__} completed. Processing time:{processing_time:.4f} sec. Average_time is {processing_time/batch_size:.4f} sec per image. IO_DEVICE: {config.IO_DEVICE}")
    print(f"Benchmark for {fn.__name__} completed. Processing time:{processing_time:.4f} sec. Average_time is {processing_time/batch_size:.4f} sec per image. IO_DEVICE: {config.IO_DEVICE}")

    return fn


# jpeg decoder: CPU
@time_it
def test_jpeg():
    for fpath in fpaths:
        img = read_jpg(fpath)
        data.append(img.numpy())


# universal decoder
@time_it
def test_universal():
    for fpath in fpaths:
        img = read_img(fpath)

# turbo jpeg reader \ decoder
@time_it
def test_turbo():
    for fpath in fpaths:
        img = img_load(fpath)


# standard base64
@time_it
def test_std_b64enc():
    for item in data:
        encoded = base64.b64encode(item.tobytes())#.decode('utf-8')
        std_encoded_data.append(encoded)
        

# tf base 64
@time_it
def test_tf_b64enc():
    for item in data:
        encoded = encode_base64(item.tobytes())
        tf_encoded_data.append(encoded)




# standard base64
@time_it
def test_std_b64dec():
    for item in std_encoded_data:
        decoded = base64.b64decode(item)
        

# tf base 64
@time_it
def test_tf_b64dec():
    for item in tf_encoded_data:
        decoded = decode_base64(item)


'''
Results:

test 1:

25-Feb-21 14:13:12: tf_load: INFO: Configuring TensorFlow...
25-Feb-21 14:13:13: tf_load: INFO: Operating System: linux
25-Feb-21 14:13:13: tf_load: INFO: GPU_2_name: /device:GPU:0
25-Feb-21 14:13:13: tf_load: INFO: GPU_2_memory_limit: 6773403808
25-Feb-21 14:13:13: tf_load: INFO: GPU_2_description: device: 0, name: GeForce RTX 2070, pci bus id: 0000:01:00.0, compute capability: 7.5
25-Feb-21 14:13:13: tf_load: INFO: CPU(s): 12
25-Feb-21 14:13:13: tf_load: INFO: Thread(s) per core: 2
25-Feb-21 14:13:13: tf_load: INFO: Core(s) per socket: 6
25-Feb-21 14:13:13: tf_load: INFO: Model name: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
25-Feb-21 14:13:13: tf_load: INFO: Setting 'memory_growth' mode for GPU device: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
25-Feb-21 14:13:13: tf_load: INFO: Configuring complete...
Batch size: 2000
25-Feb-21 14:13:32: decode_benchmark: DEBUG: Benchmark for test_jpeg completed. Processing time:18.0960 sec. Average_time is 0.0090 sec per image. IO_DEVICE: CPU:0
25-Feb-21 14:13:48: decode_benchmark: DEBUG: Benchmark for test_universal completed. Processing time:16.0336 sec. Average_time is 0.0080 sec per image. IO_DEVICE: CPU:0
25-Feb-21 14:13:53: decode_benchmark: DEBUG: Benchmark for test_std_b64enc completed. Processing time:4.7253 sec. Average_time is 0.0024 sec per image. IO_DEVICE: CPU:0
25-Feb-21 14:15:15: decode_benchmark: DEBUG: Benchmark for test_tf_b64enc completed. Processing time:81.9358 sec. Average_time is 0.0410 sec per image. IO_DEVICE: CPU:0
25-Feb-21 14:15:56: decode_benchmark: DEBUG: Benchmark for test_std_b64dec completed. Processing time:40.7693 sec. Average_time is 0.0204 sec per image. IO_DEVICE: CPU:0
25-Feb-21 14:16:06: decode_benchmark: DEBUG: Benchmark for test_tf_b64dec completed. Processing time:9.8085 sec. Average_time is 0.0049 sec per image. IO_DEVICE: CPU:0

test 2

05-Oct-21 18:18:05: tf_load: INFO: Found GPU device: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
05-Oct-21 18:18:05: tf_load: INFO: Configuring TensorFlow...
05-Oct-21 18:18:06: tf_load: INFO: Operating System: linux
05-Oct-21 18:18:06: tf_load: INFO: GPU_2_name: /device:GPU:0
05-Oct-21 18:18:06: tf_load: INFO: GPU_2_memory_limit: 1289355264
05-Oct-21 18:18:06: tf_load: INFO: GPU_2_description: device: 0, name: GeForce RTX 3080 Ti, pci bus id: 0000:04:00.0, compute capability: 8.6
05-Oct-21 18:18:06: tf_load: INFO: CPU(s): 28
05-Oct-21 18:18:06: tf_load: INFO: Thread(s) per core: 2
05-Oct-21 18:18:06: tf_load: INFO: Core(s) per socket: 14
05-Oct-21 18:18:06: tf_load: INFO: Setting 'memory_growth' mode for GPU device: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
05-Oct-21 18:18:06: tf_load: INFO: Setting IO_DEVICE: CPU:0
05-Oct-21 18:18:06: tf_load: INFO: Setting PREPROCESSING_DEVICE: CPU:0
05-Oct-21 18:18:06: tf_load: INFO: Setting INFERENCE_DEVICE: GPU:0
05-Oct-21 18:18:06: tf_load: INFO: TensorFlow configuring complete...
Batch size: 2000
Benchmark for test_jpeg completed. Processing time:5.3633 sec. Average_time is 0.0027 sec per image. IO_DEVICE: CPU:0
Benchmark for test_universal completed. Processing time:4.1384 sec. Average_time is 0.0021 sec per image. IO_DEVICE: CPU:0
Benchmark for test_turbo completed. Processing time:3.5228 sec. Average_time is 0.0018 sec per image. IO_DEVICE: CPU:0
Benchmark for test_std_b64enc completed. Processing time:7.1900 sec. Average_time is 0.0036 sec per image. IO_DEVICE: CPU:0
Benchmark for test_tf_b64enc completed. Processing time:49.3417 sec. Average_time is 0.0247 sec per image. IO_DEVICE: CPU:0
Benchmark for test_std_b64dec completed. Processing time:10.0691 sec. Average_time is 0.0050 sec per image. IO_DEVICE: CPU:0
Benchmark for test_tf_b64dec completed. Processing time:8.3545 sec. Average_time is 0.0042 sec per image. IO_DEVICE: CPU:0
'''