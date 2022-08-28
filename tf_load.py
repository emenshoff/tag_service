'''
Centralized TensorFlow initialization...
'''
import os
import sys
import logging
import subprocess



import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # Enable fp16 and fb32 mixed operations and tensor cores

# ЖРЕТ ВСЮ ПАМЯТЬ GPU
# from tensorflow.python.client import device_lib

import tensorflow as tf

tf.get_logger().setLevel('ERROR')    # Only TF errors in console    

log = logging.getLogger("engine") # setting module logger

# uncomment for hw details in tests
# logging.basicConfig(
#                     # filename='tests.log', 
#                     # filemode='w',
#                     level=logging.DEBUG,

#                     format="%(asctime)s: %(module)s: %(levelname)s: %(message)s",
#                     datefmt='%d-%b-%y %H:%M:%S'
#                     )
# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)

"""
def get_available_gpus():
    ''' Get available GPU devices info. '''
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_machine_info():
    ''' Get local machine GPU\MEM\CPU info '''
    parameter_value_map = {}
    operating_sys = sys.platform
    parameter_value_map['Operating System'] = operating_sys
    if 'linux' not in operating_sys:
        return parameter_value_map

    for i, device in enumerate(device_lib.list_local_devices()):
        if device.device_type != 'GPU':
            continue
        parameter_value_map['GPU_{}_name'.format(i + 1)] = device.name
        parameter_value_map['GPU_{}_memory_limit'.format(i + 1)] = device.memory_limit
        parameter_value_map['GPU_{}_description'.format(i + 1)] = device.physical_device_desc
    lscpu = subprocess.check_output("lscpu | grep '^CPU(s):\\|Core\\|Thread'", shell=True).strip().decode()
    lscpu = lscpu.split('\n')
    for row in lscpu:
        row = row.split(':')
        parameter_value_map[row[0]] = row[1].strip()
    return parameter_value_map     

"""

gpus = tf.config.experimental.list_physical_devices('GPU')


if not gpus:
    if config.ALLOW_CPU_INFERENCE:
        log.warning(f"GPU not found! Running inference on CPU! (Note: CPU inference slows down prediction time for local models!)")
    else:
        log.critical(f"GPU not found! Exiting...")
        exit(1)

for gpu in gpus:
    log.info(f'Found GPU device: {gpus}')
    tf.config.experimental.set_memory_growth(gpu, True)


def tf_init():
   
    log.info('Configuring TensorFlow...')

    
    # Detect available GPU devices info.
    # gpus = get_available_gpus()          
    gpus = tf.config.experimental.list_physical_devices('GPU')

    
    for gpu in gpus:
        log.info(f"Setting 'memory_growth' mode for GPU device: {gpus}")
        tf.config.experimental.set_memory_growth(gpu, True)

    # ЖРЕТ ВСЮ ПАМЯТЬ GPU, ЕСЛИ ВЫЗВАТЬ
    # machine_info = get_machine_info()
    # for key, val in machine_info.items():
    #     log.info(f"{key}: {val}")
    
    cpu_device_names = tf.config.list_physical_devices(
                device_type='CPU'
            )

    cpu_device = None
    gpu_device = None

    # checking devices available in the system
    if cpu_device_names:
        if len(cpu_device_names[0]):
            cpu_device = ':'.join(cpu_device_names[0][0].split(':')[1:])

    if gpus:
        if len(gpus[0]):
            gpu_device = ':'.join(gpus[0][0].split(':')[1:])

    devices = [cpu_device, gpu_device]

    # validating devices that found in config
    for cfg_device in [config.IO_DEVICE, config.PREPROCESSING_DEVICE, config.INFERENCE_DEVICE]:
        if not cfg_device in devices:
            err_msg = f"device {cfg_device} not found in this system. Exiting..."
            log.critical(err_msg)
            exit(1)
    
    log.info(f"Setting IO_DEVICE: {config.IO_DEVICE}")
    log.info(f"Setting PREPROCESSING_DEVICE: {config.PREPROCESSING_DEVICE}")
    
    # setting inference device
    if not gpus or len(gpus) == 0:
        if config.ALLOW_CPU_INFERENCE:
            log.warning(f"GPU not found! Running inference on CPU! (Note: CPU inference slows down prediction time for local models!)")            
            config.INFERENCE_DEVICE = config.IO_DEVICE
        else:
            log.critical(f"GPU not found! Exiting...")
            exit(1)  
    else:
        config.INFERENCE_DEVICE = gpu_device

    log.info(f"Setting INFERENCE_DEVICE: {config.INFERENCE_DEVICE}")
    log.info('TensorFlow configuring complete...')

    return tf
    
tf =   tf_init()  # one global variable for TensorFlow module
