import numpy as np
from tf_load import tf

dtype_map = {
    "tf.uint8": tf.uint8,
    "tf.float32": tf.float32,
    "uint8": tf.uint8,
    "float32": tf.float32,
    "tf.string": tf.string
}


tf2np_dtype_map = {
    tf.uint8: np.uint8,
    tf.float32: np.float32,
    tf.string: np.bytes_
}