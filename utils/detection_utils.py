import numpy as np

from tf_load import tf


def _denorm_box_coordinates_np(img_size, box):
    ''' 
    Converts box coordinates from normilized to real 
    numpy \ python version
    '''
    im_height, im_width = img_size
    y = box[0] * im_height
    x = box[1] * im_width
    h = box[2] * im_height
    w = box[3] * im_width

    if isinstance(box, np.ndarray):
        y = y.astype(np.int32)
        x = x.astype(np.int32)
        h = h.astype(np.int32)
        w = w.astype(np.int32)

    else:
        y = int(y)
        x = int(x)
        h = int(h)
        w = int(w)
    
    return (y, x, h, w)


@tf.function
def _denorm_box_coordinates_tf(img_size, box):
    ''' 
    Converts box coordinates from normilized to real 
    tensorflow version
    '''
    im_height, im_width = img_size
    y = tf.multiply(box[0], im_height)
    x = tf.multiply(box[1], im_width)
    h = tf.multiply(box[2], im_height)
    w = tf.multiply(box[3], im_width)

    x = tf.cast(x, tf.int32)
    y = tf.cast(y, tf.int32)
    w = tf.cast(w, tf.int32)
    h = tf.cast(h, tf.int32)
    
    return (y, x, h, w)


def denorm_box_coordinates(img_size, box):
    ''' 
    Converts box coordinates from normilized to real 
    '''
    if isinstance(box[0], tf.Tensor):
        return _denorm_box_coordinates_tf(img_size, box).numpy()
    return _denorm_box_coordinates_np(img_size, box)


def box_iou(box1, box2):
    '''
    Calculates iou for two box. Numpy-compatibale.
    '''
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2    
    
    box1_width = abs(xmax1 - xmin1)
    box1_heigth = abs(ymax1 - ymin1)
    box1_area = box1_width * box1_heigth
    
    box2_width = abs(xmax2 - xmin2)
    box2_heigth = abs(ymax2 - ymin2)
    box2_area = box2_width * box2_heigth

    x_inter1 = max(xmin1, xmin2)
    y_inter1 = max(ymin1, ymin2)
    x_inter2 = min(xmax1, xmax2)
    y_inter2 = min(ymax1, ymax2)
    
    intersection_width = (x_inter2 - x_inter1)
    intersection_height = (y_inter2 - y_inter1) 
    
    if any([(intersection_width < 0), (intersection_height < 0)]):
        intersection_area = 0
    else:
        intersection_area = intersection_width * intersection_height
    
    full_area = box1_area + box2_area
    union_area = full_area - intersection_area
    iou = intersection_area / union_area
    return iou


def check_overlap(box1, box2, threshold):
    ''' Overlap check with threshold'''
    return box_iou(box1, box2) > threshold


def _restore_point_aspect_ratio_np(point, img_size, original_img_size):
    '''
    Restore original coordinates. Can be used in postprocessing.
    Numpy version.
    '''
    x_factor = img_size[1] / original_img_size[1]
    y_factor = img_size[0] / original_img_size[0] 
    point = point.astype(np.float64)
    point[0] *= y_factor 
    point[1] *= x_factor      
    box = point.astype(np.int32) 


def _restore_box_aspect_ratio_np(box, img_size, original_img_size): # REFACTOR
    '''
    Restore original box coordinates. Can be used in postprocessing.
    Numpy version.
    '''
    x_factor = img_size[1] / original_img_size[1]
    y_factor = img_size[0] / original_img_size[0] 
    box = np.array(box).astype(np.float32)
    box[0] *= y_factor 
    box[2] *= y_factor 
    box[1] *= x_factor 
    box[3] *= x_factor 

    # check if box is in bounds of original image
    if (box[1] < 0): box[1] = 0
    if (box[0] < 0): box[0] = 0
    if (box[3] > original_img_size[1]): box[3] = original_img_size[1]
    if (box[2] > original_img_size[0]): box[2] = original_img_size[0]
    box = box.astype(np.int32).tolist()
    return box


@tf.function
def _restore_aspect_ratio_tf(box, img_size, original_img_size):
    '''
    Restore original box coordinates. Can be used in postprocessing.
    Tensorflow autograph-compatible
    '''
    x_factor = img_size[1] / original_img_size[1]
    y_factor = img_size[0] / original_img_size[0] 
    box = tf.cast(box, tf.float64)
    box[0] = tf.math.multiply(box[0],  y_factor)
    box[2] = tf.math.multiply(box[2],  y_factor)
    box[1] = tf.math.multiply(box[1],  x_factor) 
    box[3] = tf.math.multiply(box[3],  x_factor)
    # check if box is in bounds of original image
    if (box[1] < 0): box[1] = 0
    if (box[0] < 0): box[0] = 0
    if (box[3] > original_img_size[1]): box[3] = original_img_size[1]
    if (box[2] > original_img_size[0]): box[2] = original_img_size[0]
    box = tf.cast(box, tf.int32) 
    return box


def restore_box_aspect_ratio(box, img_size, original_img_size):
    if isinstance(box[0], tf.Tensor):
        return _restore_aspect_ratio_tf(box, img_size, original_img_size).numpy()
    
    return _restore_box_aspect_ratio_np(box, img_size, original_img_size)