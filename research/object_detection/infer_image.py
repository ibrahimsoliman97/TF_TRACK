import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
import cv2
import glob
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile("./ibrahim_training/exports/track_ssd_v3_large_512/frozen_inference_graph.pb", 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap('data/mscoco_label_map.pbtxt', use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
        # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks', 'nms_embedding', 'raw_detection_boxes', 'raw_detection_scores'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['raw_detection_boxes'] = output_dict['raw_detection_boxes'][0]
            output_dict['raw_detection_scores'] = output_dict['raw_detection_scores'][0]
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'nms_embedding' in output_dict:
                output_dict['nms_embedding'] = output_dict['nms_embedding'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# for image_path in glob.glob("images_test/*.jpg"):
#     image = Image.open(image_path)
#   # the array based representation of the image will be used later in order to prepare the
#   # result image with boxes and labels on it.
#     image_np = load_image_into_numpy_array(image)
#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#     # Actual detection.
#     output_dict = run_inference_for_single_image(image_np, detection_graph)
#     print(output_dict['nms_embedding'].shape)
#     print(output_dict['detection_scores'])
#     # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks'),
#         use_normalized_coordinates=True,
#         line_thickness=8)
#     cv2.imwrite(image_path+"_output.jpg", image_np)

image = Image.open("test1.png")
# the array based representation of the image will be used later in order to prepare the
# result image with boxes and labels on it.
image_np = load_image_into_numpy_array(image)
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
image_np_expanded = np.expand_dims(image_np, axis=0)
# Actual detection.
output_dict = run_inference_for_single_image(image_np, detection_graph)
print(output_dict['nms_embedding'].shape)
output_dict['raw_detection_scores'] = output_dict['raw_detection_scores'].reshape((7, 5118))
print(output_dict['raw_detection_scores'][:,1513])
print(output_dict['raw_detection_scores'][:,1534])
print(output_dict['raw_detection_boxes'][1513])
print(output_dict['raw_detection_boxes'][1534])
# Visualization of the results of a detection.
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)
cv2.imwrite("test1_output.jpg", image_np)