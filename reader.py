"""Reader for dataset used in the SLIM paper.

Example usage:

filenames, iterator, next_element = make_dataset(batch_size=16)

with tf.Session() as sess:
  # Initialize `iterator` with train data.
  # training_filenames = ["/var/data/train_1.tfrecord", ...]
  sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

  ne_value = sess.run(next_element)

  # Initialize `iterator` with validation data.
  # validation_filenames = ["/var/data/train_1.tfrecord", ...]
  # sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})

  ne_value = sess.run(next_element)

`next_element` is a tuple containing the query, the target, and the raw data.
The query is a tuple where the first element is the
sequence of 9 (images, cameras, captions) which can be given to the model
as context. The second element in the query is the camera angle of the
viewpoint to reconstruct. The target contains the image corresponding to the
queried viewpoint, the text description from that viewpoint and an image of
the scene viewed from above.
The raw data is a dictionary with all the fields as read from the tf.Record as
described in the documentation for `_parse_proto`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_NUM_VIEWS = 10
_NUM_RAW_CAMERA_PARAMS = 3
_IMAGE_SCALE = 0.25
_USE_SIMPLIFIED_CAPTIONS = False
_PARSE_METADATA = False


def _parse_proto(buf):
  """Parse binary protocol buffer into tensors.

  The protocol buffer is expected to contain the following fields:
    * frames: 10 views of the scene rendered as images.
    * top_down_frame: single view of the scene from above rendered as an image.
    * cameras: 10 vectors describing the camera position from which the frames
        have been rendered
    * captions: A string description of the scene. For the natural language
        dataset, contains descriptions written by human annotators. For
        synthetic data contains a string describing each relation between
        objects in the scene exactly once.
    * simplified_captions: A string description of the scene. For the natural
        language dataset contains a string describing each relation between
        objects in the scene exactly once. For synthetic datasets contains
        a string describing every possible pairwise relation between objects in
        the scene.
    * meta_shape: A vector of strings describing the object shapes.
    * meta_color: A vector of strings describing the object colors.
    * meta_size: A vector of strings describing the object sizes.
    * meta_obj_positions: A matrix of floats describing the position of each
        object in the scene.
    * meta_obj_rotations: A matrix of floats describing the rotation of each
        object in the scene.
    * meta_obj_rotations: A matrix of floats describing the color of each
        object in the scene as RGBA in the range [0, 1].

  Args:
    buf: A string containing the serialized protocol buffer.

  Returns:
    A dictionary containing tensors for each of the fields in the protocol
    buffer. If _PARSE_METADATA is False, will omit fields starting with 'meta_'.
  """
  feature_map = {
      "frames":
          tf.FixedLenFeature(shape=[_NUM_VIEWS], dtype=tf.string),
      "top_down_frame":
          tf.FixedLenFeature(shape=[1], dtype=tf.string),
      "cameras":
          tf.FixedLenFeature(
              shape=[_NUM_VIEWS * _NUM_RAW_CAMERA_PARAMS], dtype=tf.float32),
      "captions":
          tf.VarLenFeature(dtype=tf.string),
      "simplified_captions":
          tf.VarLenFeature(dtype=tf.string),
      "meta_shape":
          tf.VarLenFeature(dtype=tf.string),
      "meta_color":
          tf.VarLenFeature(dtype=tf.string),
      "meta_size":
          tf.VarLenFeature(dtype=tf.string),
      "meta_obj_positions":
          tf.VarLenFeature(dtype=tf.float32),
      "meta_obj_rotations":
          tf.VarLenFeature(dtype=tf.float32),
      "meta_obj_colors":
          tf.VarLenFeature(dtype=tf.float32),
  }

  example = tf.parse_single_example(buf, feature_map)
  images = tf.concat(example["frames"], axis=0)
  images = tf.map_fn(
      tf.image.decode_jpeg,
      tf.reshape(images, [-1]),
      dtype=tf.uint8,
      back_prop=False)
  top_down = tf.image.decode_jpeg(tf.squeeze(example["top_down_frame"]))
  cameras = tf.reshape(example["cameras"], shape=[-1, _NUM_RAW_CAMERA_PARAMS])
  captions = tf.sparse_tensor_to_dense(example["captions"], default_value="")
  simplified_captions = tf.sparse_tensor_to_dense(
      example["simplified_captions"], default_value="")
  meta_shape = tf.sparse_tensor_to_dense(
      example["meta_shape"], default_value="")
  meta_color = tf.sparse_tensor_to_dense(
      example["meta_color"], default_value="")
  meta_size = tf.sparse_tensor_to_dense(example["meta_size"], default_value="")
  meta_obj_positions = tf.sparse_tensor_to_dense(
      example["meta_obj_positions"], default_value=0)
  meta_obj_positions = tf.reshape(meta_obj_positions, shape=[-1, 3])
  meta_obj_rotations = tf.sparse_tensor_to_dense(
      example["meta_obj_rotations"], default_value=0)
  meta_obj_rotations = tf.reshape(meta_obj_rotations, shape=[-1, 4])
  meta_obj_colors = tf.sparse_tensor_to_dense(
      example["meta_obj_colors"], default_value=0)
  meta_obj_colors = tf.reshape(meta_obj_colors, shape=[-1, 4])

  data_tensors = {
      "images": images,
      "cameras": cameras,
      "captions": captions,
      "simplified_captions": simplified_captions,
      "top_down": top_down
  }
  if _PARSE_METADATA:
    data_tensors.update({
        "meta_shape": meta_shape,
        "meta_color": meta_color,
        "meta_size": meta_size,
        "meta_obj_positions": meta_obj_positions,
        "meta_obj_rotations": meta_obj_rotations,
        "meta_obj_colors": meta_obj_colors
    })
  return data_tensors


def _make_indices():
  indices = tf.range(0, _NUM_VIEWS)
  indices = tf.random_shuffle(indices)
  return indices


def _convert_and_resize_images(images, old_size):
  images = tf.image.convert_image_dtype(images, dtype=tf.float32)
  new_size = tf.cast(old_size, tf.float32) * _IMAGE_SCALE
  new_size = tf.cast(new_size, tf.int32)
  images = tf.image.resize_images(images, new_size, align_corners=True)
  return images


def _preprocess_images(images, indices):
  images_processed = tf.gather(images, indices)
  old_size = tf.shape(images_processed)[1:3]
  images_processed = _convert_and_resize_images(images_processed, old_size)
  return images_processed


def _preprocess_td(td_image):
  old_size = tf.shape(td_image)[0:2]
  td_image = _convert_and_resize_images(td_image, old_size)
  return td_image


def _preprocess_cameras(raw_cameras, indices):
  """Apply a nonlinear transformation to the vector of camera angles."""
  raw_cameras = tf.gather(raw_cameras, indices)
  azimuth = raw_cameras[:, 0]
  pos = raw_cameras[:, 1:]
  cameras = tf.concat(
      [
          pos,
          tf.expand_dims(tf.sin(azimuth), -1),
          tf.expand_dims(tf.cos(azimuth), -1)
      ],
      axis=1)
  return cameras


def _preprocess_captions(raw_caption, indices):
  return tf.gather(raw_caption, indices)


def _preprocess_data(raw_data):
  """Randomly shuffle viewpoints and apply preprocessing to each modality."""
  indices = _make_indices()
  images = _preprocess_images(raw_data["images"], indices)
  cameras = _preprocess_cameras(raw_data["cameras"], indices)
  top_down = _preprocess_td(raw_data["top_down"])
  if _USE_SIMPLIFIED_CAPTIONS:
    captions = _preprocess_captions(raw_data["simplified_captions"], indices)
  else:
    captions = _preprocess_captions(raw_data["captions"], indices)
  return [images, cameras, top_down, captions]


def _split_scene(images, cameras, top_down, captions):
  """Splits scene into query and target.

  Args:
    images: A tensor containing images.
    cameras: A tensor containing cameras.
    top_down: A tensor containing the scene seen from top.
    captions: A tensor containing captions.

  Returns:
    A tuple query, target. The query is a tuple where the first element is the
    sequence of 9 (images, cameras, captions) which can be given to the model
    as context. The second element in the query is the camera angle of the
    viewpoint to reconstruct. The target contains the image corresponding to the
    queried viewpoint, the text description from that viewpoint and an image of
    the scene viewed from above.
  """
  context_image = images[:-1, :, :, :]
  context_camera = cameras[:-1, :]
  context_caption = captions[:-1]
  target_image = images[-1, :, :, :]
  target_camera = cameras[-1, :]
  target_caption = captions[-1]

  query = ((context_image, context_camera, context_caption), target_camera)
  target = (target_image, target_caption, top_down)
  return query, target


def _parse_function(buf):
  raw_data = _parse_proto(buf)
  scene_data = _preprocess_data(raw_data)
  query, target = _split_scene(*scene_data)
  return query, target, raw_data


def make_dataset(batch_size):
  """Returns a tf.data.Dataset object with the dataset."""
  filenames = tf.placeholder(tf.string, shape=[None])
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_function)
  dataset = dataset.repeat()
  dataset = dataset.shuffle(128)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  return filenames, iterator, next_element
