# Datasets used to train Spatial Language Integrating Model (SLIM) in the ‘Encoding Spatial Relations from Natural Language’ paper.

This dataset consists of virtual scenes rendered in MuJoCo with multiple views
each presented in multiple modalities: image, and synthetic or natural language
descriptions. Each scene consists of two or three objects placed on a square
walled room, and for each of the 10 camera viewpoint we render a 3D view of the
scene as seen from that viewpoint as well as a synthetically generated
description of the scene.

# Synthetic data

We generated a dataset of 12 million 3D scenes. Each scene contains two or three
coloured 3D objects and light grey walls and floor. The language descriptions
are generated programmatically, taking into account the underlying scene graph
and camera coordinates so as to describe the spatial arrangement of the objects
as seen from each viewpoint.

# Human annotated data

We generated further scenes and used Amazon Mechanical Turk to collect natural
language descriptions. We asked annotators to describe the room in an image as
if they were describing the image to a friend who needs to draw the image
without seeing it. We asked for a short or a few sentence description that
describes object shapes, colours, relative positions, and relative sizes. We
provided the list of object names together with only two examples of
descriptions, to encourage diversity, while focusing the descriptions on the
spatial relations of the objects. The annotators annotated 6,604 scenes with 10
descriptions each, one for each view.

### Usage example

```python
import tensorflow as tf
import reader

filenames, iterator, next_element = reader.make_dataset(batch_size=16)

with tf.Session() as sess:
  # Initialize `iterator` with train data.
  # training_filenames = ["/var/data/train_1.tfrecord", ...]
  sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

  ne_value = sess.run(next_element)

  # Initialize `iterator` with validation data.
  # validation_filenames = ["/var/data/train_1.tfrecord", ...]
  # sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})

  ne_value = sess.run(next_element)
```

`next_element` is a tuple containing the query, the target, and the raw data.
The query is a tuple where the first element is the sequence of 9 (images,
cameras, captions) which can be given to the model as context. The second
element in the query is the camera angle of the viewpoint to reconstruct. The
target contains the image corresponding to the queried viewpoint, the text
description from that viewpoint and an image of the scene viewed from above. The
raw data is a dictionary with all the fields as read from the tf.Record as
described in the documentation for `_parse_proto`.

### Download

Raw data files referred to in this document are available to download
[here](https://console.cloud.google.com/storage/slim-dataset).

### Notes

This is not an official Google product.
