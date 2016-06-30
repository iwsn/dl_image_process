import gzip
import os
import re
import sys
import tarfile
import urllib
import tensorflow as tf
import string
import numpy as np
import PIL as Image
import struct
import tensorflow as tf
import glob
import time
import imghdr

# this script is modified from google inception build_imagenet_data.py
#convert data to inceptio tf record files
# ==============================================================================
"""Converts ImageNet data to TFRecords file format with Example protos.
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.

Running this script using 16 threads may take around ~2.5 hours on a HP Z420.
"""

from datetime import datetime
import os
import random
import sys
import threading


import numpy as np
import tensorflow as tf

#multi thread to process file
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convertToExample(filename, imageBuffer, labelId, human, synset, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(labelId),
      'image/class/synset': _bytes_feature(synset),
      'image/class/text': _bytes_feature(human),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([labelId] * len(xmin)),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(imageBuffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def processOneImage(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.

  print("current file path:",filename)
  imageData = tf.gfile.FastGFile(filename, 'r').read()

  imageType = imghdr.what(filename)

  if(imageType == "png"):
      imageData = coder.png_to_jpeg(imageData)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(imageData)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return imageData, height, width


def processImagesBatch(coder, threadIndex, ranges, dstDir, imagePaths, imageLabels
                               , idDict, nameDict, dataName, batch):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    threadIndex: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    imagePaths: list of strings; each string is a path to an image file
    imageLabels: list of integer; each integer identifies the ground truth
    labelDicts: map of <integer, string>; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    batch: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  numThreads = len(ranges)
  assert not batch % numThreads
  numShardsPerBatch = int(batch / numThreads)

  shardRanges = np.linspace(ranges[threadIndex][0],
                             ranges[threadIndex][1],
                             numShardsPerBatch + 1).astype(int)
  numFilesInThread = ranges[threadIndex][1] - ranges[threadIndex][0]

  counter = 0
  for s in xrange(numShardsPerBatch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = threadIndex * numShardsPerBatch + s
    outputFileName = '%s-%.5d-of-%.5d' % (dataName, shard, batch)
    outputFile = os.path.join(dstDir, outputFileName)
    writer = tf.python_io.TFRecordWriter(outputFile)

    shard_counter = 0

    filesInShard = np.arange(shardRanges[s], shardRanges[s + 1], dtype=int)

    for i in filesInShard:
      imagePath = imagePaths[i]
      labelId = imageLabels[i]
      synset = idDict[labelId]
      human = nameDict[synset]

      try :
        imageBuffer, height, width = processOneImage(imagePath, coder)
      except :
        print("exception path data:", imagePath)
        continue

      example = convertToExample(imagePath, imageBuffer, labelId, human, synset, height, width)

      writer.write(example.SerializeToString())

      shard_counter += 1

      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), threadIndex, counter, numFilesInThread))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), threadIndex, shard_counter, outputFile))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), threadIndex, counter, numFilesInThread))
  sys.stdout.flush()

#multi thread to process generate file
def processImagesFile(dataName, idDict, nameDict, imagePaths, imageLabels, dstDir, batch):
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(imagePaths), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for threadIndex in xrange(len(ranges)):
        args = (coder, threadIndex, ranges, dstDir, imagePaths,
                imageLabels, idDict, nameDict, dataName, batch)
        t = threading.Thread(target=processImagesBatch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(imagePaths)))
    sys.stdout.flush()


def convertToInception(dataName, idDict, nameDict, imagePaths, imageLabels, dstDir, batch):
    processImagesFile(dataName, idDict, nameDict, imagePaths, imageLabels, dstDir, batch)

