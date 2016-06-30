import gzip
import os
import re
import sys
import tarfile
import urllib
import tensorflow as tf

LABEL_FILE_NAME="label.txt"
#support data processing demo cnn net work
MAP_DEMO_CNN_TYPE = {"cifar": 1, "inception": 2}


"""used to split map type into train and validation part
fixed: within every pic type, random fetch fixed fraction num pics
random: within the whole pic , random fetch fraction num pics
"""
MAP_SAMPLE_TYPE   = {"fixed":1, "random":2}