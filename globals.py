# coding=utf-8
# 兼容python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf

# -------------------Constant Variable--------------------------
SEED = 1

NUM_CLASSES = 3

NUM_VIEWS = 12

IMAGE_SHAPE = (227, 227, 3)

VIEWS_IMAGE_SHAPE = (NUM_VIEWS, 227, 227, 3)

# for image which type is uint8, it's depth is 255
IMAGE_DEPTH = 255

NUM_TRAIN_EPOCH = 2

TRAIN_BATCH_SIZE = 1

# TRAIN_LIST = 'data/train_lists_demo.txt'
TRAIN_LIST = 'data/train_lists_demo.txt'

# VAL_LIST = 'data/train_lists_demo.txt'
VAL_LIST = 'data/val_lists_demo.txt'

# because I haven't test data, so I use validation data to for demo
TEST_LIST = 'data/test_lists_demo.txt'

# --------------------------------------------------------------


# def set_seed():
#     """
#     set seed to obtain reproducible results.
#     """
#     os.environ['PYTHONHASHSEED'] = str(SEED)
#     np.random.seed(seed=SEED)
#     tf.set_random_seed(seed=SEED)
#     random.seed(SEED)
