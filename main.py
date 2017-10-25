import io
import os
import sys
import time
import argparse
from time import gmtime, strftime
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

# Data
tf.app.flags.DEFINE_string('root_dir', '', """Base Path""")
tf.app.flags.DEFINE_string('dataset_dir', 'data', """Path to data""")
tf.app.flags.DEFINE_integer('image_h', 224, """Shape of image height""")
tf.app.flags.DEFINE_integer('image_w', 224, """Shape of image width""")
tf.app.flags.DEFINE_string('dataset', '', """Specify the dataset""")

# Training
tf.app.flags.DEFINE_integer('batch_size', 64, """Batch size""")
tf.app.flags.DEFINE_integer('MAX_epochs', 1000, """Max epochs for training""")
tf.app.flags.DEFINE_integer('ckpt_frq', 300, """Frequency at which to checkpoint the model""")
tf.app.flags.DEFINE_integer('train_size', 10000, """The total training size""")
tf.app.flags.DEFINE_integer('generate_frq', 1, """The frequency with which to sample images""")
tf.app.flags.DEFINE_integer('display', 1, """Display log of progress""")
tf.app.flags.DEFINE_float('D_base_lr', 2e-5, """Base learning rate for Discriminator""")
tf.app.flags.DEFINE_float('G_base_lr', 2e-5, """Base learning rate for Generator""")
tf.app.flags.DEFINE_boolean('train', False, """Training or testing""")
tf.app.flags.DEFINE_boolean('test', False, """Training or testing""")
tf.app.flags.DEFINE_boolean('resume', False, """Resume the training ?""")


# Architecture
tf.app.flags.DEFINE_integer('code_len', 100, """Latent code length in case of GAN""")
tf.app.flags.DEFINE_integer('dims', 32, """Number of kernels for the first convolutional lalyer in the network for GAN/VAE""")

# Model Saving
tf.app.flags.DEFINE_string('ckpt_dir', "ckpt", """Checkpoint Directory""")
tf.app.flags.DEFINE_string('sample_dir', "imgs", """Generate sample images""")
tf.app.flags.DEFINE_string('summary_dir', "summary", """Summaries directory""")


def main(_):
   pass

if __name__ == '__main__':
   try:
      tf.app.run()
   except Exception as E:
      print E
