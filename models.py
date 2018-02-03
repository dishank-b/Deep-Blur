# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from Ops import *

class UNET(object):
	"""docstring for UNET"""
	def __init__(self, arg):
		self.arg = arg

	def set_placeholders(self):
		"""
		"""
		self.images = tf.placeholder(tf.float32, [None, self.arg.h, self.arg.w, 3], name='images')
		self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

	def encoder(self, img, code_length):
		with tf.variable_scope("Encoder") as scope:
			E_conv1 = Conv_2D(img, output_chan=32, use_bn=True,name="E_Conv1")
			E_conv2 = Conv_2D(E_conv1, output_chan=64, use_bn=True ,name="E_Conv2")
			E_conv3 = Conv_2D(E_conv1, output_chan=128, use_bn=True ,name="E_Conv3")
			E_conv4 = Conv_2D(E_conv1, output_chan=512, use_bn=True ,name="E_Conv4")
			E_conv4_r = tf.reshape(E_conv4, shape=[-1, int(np.prod(D_conv2.get_shape()[1:]))])
			E_mean = Dense(D_conv4_r, output_dim=code_length, activation=None,use_bn=False, name="E_mean")
			E_signma = Dense(D_conv4_r, output_dim=code_length, activation=None, use_bn=False, name ="E_sigma")
			return E_mean, E_sigma

	def generator(self,z):
		with tf.variable_scope("Generator") as scope:
			G_linear1 = Dense(x, output_dim=1024, name="G_hidden1")
			G_linear2 = Dense(G_linear1, output_dim=4*4*512, name="G_hidden2")
			G_linear2_r = tf.reshape(G_linear2, shape=[-1,4, 4, 256])
			G_Dconv3 = Dconv_2D(G_linear2_r, output_chan=128, name="G_hidden3")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=128, name="G_hidden4")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=64, name="G_hidden4")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=32, name="G_hidden4")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=16, name="G_hidden4")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=8, name="G_hidden4")
			G_Dconv5 = Dconv_2D(G_Dconv4, output_chan=3, name="G_output")
			return tf.nn.sigmoid(G_Dconv5)

	def discriminator(self, in_layer, layers=4, kernels=64, reuse=False):
		"""PatchGAN Discriminator
		"""
		factor = 1.0;
		with tf.variable_scope('discriminator', reuse=reuse):
			conv1 = Conv_2D(in_layer, output_chan=kernels, use_bn=True,
								 train_phase=self.is_training, name='conv0')

      in_layer = conv1
      for idx in range(1, layers):
         factor = min(2**idx, 8)
			convk = Conv_2D(in_layer, output_chan=kernels*factor, use_bn=True,
								 train_phase=self.is_training, name='conv{}'.format(idx))
      	in_layer = convk

      factor = min(2**num_layers, 8)
		convk = Conv_2D(in_layer, output_chan=kernels*factor, use_bn=True,
							 train_phase=self.is_training, name='conv{}'.format(layers))

		convk = Conv_2D(in_layer, output_chan=1, use_bn=False,
							 train_phase=self.is_training, name='conv{}'.format(layers),
							 activation=None)

		logits = tf.reduce_mean(tf.reshape(convk, [self.arg.batch_size, -1]), axis=1)
      return tf.nn.sigmoid(logits), logits

	def buid_model(self):

		self.d_prob, self.d_logits = self.discriminator(self.images)

	def train_model(self):
