# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np

def Conv_2D(x, output_chan, kernel=[5,5], stride=[2,2],padding="SAME" ,activation=tf.nn.relu, use_bn=True, train_phase=True,add_summary=False,name="Conv_2D"):
	input_shape = x.get_shape()
	kern = [kernel[0], kernel[1], input_shape[-1], output_chan]
	strd = [1, stride[0], stride[1], 1]
	with tf.variable_scope(name) as scope:
		W = tf.get_variable(name="W", shape=kern, initializer=tf.truncated_normal_initializer(stddev=0.01))
		b = tf.get_variable(name="b", shape=[output_chan], initializer=tf.constant_initializer(0.1))

		Conv2D = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=W, strides=strd, padding=padding), b)
		
		if use_bn==True:
			Conv2D = Bn(Conv2D, is_train=train_phase)
			assert activation!=None
			out = activation(Conv2D)

		else:
			if activation!=None:
				out = activation(Conv2D)
			else:
				out = Conv2D
		if add_summary==True:
			weight_summ= tf.summary.histogram(name+"_W", W)
			bias_summ= tf.summary.histogram(name+"_b", b)
			return out, [weight_summ, bias_summ]
		else:
			return out

def Dconv_2D(x, output_chan,batch_size ,kernel=[5,5], stride=[2,2], padding="SAME",activation=tf.nn.relu, use_bn=True, train_phase=True,add_summary=False, name="D_conv2D"):
	input_shape = x.get_shape().as_list()
	kern = [kernel[0], kernel[1], output_chan, input_shape[-1]]
	strd = [1, stride[0], stride[1], 1]
	output_shape = [batch_size,input_shape[1]*strd[1],input_shape[2]*strd[2],output_chan]
	with tf.variable_scope(name) as scope:
		W = tf.get_variable(name="W", shape=kern, initializer=tf.truncated_normal_initializer(stddev=0.01))
		b = tf.get_variable(name="b", shape=[output_chan], initializer=tf.constant_initializer(0.1))

		D_Conv2D = tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape,strides=strd, padding=padding), b)
		
		if use_bn==True:
			D_Conv2D = Bn(D_Conv2D, is_train=train_phase)
			assert activation!=None
			out = activation(D_Conv2D)	

		else:
			if activation!=None:
				out= activation(D_Conv2D)
			else:
				out= D_Conv2D

		if add_summary==True:
			weight_summ= tf.summary.histogram(name+"_W", W)
			bias_summ= tf.summary.histogram(name+"_b", b)
			return out, [weight_summ, bias_summ]
		else:
			return out

def Dense(x, output_dim, use_bn=True, activation=tf.nn.relu, train_phase=True,add_summary=False, name="Dense"):
	input_dim = x.get_shape()[-1]
	with tf.variable_scope(name) as scope:
		W = tf.get_variable('W', shape=[input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
		b = tf.get_variable('b', shape=[output_dim], initializer=tf.constant_initializer(0.1))

		dense = tf.nn.bias_add(tf.matmul(x, W), b)

		if use_bn==True:
			dense = Bn(dense, is_train=train_phase)
			assert activation!=None
			out = activation(dense)
		else:
			if activation!=None:
				out= activation(dense)
			else:
				out = dense

		if add_summary==True:
			return out, [weight_summ, bias_summ]
		else:
			return out

def Bn(x, is_train=True):
	return tf.contrib.layers.batch_norm(x, updates_collections = None,is_training=is_train, 
			center= True, scale=True, reuse=False)

def L_Relu(x, alpha=0.1):
	return tf.maximum(x, alpha*x)

def max_pool(input, kernel=3, stride=2, name=None):
   """Max-pool

   Args:
      input : Input Tensor
      kernel: filter's width (= filter's height)
      stride: stride of the filter
      name  : Optional name for the operation

   Returns:
      Tensor after max-pool operation
   """
   if name is None: 
      name='max_pool'

   with tf.variable_scope(name):
      ksize = [1, kernel, kernel, 1]
      strides = [1, stride, stride, 1]
      output = tf.nn.max_pool(input, ksize=ksize, strides=strides,
         padding='SAME')
      return output
